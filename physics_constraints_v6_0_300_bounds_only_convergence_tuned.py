import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.linalg import expm
from scipy.optimize import least_squares

try:
    from scipy.stats import qmc
    HAVE_QMC = True
except Exception:
    HAVE_QMC = False

# ============================================================
# User settings
# ============================================================
DATA_FILE = "EHBIPOAc_zeonex_300.txt"
OUTPUT_PREFIX = "rigorous_trpl_1CT_3CT_3LE"

PLQY_TARGET = 0.87
ENABLE_KNRL_GEQ_KRL_CONSTRAINT = True
# Plot window
MID_TMIN = 4e4
MID_TMAX = 1e7
PLQY_SIGMA = 0.01
SINK_TARGET = 1.0
SINK_SIGMA = 0.01

# Set KRL_FIXED = 0.0 to turn off krL * L(t)
# Leave as None to fit krL normally
KRL_FIXED = 0.0

# Search settings
RANDOM_SEED = 1
N_GLOBAL_STARTS = 2048         # MODIFIED: denser Sobol exploration of a still-degenerate landscape
N_STAGE1_KEEP = 64             # MODIFIED: keep more distinct basins for stage-2 refinement
STAGE1_MAX_NFEV = 12000
STAGE2_MAX_NFEV = 40000
STAGE1_LOSS = "soft_l1"
STAGE1_F_SCALE = 0.35

# Parallelism settings
# Use None to auto-pick max(1, cpu_count() - 1)
N_WORKERS = 16
PARALLEL_BACKEND = "loky"    # robust process-based backend in joblib
JOBLIB_VERBOSE = 0
JOBLIB_BATCH_SIZE = 1
LIMIT_INNER_NUM_THREADS = True

# Diagnostics / expensive extras
SAVE_TOP_N_SOLUTIONS = 50
ENABLE_BOOTSTRAP = True
N_BOOTSTRAP = 100

ENABLE_PROFILE_LIKELIHOOD = True
PROFILE_POINTS = 9
PROFILE_HALF_WIDTH_DECADES = 0.8
PROFILE_SELECTED_PARAMETERS = [
    "log10_kRISC",
    "log10_knrC",
    "log10_kCTLE",
    "log10_kLECT",
    "log10_krL",
    "log10_knrL",
]

# Identifiability safeguards
COND_WARN = 1e12
COND_SKIP_UNCERTAINTY = 1e16
MAX_LOCAL_SIGMA_TRANSFORM = 50.0   # beyond this, local uncertainty is practically unbounded
CORR_FLAG_ABS = 0.98


# Mid-window R^2 penalty settings
R2_MID_TARGET = 0.98
R2_MID_SIGMA = 0.02
R2_MID_HARD_FAIL = False

# Physics-based ordering / ratio constraints
ENABLE_PHYSICS_CONSTRAINTS = True
PHYSICS_HARD_FAIL = False
PHYSICS_PENALTY_SCALE = 1.0
MAX_LOG10_RATIO_LECT_CTLE = 0.0   # enforce kLECT / kCTLE <= 1
MAX_LOG10_RATIO_RISC_ISC = 0.0    # enforce kRISC <= kISC
MIN_LOG10_RATIO_KNRL_KRL = 0.0    # when enabled and krL is fitted/nonzero, enforce knrL >= krL
ENABLE_KNRS_TO_KNRL_RANGE_CONSTRAINT = False
MIN_LOG10_RATIO_KNRS_KNRL = 1.0   # enforce knrS / knrL >= 10
MAX_LOG10_RATIO_KNRS_KNRL = 5.0   # enforce knrS / knrL <= 10000

# Optional prompt-lifetime constraint on the intrinsic S1 lifetime
ENABLE_PROMPT_S1_LIFETIME_CONSTRAINT = True
PROMPT_S1_TAU_MIN_NS = 1.0
PROMPT_S1_TAU_MAX_NS = 100.0

# Optional radiative hierarchy: enforce krS >> krL
ENABLE_RADIATIVE_HIERARCHY_CONSTRAINT = True
MIN_LOG10_RATIO_KRS_KRL = 3.0   # enforce krS / krL >= 1e4

# Optional intrinsic T1 radiative lifetime constraint
# Typical T1 radiative lifetime window: 10 ms to 10 s
ENABLE_T1_RADIATIVE_LIFETIME_CONSTRAINT = True
T1_RADIATIVE_TAU_MIN_NS = 1.0e7
T1_RADIATIVE_TAU_MAX_NS = 1.0e10

# Optional tail-lifetime constraint from the long-lived component
ENABLE_TAIL_LIFETIME_CONSTRAINT = True
TAIL_TAU_TARGET_NS = None       # if None, estimate from the data tail window below
TAIL_TAU_MATCH_TOL_DECADES = 1.0
TAIL_FIT_TMIN = MID_TMAX
TAIL_FIT_TMAX = None
TAIL_LAST_FRACTION = 0.05

# ============================================================
# Utilities
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)
RATE_NAMES = ("krS", "knrS", "kISC", "kRISC", "knrC", "kCTLE", "kLECT", "krL", "knrL")
INF_RESIDUAL = 1e6


def safe_log(y):
    return np.log(np.maximum(y, 1e-300))


def rate_args(pars):
    return [pars[name] for name in RATE_NAMES]


def signal_from_pars(ts, pars):
    return physical_signal(ts, *rate_args(pars))


def yields_from_pars(pars):
    return plqy_and_channel_yields(*rate_args(pars))

def requested_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1) if N_WORKERS is None else max(1, int(N_WORKERS))

def effective_n_workers(n_tasks: int) -> int:
    requested = requested_worker_count()
    return max(1, min(requested, max(1, int(n_tasks))))

def run_parallel(func, iterable, n_jobs):
    if n_jobs <= 1:
        return [func(item) for item in iterable]

    config_kwargs = {
        "backend": PARALLEL_BACKEND,
        "n_jobs": n_jobs,
        "verbose": JOBLIB_VERBOSE,
    }
    if LIMIT_INNER_NUM_THREADS:
        config_kwargs["inner_max_num_threads"] = 1

    with parallel_config(**config_kwargs):
        return Parallel(batch_size=JOBLIB_BATCH_SIZE)(delayed(func)(item) for item in iterable)

def ensure_path(data_file: str) -> Path:
    p1 = Path(data_file)
    if p1.exists():
        return p1
    here = Path(__file__).resolve().parent
    p2 = here / data_file
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"Could not find {data_file!r}. Put the data file in the current working "
        f"directory or in the same folder as this script."
    )

def load_data(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = re.split(r"[\s,;\t]+", line.strip())
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except Exception:
                    pass
    if not rows:
        raise ValueError(f"No numeric two-column data could be read from {path}")

    t = np.array([r[0] for r in rows], dtype=float)
    I = np.array([r[1] for r in rows], dtype=float)

    order = np.argsort(t)
    t = t[order]
    I = I[order]

    t0 = float(t[np.argmax(I)])
    mask = t >= t0
    tf = t[mask] - t0
    If = I[mask]

    B_tail = float(np.median(If[int(0.98 * len(If)):]))
    sel = (tf > 0) & np.isfinite(If) & (If > 0)
    tf_fit = tf[sel]
    If_fit = If[sel]

    logt = np.log10(tf_fit)
    nbins = 90
    bins = np.linspace(logt.min(), logt.max(), nbins)
    idx = np.clip(np.digitize(logt, bins) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    w = 1.0 / np.sqrt(np.maximum(counts[idx], 1))

    valid = (tf > 0) & np.isfinite(If) & (If > 0)
    return {
        "t": t,
        "I": I,
        "t0": t0,
        "tf": tf,
        "If": If,
        "B_tail": B_tail,
        "tf_fit": tf_fit,
        "If_fit": If_fit,
        "w": w,
        "valid": valid,
    }

# ============================================================
# Model
# ============================================================
def rate_matrix(krS, knrS, kISC, kRISC, knrC, kCTLE, kLECT, krL, knrL):
    return np.array(
        [
            [-(krS + knrS + kISC),                 kRISC,                    0.0],
            [ kISC,                       -(kRISC + knrC + kCTLE),         kLECT],
            [ 0.0,                                 kCTLE,          -(krL + knrL + kLECT)],
        ],
        dtype=float,
    )

def stable_linear_system_cpu(ts, M):
    """
    Propagate x(t) = exp(M t) x0 with x0 = [1, 0, 0]^T.

    Uses eigendecomposition when the eigenvector matrix is well-conditioned.
    Falls back to expm otherwise.
    """
    x0 = np.array([1.0, 0.0, 0.0], dtype=float)
    vals = np.linalg.eigvals(M)
    if np.any(np.real(vals) >= -1e-15):
        raise ValueError("Rate matrix has non-decaying mode(s).")

    try:
        evals, V = np.linalg.eig(M)
        condV = np.linalg.cond(V)
        if not np.isfinite(condV) or condV > 1e10:
            raise np.linalg.LinAlgError("Ill-conditioned eigenvectors.")
        Vinv = np.linalg.inv(V)
        coeff = Vinv @ x0
        expv = np.exp(np.outer(ts, evals))
        X = (expv * coeff) @ V.T
        X = np.real_if_close(X, tol=1000)
        X = np.asarray(X, dtype=float)
    except Exception:
        X = np.vstack([expm(M * ti) @ x0 for ti in ts])

    return X

def stable_linear_system(ts, M):
    return stable_linear_system_cpu(ts, M)

def physical_signal(ts, krS, knrS, kISC, kRISC, knrC, kCTLE, kLECT, krL, knrL):
    M = rate_matrix(krS, knrS, kISC, kRISC, knrC, kCTLE, kLECT, krL, knrL)
    pops = stable_linear_system(ts, M)
    if not np.all(np.isfinite(pops)):
        raise ValueError("Population propagation generated non-finite values.")

    min_pop = float(np.min(pops))
    if min_pop < -1e-7:
        raise ValueError(f"Significant negative population encountered: {min_pop:.3e}")

    pops = np.maximum(pops, 0.0)
    S, _, L = pops[:, 0], pops[:, 1], pops[:, 2]
    eS = krS * S
    eL = krL * L
    iphys = eS + eL
    return iphys, pops, eS, eL

def plqy_and_channel_yields(krS, knrS, kISC, kRISC, knrC, kCTLE, kLECT, krL, knrL):
    M = rate_matrix(krS, knrS, kISC, kRISC, knrC, kCTLE, kLECT, krL, knrL)
    eigvals = np.linalg.eigvals(M)
    if np.any(np.real(eigvals) >= -1e-15):
        return None

    x0 = np.array([1.0, 0.0, 0.0], dtype=float)
    try:
        tint = -np.linalg.solve(M, x0)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(tint)):
        return None

    TS, TC, TL = [float(v) for v in tint]
    if min(TS, TC, TL) < -1e-10:
        return None

    TS = max(TS, 0.0)
    TC = max(TC, 0.0)
    TL = max(TL, 0.0)

    phi_rad_S = krS * TS
    phi_nr_S = knrS * TS
    phi_nr_C = knrC * TC
    phi_rad_L = krL * TL
    phi_nr_L = knrL * TL

    plqy = phi_rad_S + phi_rad_L
    total_sink = phi_rad_S + phi_nr_S + phi_nr_C + phi_rad_L + phi_nr_L

    out = {
        "plqy": plqy,
        "phi_rad_1CT": phi_rad_S,
        "phi_rad_3LE": phi_rad_L,
        "phi_nr_1CT": phi_nr_S,
        "phi_nr_3CT": phi_nr_C,
        "phi_nr_3LE": phi_nr_L,
        "sum_all_sink_yields": total_sink,
        "int_pop_1CT_ns": TS,
        "int_pop_3CT_ns": TC,
        "int_pop_3LE_ns": TL,
    }
    if not np.all(np.isfinite(list(out.values()))):
        return None
    return out

# ============================================================
# Parameterization
# ============================================================
def get_free_parameter_spec():
    spec = [
        ("log10_krS", "log10"),
        ("log10_knrS", "log10"),
        ("log10_kISC", "log10"),
        ("log10_kRISC", "log10"),
        ("log10_knrC", "log10"),
        ("log10_kCTLE", "log10"),
        ("log10_kLECT", "log10"),
    ]
    if KRL_FIXED is None:
        spec.append(("log10_krL", "log10"))
    spec.extend(
        [
            ("log10_knrL", "log10"),
            ("ln_scale", "ln"),
            ("ln_B", "ln"),
        ]
    )
    return spec

FREE_SPEC = get_free_parameter_spec()
FREE_NAMES = [name for name, _ in FREE_SPEC]

PARAM_BOUNDS = {
    "log10_krS": (
        -2.50,  # MODIFIED: trim unused very-slow side
        -1.25,  # MODIFIED: widen fast side because current optimum sits on the upper bound
    ),
    "log10_knrS": (
        -6.40,  # MODIFIED: tighten weakly identifiable broad tail
        -5.00,  # MODIFIED
    ),
    "log10_kISC": (
        -1.42,  # MODIFIED: narrow around the bootstrap-supported basin
        -1.15,  # MODIFIED
    ),
    "log10_kRISC": (
        -4.35,  # MODIFIED: narrow around the profile minimum
        -3.85,  # MODIFIED
    ),
    "log10_knrC": (
        -7.60,  # MODIFIED: remove the extremely flat far-slow region
        -5.60,  # MODIFIED: still keeps the broad profile-supported band
    ),
    "log10_kCTLE": (
        -5.65,  # MODIFIED: shift emphasis to the slower profile-supported side
        -4.95,  # MODIFIED: trim the fast side that worsens chi2
    ),
    "log10_kLECT": (
        -6.35,  # MODIFIED: focus on the identifiable valley
        -5.55,  # MODIFIED
    ),
    "log10_krL": (
        -11.20, # unchanged; inactive when KRL_FIXED = 0.0
        -6.20,  # unchanged; inactive when KRL_FIXED = 0.0
    ),
    "log10_knrL": (
        -6.15,  # MODIFIED: narrow around the bootstrap/profile-supported region
        -5.35,  # MODIFIED
    ),
    "ln_scale": (
        math.log(1e-4) - 0.5 * math.log(10.0),
        math.log(1e12) + 0.5 * math.log(10.0),
    ),
    "ln_B": (
        math.log(1e-10) - 0.5 * math.log(10.0),
        math.log(1e2) + 0.5 * math.log(10.0),
    ),
}


def build_bounds_from_free_names():
    missing = [name for name in FREE_NAMES if name not in PARAM_BOUNDS]
    if missing:
        raise KeyError(f"Missing bounds for parameters: {missing}")

    lb = np.array([PARAM_BOUNDS[name][0] for name in FREE_NAMES], dtype=float)
    ub = np.array([PARAM_BOUNDS[name][1] for name in FREE_NAMES], dtype=float)
    if lb.shape != ub.shape or np.any(~np.isfinite(lb)) or np.any(~np.isfinite(ub)):
        raise ValueError("Parameter bounds are malformed or non-finite.")
    if np.any(lb >= ub):
        bad = [(FREE_NAMES[i], float(lb[i]), float(ub[i])) for i in range(len(FREE_NAMES)) if lb[i] >= ub[i]]
        raise ValueError(f"Parameter bounds are not consistent a < b: {bad}")
    return lb, ub


# hard bounds
# built from the active free-parameter list so they stay consistent when KRL_FIXED is changed
LB, UB = build_bounds_from_free_names()

DEFAULT_START_BOX = {
    "log10_krS":  (-2.05, -1.35),  # MODIFIED
    "log10_knrS": (-6.10, -5.15),  # MODIFIED
    "log10_kISC": (-1.38, -1.20),  # MODIFIED
    "log10_kRISC":(-4.22, -3.92),  # MODIFIED
    "log10_knrC": (-7.20, -5.90),  # MODIFIED
    "log10_kCTLE":(-5.55, -5.10),  # MODIFIED
    "log10_kLECT":(-6.20, -5.70),  # MODIFIED
    "log10_krL":  (-10.80, -6.35), # unchanged; inactive when KRL_FIXED = 0.0
    "log10_knrL": (-6.00, -5.45),  # MODIFIED
    "ln_scale": None,
    "ln_B": None,
}

def unpack_params(p):
    vals = {}
    i = 0
    for name, kind in FREE_SPEC:
        raw = float(p[i])
        i += 1
        if kind == "log10":
            vals[name] = 10.0 ** raw
        elif kind == "ln":
            vals[name] = math.exp(raw)

    krL = float(KRL_FIXED) if KRL_FIXED is not None else vals["log10_krL"]

    out = {
        "krS": vals["log10_krS"],
        "knrS": vals["log10_knrS"],
        "kISC": vals["log10_kISC"],
        "kRISC": vals["log10_kRISC"],
        "knrC": vals["log10_knrC"],
        "kCTLE": vals["log10_kCTLE"],
        "kLECT": vals["log10_kLECT"],
        "krL": krL,
        "knrL": vals["log10_knrL"],
        "scale": vals["ln_scale"],
        "B": vals["ln_B"],
    }
    return out

def transformed_to_named_dict(p):
    return {name: float(v) for name, v in zip(FREE_NAMES, p)}


def positive_krL_possible():
    return KRL_FIXED is None or float(KRL_FIXED) > 0.0


def penalty(delta):
    return max(0.0, float(delta)) / PHYSICS_PENALTY_SCALE


def active_tail_target(data=None):
    if data is None:
        return np.nan
    val = float(data.get("tail_tau_target_ns", np.nan))
    return val if np.isfinite(val) and val > 0.0 else np.nan


def active_physics_constraint_names(data=None):
    if not ENABLE_PHYSICS_CONSTRAINTS:
        return []
    names = ["risc_le_isc", "lect_le_ctle"]
    if ENABLE_KNRL_GEQ_KRL_CONSTRAINT and positive_krL_possible():
        names.append("knrl_ge_krl")
    if ENABLE_KNRS_TO_KNRL_RANGE_CONSTRAINT:
        names.extend(["knrs_knrl_low", "knrs_knrl_high"])
    if ENABLE_PROMPT_S1_LIFETIME_CONSTRAINT:
        names.extend(["prompt_tau_low", "prompt_tau_high"])
    if ENABLE_RADIATIVE_HIERARCHY_CONSTRAINT and positive_krL_possible():
        names.append("krs_gg_krl")
    if ENABLE_T1_RADIATIVE_LIFETIME_CONSTRAINT and positive_krL_possible():
        names.extend(["t1_rad_tau_low", "t1_rad_tau_high"])
    if ENABLE_TAIL_LIFETIME_CONSTRAINT and np.isfinite(active_tail_target(data)):
        names.append("tail_tau_match")
    return names


def s1_prompt_lifetime_ns(pars):
    total_loss = float(pars["krS"] + pars["knrS"] + pars["kISC"])
    if not np.isfinite(total_loss) or total_loss <= 0.0:
        return np.nan
    return 1.0 / total_loss

def t1_radiative_lifetime_ns(pars):
    krL = float(pars["krL"])
    if not np.isfinite(krL) or krL <= 0.0:
        return np.nan
    return 1.0 / krL

def slowest_mode_lifetime_ns(pars):
    M = rate_matrix(
        pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"],
        pars["kCTLE"], pars["kLECT"], pars["krL"], pars["knrL"],
    )
    eigvals = np.linalg.eigvals(M)
    lam = float(np.max(np.real(eigvals)))
    if not np.isfinite(lam) or lam >= 0.0:
        return np.nan
    return -1.0 / lam

def estimate_tail_lifetime_from_data(data):
    if TAIL_TAU_TARGET_NS is not None:
        try:
            val = float(TAIL_TAU_TARGET_NS)
            return val if np.isfinite(val) and val > 0.0 else np.nan
        except Exception:
            return np.nan

    t = np.asarray(data["tf_fit"], dtype=float)
    y = np.asarray(data["If_fit"], dtype=float)
    m = np.isfinite(t) & np.isfinite(y) & (t > 0.0) & (y > 0.0)
    if TAIL_FIT_TMIN is not None:
        m &= t >= float(TAIL_FIT_TMIN)
    if TAIL_FIT_TMAX is not None:
        m &= t <= float(TAIL_FIT_TMAX)

    if int(np.count_nonzero(m)) < 5:
        n = len(t)
        if n < 5:
            return np.nan
        start = max(0, int(math.floor((1.0 - float(TAIL_LAST_FRACTION)) * n)))
        m = np.zeros(n, dtype=bool)
        m[start:] = True
        m &= np.isfinite(t) & np.isfinite(y) & (t > 0.0) & (y > 0.0)

    if int(np.count_nonzero(m)) < 5:
        return np.nan

    tt = t[m]
    ly = np.log(y[m])
    try:
        slope, _ = np.polyfit(tt, ly, 1)
    except Exception:
        return np.nan

    if not np.isfinite(slope) or slope >= 0.0:
        return np.nan
    return -1.0 / float(slope)

def expected_physics_constraint_count(data=None):
    return len(active_physics_constraint_names(data))


def physics_constraint_residuals(pars, data=None):
    if not ENABLE_PHYSICS_CONSTRAINTS:
        return np.zeros(0, dtype=float), False

    penalties = []
    for name in active_physics_constraint_names(data):
        if name == "risc_le_isc":
            penalties.append(penalty(math.log10(pars["kRISC"]) - math.log10(pars["kISC"]) - MAX_LOG10_RATIO_RISC_ISC))
        elif name == "lect_le_ctle":
            penalties.append(penalty(math.log10(pars["kLECT"]) - math.log10(pars["kCTLE"]) - MAX_LOG10_RATIO_LECT_CTLE))
        elif name == "knrl_ge_krl":
            penalties.append(penalty(math.log10(pars["krL"]) - math.log10(pars["knrL"]) + MIN_LOG10_RATIO_KNRL_KRL))
        elif name == "knrs_knrl_low":
            log_ratio = math.log10(pars["knrS"]) - math.log10(pars["knrL"])
            penalties.append(penalty(MIN_LOG10_RATIO_KNRS_KNRL - log_ratio))
        elif name == "knrs_knrl_high":
            log_ratio = math.log10(pars["knrS"]) - math.log10(pars["knrL"])
            penalties.append(penalty(log_ratio - MAX_LOG10_RATIO_KNRS_KNRL))
        elif name in {"prompt_tau_low", "prompt_tau_high"}:
            tau_prompt = s1_prompt_lifetime_ns(pars)
            if not np.isfinite(tau_prompt) or tau_prompt <= 0.0:
                penalties.append(INF_RESIDUAL)
            else:
                log_tau = math.log10(tau_prompt)
                bound = math.log10(PROMPT_S1_TAU_MIN_NS) if name.endswith("low") else math.log10(PROMPT_S1_TAU_MAX_NS)
                penalties.append(penalty(bound - log_tau) if name.endswith("low") else penalty(log_tau - bound))
        elif name == "krs_gg_krl":
            penalties.append(penalty(MIN_LOG10_RATIO_KRS_KRL - (math.log10(pars["krS"]) - math.log10(pars["krL"]))))
        elif name in {"t1_rad_tau_low", "t1_rad_tau_high"}:
            tau_t1_rad = t1_radiative_lifetime_ns(pars)
            if not np.isfinite(tau_t1_rad) or tau_t1_rad <= 0.0:
                penalties.append(INF_RESIDUAL)
            else:
                log_tau = math.log10(tau_t1_rad)
                bound = math.log10(T1_RADIATIVE_TAU_MIN_NS) if name.endswith("low") else math.log10(T1_RADIATIVE_TAU_MAX_NS)
                penalties.append(penalty(bound - log_tau) if name.endswith("low") else penalty(log_tau - bound))
        elif name == "tail_tau_match":
            tau_model = slowest_mode_lifetime_ns(pars)
            tau_target = active_tail_target(data)
            if not np.isfinite(tau_model) or tau_model <= 0.0:
                penalties.append(INF_RESIDUAL)
            else:
                penalties.append(penalty(abs(math.log10(tau_model) - math.log10(tau_target)) - TAIL_TAU_MATCH_TOL_DECADES))

    arr = np.asarray(penalties, dtype=float)
    return arr, bool(PHYSICS_HARD_FAIL and np.any(arr > 0.0))


def standardized_residuals(p, data):
    pars = unpack_params(p)
    n_trace = len(data["tf_fit"])
    r_phys, hard_fail = physics_constraint_residuals(pars, data)
    n_extra = 3 + len(r_phys)  # PLQY, sink, mid-window R^2, plus physics constraints

    if hard_fail:
        return np.full(n_trace + n_extra, INF_RESIDUAL, dtype=float)

    try:
        iphys, _, _, _ = signal_from_pars(data["tf_fit"], pars)
    except Exception:
        return np.full(n_trace + n_extra, INF_RESIDUAL, dtype=float)

    y = pars["B"] + pars["scale"] * iphys
    if np.any(~np.isfinite(y)) or np.any(y <= 0.0):
        return np.full(n_trace + n_extra, INF_RESIDUAL, dtype=float)

    r_trace = data["w"] * (safe_log(y) - safe_log(data["If_fit"]))

    yld = yields_from_pars(pars)
    if yld is None:
        return np.full(n_trace + n_extra, INF_RESIDUAL, dtype=float)

    r_plqy = np.array([(yld["plqy"] - PLQY_TARGET) / PLQY_SIGMA], dtype=float)
    r_sink = np.array([(yld["sum_all_sink_yields"] - SINK_TARGET) / SINK_SIGMA], dtype=float)

    r2_mid = r2_log(
        y=data["If_fit"],
        yp=y,
        tvals=data["tf_fit"],
        lo=MID_TMIN,
        hi=MID_TMAX,
    )

    if not np.isfinite(r2_mid):
        if R2_MID_HARD_FAIL:
            return np.full(n_trace + n_extra, INF_RESIDUAL, dtype=float)
        r_mid = np.array([1e3], dtype=float)
    else:
        shortfall = max(0.0, R2_MID_TARGET - r2_mid)
        r_mid = np.array([shortfall / R2_MID_SIGMA], dtype=float)

    return np.concatenate([r_trace, r_plqy, r_sink, r_mid, r_phys])

def candidate_metrics(p, data):
    r = standardized_residuals(p, data)
    chi2 = float(np.sum(r**2))
    m = len(r)
    k = len(p)
    aic = chi2 + 2.0 * k
    bic = chi2 + k * math.log(m)

    pars = unpack_params(p)
    yld = yields_from_pars(pars)
    return {
        "chi2": chi2,
        "aic_relative": aic,
        "bic_relative": bic,
        "n_residuals": m,
        "n_free_params": k,
        "plqy_model": np.nan if yld is None else yld["plqy"],
        "sum_all_sink_yields": np.nan if yld is None else yld["sum_all_sink_yields"],
    }

def make_start_points(data, n_points):
    lo = LB.copy()
    hi = UB.copy()

    for i, name in enumerate(FREE_NAMES):
        if name in ("ln_scale", "ln_B"):
            continue
        box = DEFAULT_START_BOX.get(name)
        if box is not None:
            lo[i] = max(lo[i], box[0])
            hi[i] = min(hi[i], box[1])

    scale_idx = FREE_NAMES.index("ln_scale")
    base_idx = FREE_NAMES.index("ln_B")
    lo[scale_idx] = max(lo[scale_idx], math.log(max(np.max(data["If_fit"]) * 1e-3, 1e-12)))
    hi[scale_idx] = min(hi[scale_idx], math.log(max(np.max(data["If_fit"]) * 1e3, 1e-6)))
    lo[base_idx] = max(lo[base_idx], math.log(max(data["B_tail"] * 1e-2, 1e-12)))
    hi[base_idx] = min(hi[base_idx], math.log(max(data["B_tail"] * 1e2, 1e-10)))

    for i, name in enumerate(FREE_NAMES):
        if not np.isfinite(lo[i]) or not np.isfinite(hi[i]):
            lo[i], hi[i] = LB[i], UB[i]
        if lo[i] >= hi[i]:
            lo[i], hi[i] = LB[i], UB[i]
        if lo[i] >= hi[i]:
            raise ValueError(
                f"Start-point bounds collapsed for {name}: lo={lo[i]!r}, hi={hi[i]!r}, "
                f"global_lo={LB[i]!r}, global_hi={UB[i]!r}"
            )

    if HAVE_QMC:
        m = int(math.ceil(math.log2(max(2, n_points))))
        sampler = qmc.Sobol(d=len(lo), scramble=True, seed=RANDOM_SEED)
        U = sampler.random_base2(m=m)[:n_points]
        starts = qmc.scale(U, lo, hi)
    else:
        U = rng.random((n_points, len(lo)))
        starts = lo + (hi - lo) * U

    starts = np.clip(starts, LB, UB)
    return starts

def unique_candidates(rows, atol=5e-3):
    kept = []
    xs = []
    for row in rows:
        x = row["x"]
        if not xs:
            kept.append(row)
            xs.append(x)
            continue
        if all(np.linalg.norm(x - y) > atol for y in xs):
            kept.append(row)
            xs.append(x)
    return kept


def run_standard_fit(x0, data, *, loss, max_nfev, **kwargs):
    return least_squares(
        standardized_residuals,
        x0,
        args=(data,),
        bounds=(LB, UB),
        method="trf",
        loss=loss,
        x_scale="jac",
        max_nfev=max_nfev,
        **kwargs,
    )


def failure_row(stage, index, x, data, message, **extra):
    return {
        "stage": stage,
        "index": index,
        "success": False,
        "status": -999,
        "message": message,
        "cost_reported": np.inf,
        "x": np.asarray(x, dtype=float).copy(),
        "chi2": np.inf,
        "aic_relative": np.inf,
        "bic_relative": np.inf,
        "n_free_params": len(x),
        "plqy_model": np.nan,
        "sum_all_sink_yields": np.nan,
        **extra,
    }


def stage1_one(task):
    i, p0, data = task
    try:
        sol = run_standard_fit(p0, data, loss=STAGE1_LOSS, max_nfev=STAGE1_MAX_NFEV, f_scale=STAGE1_F_SCALE)
        return {
            "stage": 1,
            "index": i,
            "success": bool(sol.success),
            "status": int(sol.status),
            "message": str(sol.message),
            "cost_reported": float(sol.cost),
            "x": sol.x.copy(),
            **candidate_metrics(sol.x, data),
        }
    except Exception as exc:
        return failure_row(1, i, p0, data, f"exception: {exc}")

def run_stage1(starts, data):
    tasks = [(i, np.asarray(p0, dtype=float), data) for i, p0 in enumerate(starts)]
    rows = run_parallel(stage1_one, tasks, effective_n_workers(len(tasks)))
    rows.sort(key=lambda d: (d["chi2"], d["aic_relative"]))
    rows = unique_candidates(rows)
    return rows

def stage2_one(task):
    i, seed, data = task
    try:
        sol = run_standard_fit(
            seed["x"],
            data,
            loss="linear",
            max_nfev=STAGE2_MAX_NFEV,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        return {
            "stage": 2,
            "index": i,
            "seed_index": seed["index"],
            "success": bool(sol.success),
            "status": int(sol.status),
            "message": str(sol.message),
            "cost_reported": float(sol.cost),
            "x": sol.x.copy(),
            "sol": sol,
            **candidate_metrics(sol.x, data),
        }
    except Exception as exc:
        return failure_row(2, i, seed["x"], data, f"exception: {exc}", seed_index=seed["index"], sol=None)

def run_stage2(seeds, data):
    tasks = [(i, seed, data) for i, seed in enumerate(seeds)]
    rows = run_parallel(stage2_one, tasks, effective_n_workers(len(tasks)))
    rows = unique_candidates(rows)
    
    ok_rows = [r for r in rows if r.get("success", False)]
    bad_rows = [r for r in rows if not r.get("success", False)]
    
    ok_rows.sort(key=lambda d: (d["aic_relative"], d["chi2"]))
    bad_rows.sort(key=lambda d: (d["aic_relative"], d["chi2"]))
    
    rows = ok_rows + bad_rows
    
    if rows:
        best_aic = rows[0]["aic_relative"]
        for row in rows:
            row["delta_aic"] = float(row["aic_relative"] - best_aic)
    return rows

# ============================================================
# Diagnostics
# ============================================================
def r2_log(y, yp, tvals=None, lo=None, hi=None):
    m = np.isfinite(y) & np.isfinite(yp) & (y > 0) & (yp > 0)
    if tvals is not None:
        if lo is not None:
            m &= tvals >= lo
        if hi is not None:
            m &= tvals <= hi
    ly = np.log10(y[m])
    lp = np.log10(yp[m])
    denom = np.sum((ly - np.mean(ly)) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((ly - lp) ** 2) / denom

def r2_linear(y, yp, tvals=None, lo=None, hi=None):
    m = np.isfinite(y) & np.isfinite(yp)
    if tvals is not None:
        if lo is not None:
            m &= tvals >= lo
        if hi is not None:
            m &= tvals <= hi
    yy = y[m]
    ff = yp[m]
    denom = np.sum((yy - np.mean(yy)) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((yy - ff) ** 2) / denom

def jacobian_diagnostics(sol):
    diag = {
        "cond_JTJ": np.nan,
        "rank_J": np.nan,
        "n_params": np.nan,
        "min_singular_value": np.nan,
        "max_singular_value": np.nan,
        "identifiability_note": "not_available",
        "skip_uncertainty": True,
    }
    if sol is None or not hasattr(sol, "jac") or sol.jac is None:
        return diag, None, None

    try:
        J = np.asarray(sol.jac, dtype=float)
        u, s, vt = np.linalg.svd(J, full_matrices=False)
        JTJ = J.T @ J
        cond = float(np.linalg.cond(JTJ))
        rank = int(np.linalg.matrix_rank(J))
        smax = float(np.max(s)) if len(s) else np.nan
        smin = float(np.min(s)) if len(s) else np.nan

        diag["cond_JTJ"] = cond
        diag["rank_J"] = rank
        diag["n_params"] = J.shape[1]
        diag["min_singular_value"] = smin
        diag["max_singular_value"] = smax

        if not np.isfinite(cond):
            diag["identifiability_note"] = "ill_conditioned_nonfinite"
            diag["skip_uncertainty"] = True
            return diag, None, None

        if rank < J.shape[1]:
            diag["identifiability_note"] = "rank_deficient"
            diag["skip_uncertainty"] = True
            return diag, None, None

        if cond >= COND_SKIP_UNCERTAINTY:
            diag["identifiability_note"] = "condition_too_large_skip_uncertainty"
            diag["skip_uncertainty"] = True
            return diag, None, None

        if cond >= COND_WARN:
            diag["identifiability_note"] = "condition_large_warn"
        else:
            diag["identifiability_note"] = "ok"

        cov = np.linalg.pinv(JTJ)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        d = np.sqrt(np.maximum(np.diag(cov), 1e-300))
        corr = cov / np.outer(d, d)
        diag["skip_uncertainty"] = False
        return diag, se, corr
    except Exception:
        diag["identifiability_note"] = "jacobian_diagnostics_failed"
        diag["skip_uncertainty"] = True
        return diag, None, None

def max_abs_offdiag(corr):
    if corr is None:
        return np.nan
    c = np.asarray(corr, dtype=float).copy()
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        return np.nan
    np.fill_diagonal(c, np.nan)
    with np.errstate(invalid="ignore"):
        return float(np.nanmax(np.abs(c)))

def parameter_uncertainty_rows(best_row, se, ident_diag):
    rows = []
    best_x = best_row["x"]
    best_pars = unpack_params(best_x)

    if se is None or ident_diag.get("skip_uncertainty", True):
        note = ident_diag.get("identifiability_note", "uncertainty_not_available")
        for name in FREE_NAMES:
            if name.startswith("log10_"):
                phys_name = name.replace("log10_", "")
                phys_value = best_pars[phys_name]
                rows.append([phys_name, phys_value, np.nan, np.nan, np.nan, note])
            elif name.startswith("ln_"):
                phys_name = name.replace("ln_", "")
                phys_value = best_pars[phys_name]
                rows.append([phys_name, phys_value, np.nan, np.nan, np.nan, note])
        return rows

    log10_float_max = math.log10(sys.float_info.max)
    ln_float_max = math.log(sys.float_info.max)

    for i, name in enumerate(FREE_NAMES):
        sigma_t = float(se[i])

        if not np.isfinite(sigma_t):
            factor_hi = np.inf
            factor_lo = 0.0
            note = "non_finite_local_se"
        else:
            sigma_t = abs(sigma_t)
            note = "ok"
            if sigma_t > MAX_LOCAL_SIGMA_TRANSFORM:
                factor_hi = np.inf
                factor_lo = 0.0
                note = "practically_unbounded_local_se"
            elif name.startswith("log10_"):
                if sigma_t > log10_float_max:
                    factor_hi = np.inf
                    factor_lo = 0.0
                    note = "unidentifiable_overflow_capped"
                else:
                    factor_hi = 10.0 ** sigma_t
                    factor_lo = 10.0 ** (-sigma_t)
            elif name.startswith("ln_"):
                if sigma_t > ln_float_max:
                    factor_hi = np.inf
                    factor_lo = 0.0
                    note = "unidentifiable_overflow_capped"
                else:
                    factor_hi = math.exp(sigma_t)
                    factor_lo = math.exp(-sigma_t)
            else:
                factor_hi = np.nan
                factor_lo = np.nan
                note = "unsupported_parameterization"

        if name.startswith("log10_"):
            phys_name = name.replace("log10_", "")
            phys_value = best_pars[phys_name]
            rows.append([phys_name, phys_value, sigma_t, factor_hi, factor_lo, note])
        elif name.startswith("ln_"):
            phys_name = name.replace("ln_", "")
            phys_value = best_pars[phys_name]
            rows.append([phys_name, phys_value, sigma_t, factor_hi, factor_lo, note])
    return rows

def build_candidate_dataframe(rows):
    flat = []
    for row in rows[:SAVE_TOP_N_SOLUTIONS]:
        pars = unpack_params(row["x"])
        one = {
            "stage": row["stage"],
            "index": row["index"],
            "success": row["success"],
            "status": row["status"],
            "chi2": row["chi2"],
            "aic_relative": row["aic_relative"],
            "bic_relative": row["bic_relative"],
            "delta_aic": row.get("delta_aic", np.nan),
            "plqy_model": row["plqy_model"],
            "sum_all_sink_yields": row["sum_all_sink_yields"],
        }
        for k, v in pars.items():
            one[k] = v
        for k, v in transformed_to_named_dict(row["x"]).items():
            one[k] = v
        flat.append(one)
    return pd.DataFrame(flat)

def selected_profile_names():
    names = []
    for pname in PROFILE_SELECTED_PARAMETERS:
        if pname in FREE_NAMES:
            names.append(pname)
    return names

def profile_point_one(task):
    pname, j, val, best_x, data = task
    k = len(best_x)
    free_mask = np.ones(k, dtype=bool)
    free_mask[j] = False
    lb_free = LB[free_mask]
    ub_free = UB[free_mask]
    x_seed_free = best_x[free_mask].copy()

    def resid_fixed(x_free, fixed_value, fixed_idx):
        x_full = np.zeros(k, dtype=float)
        x_full[free_mask] = x_free
        x_full[fixed_idx] = fixed_value
        return standardized_residuals(x_full, data)

    try:
        sol = least_squares(
            resid_fixed,
            x_seed_free,
            args=(float(val), j),
            bounds=(lb_free, ub_free),
            method="trf",
            loss="linear",
            x_scale="jac",
            max_nfev=max(8000, STAGE2_MAX_NFEV // 2),
        )
        x_full = np.zeros(k, dtype=float)
        x_full[free_mask] = sol.x
        x_full[j] = val
        metrics = candidate_metrics(x_full, data)
        return {
            "parameter": pname,
            "fixed_value_transformed": float(val),
            "chi2": metrics["chi2"],
            "aic_relative": metrics["aic_relative"],
            "bic_relative": metrics["bic_relative"],
        }
    except Exception:
        return {
            "parameter": pname,
            "fixed_value_transformed": float(val),
            "chi2": np.inf,
            "aic_relative": np.inf,
            "bic_relative": np.inf,
        }

def profile_likelihood(best_row, data):
    if not ENABLE_PROFILE_LIKELIHOOD:
        return None

    target_names = selected_profile_names()
    if not target_names:
        return None

    best_x = best_row["x"]
    tasks = []

    for pname in target_names:
        j = FREE_NAMES.index(pname)
        width = PROFILE_HALF_WIDTH_DECADES if pname.startswith("log10_") else PROFILE_HALF_WIDTH_DECADES * math.log(10.0)
        lo = max(LB[j], best_x[j] - width)
        hi = min(UB[j], best_x[j] + width)
        grid = np.linspace(lo, hi, PROFILE_POINTS)
        for val in grid:
            tasks.append((pname, j, float(val), best_x, data))

    out_rows = run_parallel(profile_point_one, tasks, effective_n_workers(len(tasks)))
    if not out_rows:
        return None
    prof = pd.DataFrame(out_rows)
    prof["delta_chi2"] = prof.groupby("parameter")["chi2"].transform(lambda s: s - np.nanmin(s.values))
    return prof

def bootstrap_one(task):
    b, best_x, y_fit, log_resid, data = task
    local_rng = np.random.default_rng(RANDOM_SEED + 100000 + int(b))
    draw = local_rng.choice(log_resid, size=len(log_resid), replace=True)
    synth_data = dict(data)
    synth_data["If_fit"] = np.exp(safe_log(y_fit) + draw)

    try:
        sol = run_standard_fit(best_x, synth_data, loss="linear", max_nfev=max(10000, STAGE2_MAX_NFEV // 2))
        pars_b = unpack_params(sol.x)
        row = {"bootstrap_index": b, "chi2": candidate_metrics(sol.x, synth_data)["chi2"]}
        row.update(pars_b)
        return row
    except Exception:
        return None

def bootstrap_fits(best_row, data):
    if not ENABLE_BOOTSTRAP:
        return None

    best_x = best_row["x"]
    pars = unpack_params(best_x)
    iphys, _, _, _ = signal_from_pars(data["tf_fit"], pars)
    y_fit = pars["B"] + pars["scale"] * iphys
    log_resid = safe_log(data["If_fit"]) - safe_log(y_fit)

    tasks = [(b, best_x, y_fit, log_resid, data) for b in range(N_BOOTSTRAP)]
    rows = run_parallel(bootstrap_one, tasks, effective_n_workers(len(tasks)))
    rows = [row for row in rows if row is not None]
    if not rows:
        return None
    return pd.DataFrame(rows)

def add_summary_rows(summary_rows, section, mapping):
    summary_rows.extend([[section, key, value] for key, value in mapping])


# ============================================================
# Main workflow
# ============================================================
def main():
    data_path = ensure_path(DATA_FILE)
    data = load_data(data_path)
    data["tail_tau_target_ns"] = estimate_tail_lifetime_from_data(data)

    starts = make_start_points(data, N_GLOBAL_STARTS)
    stage1_rows = run_stage1(starts, data)
    seeds = stage1_rows[:N_STAGE1_KEEP]
    stage2_rows = run_stage2(seeds, data)

    if not stage2_rows:
        raise RuntimeError("No successful stage-2 solution was found.")

    best_row = stage2_rows[0]
    best_sol = best_row["sol"]
    best_x = best_row["x"]
    best_pars = unpack_params(best_x)

    t_valid = data["tf"][data["valid"]]
    y_data = data["If"][data["valid"]]
    iphys_fit, pops_fit, eS_fit, eL_fit = signal_from_pars(t_valid, best_pars)
    y_fit = best_pars["B"] + best_pars["scale"] * iphys_fit

    tline = np.logspace(np.log10(t_valid.min()), np.log10(t_valid.max()), 1400)
    iphys_line, pops_line, eS_line, eL_line = signal_from_pars(tline, best_pars)
    y_line = best_pars["B"] + best_pars["scale"] * iphys_line

    yld = yields_from_pars(best_pars)
    metrics = candidate_metrics(best_x, data)
    r2_log_mid_value = r2_log(y_data, y_fit, t_valid, MID_TMIN, MID_TMAX)
    ident_diag, se, corr = jacobian_diagnostics(best_sol)
    unc_rows = parameter_uncertainty_rows(best_row, se, ident_diag)

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------
    fig1 = f"{OUTPUT_PREFIX}_fit.png"
    plt.figure(figsize=(8.2, 5.3))
    plt.loglog(t_valid, y_data, ".", markersize=2, label="Data")
    plt.loglog(tline, y_line, "-", linewidth=2.2, label="Best rigorous fit")
    plt.loglog(tline, best_pars["B"] + best_pars["scale"] * eS_line, "--", linewidth=1.7, label="1CT radiative")
    plt.loglog(tline, best_pars["B"] + best_pars["scale"] * eL_line, ":", linewidth=2.0, label="3LE radiative")
    plt.xlabel("Time after peak (ns)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Rigorous TRPL fit: 1CT-3CT-3LE model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=220)
    plt.close()

    fig2 = f"{OUTPUT_PREFIX}_midregion.png"
    mid = (t_valid >= MID_TMIN) & (t_valid <= MID_TMAX)
    tline_mid = (tline >= MID_TMIN) & (tline <= MID_TMAX)
    plt.figure(figsize=(8.2, 5.3))
    plt.loglog(t_valid[mid], y_data[mid], ".", markersize=3, label="Data")
    plt.loglog(tline[tline_mid], y_line[tline_mid], "-", linewidth=2.2, label="Best rigorous fit")
    plt.loglog(tline[tline_mid], (best_pars["B"] + best_pars["scale"] * eS_line)[tline_mid], "--", linewidth=1.7, label="1CT radiative")
    plt.loglog(tline[tline_mid], (best_pars["B"] + best_pars["scale"] * eL_line)[tline_mid], ":", linewidth=2.0, label="3LE radiative")
    plt.xlabel("Time after peak (ns)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Rigorous TRPL fit in the {MID_TMIN:.1e}–{MID_TMAX:.1e} ns region")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=220)
    plt.close()

    # --------------------------------------------------------
    # Save curves
    # --------------------------------------------------------
    curve_csv = f"{OUTPUT_PREFIX}_curves.csv"
    curve_df = pd.DataFrame(
        {
            "time_ns": tline,
            "measured_fit_total": y_line,
            "baseline": np.full_like(tline, best_pars["B"]),
            "measured_1CT_radiative_component": best_pars["scale"] * eS_line,
            "measured_3LE_radiative_component": best_pars["scale"] * eL_line,
            "physical_emission_total_per_ns": iphys_line,
            "physical_1CT_radiative_per_ns": eS_line,
            "physical_3LE_radiative_per_ns": eL_line,
            "population_1CT": pops_line[:, 0],
            "population_3CT": pops_line[:, 1],
            "population_3LE": pops_line[:, 2],
        }
    )
    curve_df.to_csv(curve_csv, index=False)

    # --------------------------------------------------------
    # Save candidate table
    # --------------------------------------------------------
    candidate_csv = f"{OUTPUT_PREFIX}_candidate_solutions.csv"
    candidate_df = build_candidate_dataframe(stage2_rows)
    candidate_df.to_csv(candidate_csv, index=False)

    # --------------------------------------------------------
    # Save uncertainty / correlation
    # --------------------------------------------------------
    uncertainty_csv = f"{OUTPUT_PREFIX}_parameter_uncertainty.csv"
    uncertainty_df = pd.DataFrame(
        unc_rows,
        columns=[
            "parameter",
            "best_value",
            "sigma_transformed_space",
            "x_sigma_factor",
            "divide_by_factor_for_minus_1sigma",
            "uncertainty_note",
        ],
    )
    uncertainty_df.to_csv(uncertainty_csv, index=False)

    corr_csv = f"{OUTPUT_PREFIX}_parameter_correlation.csv"
    if corr is None:
        corr_df = pd.DataFrame({"message": ["Correlation matrix not reported because the local Jacobian was too ill-conditioned or rank-deficient."]})
    else:
        corr_df = pd.DataFrame(corr, index=FREE_NAMES, columns=FREE_NAMES)
    corr_df.to_csv(corr_csv, index=True)

    # --------------------------------------------------------
    # Optional expensive extras
    # --------------------------------------------------------
    profile_csv = None
    profile_df = profile_likelihood(best_row, data)
    if profile_df is not None:
        profile_csv = f"{OUTPUT_PREFIX}_profile_likelihood.csv"
        profile_df.to_csv(profile_csv, index=False)

    bootstrap_csv = None
    bootstrap_df = bootstrap_fits(best_row, data)
    if bootstrap_df is not None:
        bootstrap_csv = f"{OUTPUT_PREFIX}_bootstrap.csv"
        bootstrap_df.to_csv(bootstrap_csv, index=False)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------
    workers_used_stage1 = effective_n_workers(len(starts))
    workers_used_stage2 = effective_n_workers(len(seeds))
    summary_rows = []
    add_summary_rows(summary_rows, "meta", [
        ("data_file", str(data_path)),
        ("KRL_FIXED", "fit" if KRL_FIXED is None else float(KRL_FIXED)),
        ("N_GLOBAL_STARTS", N_GLOBAL_STARTS),
        ("N_STAGE1_KEEP", N_STAGE1_KEEP),
        ("STAGE1_LOSS", STAGE1_LOSS),
        ("PLQY_TARGET", PLQY_TARGET),
        ("PLQY_SIGMA", PLQY_SIGMA),
        ("SINK_TARGET", SINK_TARGET),
        ("SINK_SIGMA", SINK_SIGMA),
        ("R2_MID_TARGET", R2_MID_TARGET),
        ("R2_MID_SIGMA", R2_MID_SIGMA),
        ("R2_MID_HARD_FAIL", R2_MID_HARD_FAIL),
    ])
    add_summary_rows(summary_rows, "physics", [
        ("enable_constraints", ENABLE_PHYSICS_CONSTRAINTS),
        ("hard_fail", PHYSICS_HARD_FAIL),
        ("penalty_scale_decades", PHYSICS_PENALTY_SCALE),
        ("max_log10_ratio_lect_ctle", MAX_LOG10_RATIO_LECT_CTLE),
        ("max_log10_ratio_risc_isc", MAX_LOG10_RATIO_RISC_ISC),
        ("enable_knrl_geq_krl_constraint", ENABLE_KNRL_GEQ_KRL_CONSTRAINT),
        ("min_log10_ratio_knrl_krl", MIN_LOG10_RATIO_KNRL_KRL),
        ("enable_knrs_to_knrl_range_constraint", ENABLE_KNRS_TO_KNRL_RANGE_CONSTRAINT),
        ("min_log10_ratio_knrs_knrl", MIN_LOG10_RATIO_KNRS_KNRL),
        ("max_log10_ratio_knrs_knrl", MAX_LOG10_RATIO_KNRS_KNRL),
        ("enable_prompt_s1_lifetime_constraint", ENABLE_PROMPT_S1_LIFETIME_CONSTRAINT),
        ("prompt_s1_tau_min_ns", PROMPT_S1_TAU_MIN_NS),
        ("prompt_s1_tau_max_ns", PROMPT_S1_TAU_MAX_NS),
        ("enable_radiative_hierarchy_constraint", ENABLE_RADIATIVE_HIERARCHY_CONSTRAINT),
        ("min_log10_ratio_krs_krl", MIN_LOG10_RATIO_KRS_KRL),
        ("enable_t1_radiative_lifetime_constraint", ENABLE_T1_RADIATIVE_LIFETIME_CONSTRAINT),
        ("knrC_fitted", True),
        ("kLECT_fitted", True),
        ("t1_radiative_tau_min_ns", T1_RADIATIVE_TAU_MIN_NS),
        ("t1_radiative_tau_max_ns", T1_RADIATIVE_TAU_MAX_NS),
        ("enable_tail_lifetime_constraint", ENABLE_TAIL_LIFETIME_CONSTRAINT),
        ("tail_tau_target_ns_input", TAIL_TAU_TARGET_NS),
        ("tail_tau_target_ns_used", data.get("tail_tau_target_ns", np.nan)),
        ("tail_tau_match_tol_decades", TAIL_TAU_MATCH_TOL_DECADES),
        ("tail_fit_tmin", TAIL_FIT_TMIN),
        ("tail_fit_tmax", TAIL_FIT_TMAX),
        ("tail_last_fraction", TAIL_LAST_FRACTION),
    ])
    add_summary_rows(summary_rows, "parallel", [
        ("joblib_backend", PARALLEL_BACKEND),
        ("workers_used_stage1", workers_used_stage1),
        ("workers_used_stage2", workers_used_stage2),
        ("limit_inner_num_threads", LIMIT_INNER_NUM_THREADS),
    ])
    add_summary_rows(summary_rows, "fit_quality", [
        ("chi2_standardized", metrics["chi2"]),
        ("AIC_relative", metrics["aic_relative"]),
        ("BIC_relative", metrics["bic_relative"]),
        ("n_residuals", metrics["n_residuals"]),
        ("n_free_params", metrics["n_free_params"]),
        ("R2_log_overall", r2_log(y_data, y_fit, t_valid)),
        ("R2_linear_overall", r2_linear(y_data, y_fit, t_valid)),
        ("R2_log_mid_window", r2_log_mid_value),
        ("R2_mid_penalty_residual", max(0.0, R2_MID_TARGET - r2_log_mid_value) / R2_MID_SIGMA if np.isfinite(r2_log_mid_value) else np.nan),
        ("solver_success", bool(best_sol.success)),
        ("solver_status", int(best_sol.status)),
        ("solver_message", str(best_sol.message)),
    ])
    max_corr = max_abs_offdiag(corr)
    add_summary_rows(summary_rows, "identifiability", [
        ("cond_JTJ", ident_diag["cond_JTJ"]),
        ("rank_J", ident_diag["rank_J"]),
        ("n_params", ident_diag["n_params"]),
        ("min_singular_value", ident_diag["min_singular_value"]),
        ("max_singular_value", ident_diag["max_singular_value"]),
        ("identifiability_note", ident_diag["identifiability_note"]),
        ("skip_uncertainty", ident_diag["skip_uncertainty"]),
        ("max_abs_parameter_correlation", max_corr),
        ("n_stage2_candidates", len(stage2_rows)),
        ("n_candidates_deltaAIC_lt_2", int(np.sum(candidate_df["delta_aic"] < 2.0)) if "delta_aic" in candidate_df else np.nan),
        ("n_uncertainty_rows_flagged", int(np.sum(uncertainty_df["uncertainty_note"] != "ok")) if "uncertainty_note" in uncertainty_df else np.nan),
        ("high_correlation_flag", bool(np.isfinite(max_corr) and max_corr >= CORR_FLAG_ABS)),
    ])

    for pname, pval in best_pars.items():
        summary_rows.append(["best_parameters", pname, pval])

    if yld is not None:
        for key, val in yld.items():
            summary_rows.append(["yields", key, val])

    summary_csv = f"{OUTPUT_PREFIX}_summary.csv"
    summary_df = pd.DataFrame(summary_rows, columns=["section", "quantity", "value"])
    summary_df.to_csv(summary_csv, index=False)

    print(fig1)
    print(fig2)
    print(curve_csv)
    print(candidate_csv)
    print(uncertainty_csv)
    print(corr_csv)
    if profile_csv is not None:
        print(profile_csv)
    if bootstrap_csv is not None:
        print(bootstrap_csv)
    print(summary_csv)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
