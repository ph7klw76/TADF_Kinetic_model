
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
DATA_FILE = "Vacuum_EHBIPOPh_zeonex.txt"
OUTPUT_PREFIX = "rigorous_trpl_1CT_3CT_3LE"

PLQY_TARGET = 0.97
PLQY_SIGMA = 0.01
SINK_TARGET = 1.0
SINK_SIGMA = 0.01

# Set KRL_FIXED = 0.0 to turn off krL * L(t)
# Leave as None to fit krL normally
KRL_FIXED = 0.0

# Search settings
RANDOM_SEED = 1
N_GLOBAL_STARTS = 512          # best as a power of 2 when Sobol is available
N_STAGE1_KEEP = 24
STAGE1_MAX_NFEV = 12000
STAGE2_MAX_NFEV = 40000
STAGE1_LOSS = "soft_l1"
STAGE1_F_SCALE = 0.35

# Diagnostics / expensive extras
SAVE_TOP_N_SOLUTIONS = 50
ENABLE_BOOTSTRAP = False
N_BOOTSTRAP = 100

ENABLE_PROFILE_LIKELIHOOD = True
PROFILE_POINTS = 9
PROFILE_HALF_WIDTH_DECADES = 0.8
PROFILE_SELECTED_PARAMETERS = [
    "log10_kRISC",
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

# Plot window
MID_TMIN = 3e4
MID_TMAX = 3e6


# ============================================================
# Utilities
# ============================================================
rng = np.random.default_rng(RANDOM_SEED)


def safe_log(y):
    return np.log(np.maximum(y, 1e-300))


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


def stable_linear_system(ts, M):
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

if KRL_FIXED is None:
    LB = np.array(
        [
            -8.0, -8.0, -8.0,
            -12.0, -12.0, -12.0, -12.0,
            -14.0,
            -12.0,
            math.log(1e-8),
            math.log(1e-10),
        ],
        dtype=float,
    )
    UB = np.array(
        [
            1.0, 1.0, 1.0,
            -1.0, -1.0, -1.0, -1.0,
            -2.0,
            -1.0,
            math.log(1e12),
            math.log(1e2),
        ],
        dtype=float,
    )
else:
    LB = np.array(
        [
            -8.0, -8.0, -8.0,
            -12.0, -12.0, -12.0, -12.0,
            -12.0,
            math.log(1e-8),
            math.log(1e-10),
        ],
        dtype=float,
    )
    UB = np.array(
        [
            1.0, 1.0, 1.0,
            -1.0, -1.0, -1.0, -1.0,
            -1.0,
            math.log(1e12),
            math.log(1e2),
        ],
        dtype=float,
    )

DEFAULT_START_BOX = {
    "log10_krS": (-3.0, -0.2),
    "log10_knrS": (-4.0, -1.0),
    "log10_kISC": (-3.0, -0.2),
    "log10_kRISC": (-9.0, -4.0),
    "log10_knrC": (-9.0, -4.0),
    "log10_kCTLE": (-9.5, -4.5),
    "log10_kLECT": (-9.5, -4.5),
    "log10_krL": (-12.0, -6.0),
    "log10_knrL": (-10.0, -5.0),
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
        else:
            raise ValueError(f"Unknown transform kind: {kind}")

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


def standardized_residuals(p, data):
    pars = unpack_params(p)
    n_trace = len(data["tf_fit"])

    try:
        iphys, _, _, _ = physical_signal(
            data["tf_fit"],
            pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"],
            pars["kCTLE"], pars["kLECT"], pars["krL"], pars["knrL"],
        )
    except Exception:
        return np.full(n_trace + 2, 1e6, dtype=float)

    y = pars["B"] + pars["scale"] * iphys
    if np.any(~np.isfinite(y)) or np.any(y <= 0.0):
        return np.full(n_trace + 2, 1e6, dtype=float)

    r_trace = data["w"] * (safe_log(y) - safe_log(data["If_fit"]))

    yld = plqy_and_channel_yields(
        pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"],
        pars["kCTLE"], pars["kLECT"], pars["krL"], pars["knrL"],
    )
    if yld is None:
        return np.full(n_trace + 2, 1e6, dtype=float)

    r_plqy = np.array([(yld["plqy"] - PLQY_TARGET) / PLQY_SIGMA], dtype=float)
    r_sink = np.array([(yld["sum_all_sink_yields"] - SINK_TARGET) / SINK_SIGMA], dtype=float)

    return np.concatenate([r_trace, r_plqy, r_sink])


def candidate_metrics(p, data):
    r = standardized_residuals(p, data)
    chi2 = float(np.sum(r**2))
    m = len(r)
    k = len(p)
    aic = chi2 + 2.0 * k
    bic = chi2 + k * math.log(m)

    pars = unpack_params(p)
    yld = plqy_and_channel_yields(
        pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"],
        pars["kCTLE"], pars["kLECT"], pars["krL"], pars["knrL"],
    )
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


def run_stage1(starts, data):
    rows = []
    for i, p0 in enumerate(starts):
        try:
            sol = least_squares(
                standardized_residuals,
                p0,
                args=(data,),
                bounds=(LB, UB),
                method="trf",
                loss=STAGE1_LOSS,
                f_scale=STAGE1_F_SCALE,
                x_scale="jac",
                max_nfev=STAGE1_MAX_NFEV,
            )
            metrics = candidate_metrics(sol.x, data)
            rows.append(
                {
                    "stage": 1,
                    "index": i,
                    "success": bool(sol.success),
                    "status": int(sol.status),
                    "message": str(sol.message),
                    "cost_reported": float(sol.cost),
                    "x": sol.x.copy(),
                    **metrics,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "stage": 1,
                    "index": i,
                    "success": False,
                    "status": -999,
                    "message": f"exception: {exc}",
                    "cost_reported": np.inf,
                    "x": np.asarray(p0, dtype=float).copy(),
                    "chi2": np.inf,
                    "aic_relative": np.inf,
                    "bic_relative": np.inf,
                    "n_residuals": len(data["tf_fit"]) + 2,
                    "n_free_params": len(p0),
                    "plqy_model": np.nan,
                    "sum_all_sink_yields": np.nan,
                }
            )

    rows.sort(key=lambda d: (d["chi2"], d["aic_relative"]))
    rows = unique_candidates(rows)
    return rows


def run_stage2(seeds, data):
    rows = []
    for i, seed in enumerate(seeds):
        try:
            sol = least_squares(
                standardized_residuals,
                seed["x"],
                args=(data,),
                bounds=(LB, UB),
                method="trf",
                loss="linear",
                x_scale="jac",
                max_nfev=STAGE2_MAX_NFEV,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            metrics = candidate_metrics(sol.x, data)
            rows.append(
                {
                    "stage": 2,
                    "index": i,
                    "seed_index": seed["index"],
                    "success": bool(sol.success),
                    "status": int(sol.status),
                    "message": str(sol.message),
                    "cost_reported": float(sol.cost),
                    "x": sol.x.copy(),
                    "sol": sol,
                    **metrics,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "stage": 2,
                    "index": i,
                    "seed_index": seed["index"],
                    "success": False,
                    "status": -999,
                    "message": f"exception: {exc}",
                    "cost_reported": np.inf,
                    "x": seed["x"].copy(),
                    "sol": None,
                    "chi2": np.inf,
                    "aic_relative": np.inf,
                    "bic_relative": np.inf,
                    "n_residuals": len(data["tf_fit"]) + 2,
                    "n_free_params": len(seed["x"]),
                    "plqy_model": np.nan,
                    "sum_all_sink_yields": np.nan,
                }
            )
    rows.sort(key=lambda d: (d["aic_relative"], d["chi2"]))
    rows = unique_candidates(rows)
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


def profile_likelihood(best_row, data):
    if not ENABLE_PROFILE_LIKELIHOOD:
        return None

    target_names = selected_profile_names()
    if not target_names:
        return None

    best_x = best_row["x"]
    out_rows = []
    k = len(best_x)

    for pname in target_names:
        j = FREE_NAMES.index(pname)
        width = PROFILE_HALF_WIDTH_DECADES if pname.startswith("log10_") else PROFILE_HALF_WIDTH_DECADES * math.log(10.0)
        lo = max(LB[j], best_x[j] - width)
        hi = min(UB[j], best_x[j] + width)
        grid = np.linspace(lo, hi, PROFILE_POINTS)

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

        for val in grid:
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
                out_rows.append({
                    "parameter": pname,
                    "fixed_value_transformed": float(val),
                    "chi2": metrics["chi2"],
                    "aic_relative": metrics["aic_relative"],
                    "bic_relative": metrics["bic_relative"],
                })
            except Exception:
                out_rows.append({
                    "parameter": pname,
                    "fixed_value_transformed": float(val),
                    "chi2": np.inf,
                    "aic_relative": np.inf,
                    "bic_relative": np.inf,
                })

    if not out_rows:
        return None
    prof = pd.DataFrame(out_rows)
    prof["delta_chi2"] = prof.groupby("parameter")["chi2"].transform(lambda s: s - np.nanmin(s.values))
    return prof


def bootstrap_fits(best_row, data):
    if not ENABLE_BOOTSTRAP:
        return None

    best_x = best_row["x"]
    pars = unpack_params(best_x)
    iphys, _, _, _ = physical_signal(
        data["tf_fit"],
        pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"],
        pars["kCTLE"], pars["kLECT"], pars["krL"], pars["knrL"],
    )
    y_fit = pars["B"] + pars["scale"] * iphys
    log_resid = safe_log(data["If_fit"]) - safe_log(y_fit)

    rows = []
    for b in range(N_BOOTSTRAP):
        draw = rng.choice(log_resid, size=len(log_resid), replace=True)
        synth_data = dict(data)
        synth_data["If_fit"] = np.exp(safe_log(y_fit) + draw)

        try:
            sol = least_squares(
                standardized_residuals,
                best_x,
                args=(synth_data,),
                bounds=(LB, UB),
                method="trf",
                loss="linear",
                x_scale="jac",
                max_nfev=max(10000, STAGE2_MAX_NFEV // 2),
            )
            pars_b = unpack_params(sol.x)
            row = {"bootstrap_index": b, "chi2": candidate_metrics(sol.x, synth_data)["chi2"]}
            row.update(pars_b)
            rows.append(row)
        except Exception:
            continue

    if not rows:
        return None
    return pd.DataFrame(rows)


# ============================================================
# Main workflow
# ============================================================
def main():
    data_path = ensure_path(DATA_FILE)
    data = load_data(data_path)

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
    iphys_fit, pops_fit, eS_fit, eL_fit = physical_signal(
        t_valid,
        best_pars["krS"], best_pars["knrS"], best_pars["kISC"], best_pars["kRISC"], best_pars["knrC"],
        best_pars["kCTLE"], best_pars["kLECT"], best_pars["krL"], best_pars["knrL"],
    )
    y_fit = best_pars["B"] + best_pars["scale"] * iphys_fit

    tline = np.logspace(np.log10(t_valid.min()), np.log10(t_valid.max()), 1400)
    iphys_line, pops_line, eS_line, eL_line = physical_signal(
        tline,
        best_pars["krS"], best_pars["knrS"], best_pars["kISC"], best_pars["kRISC"], best_pars["knrC"],
        best_pars["kCTLE"], best_pars["kLECT"], best_pars["krL"], best_pars["knrL"],
    )
    y_line = best_pars["B"] + best_pars["scale"] * iphys_line

    yld = plqy_and_channel_yields(
        best_pars["krS"], best_pars["knrS"], best_pars["kISC"], best_pars["kRISC"], best_pars["knrC"],
        best_pars["kCTLE"], best_pars["kLECT"], best_pars["krL"], best_pars["knrL"],
    )
    metrics = candidate_metrics(best_x, data)
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
    summary_rows = [
        ["meta", "data_file", str(data_path)],
        ["meta", "KRL_FIXED", "fit" if KRL_FIXED is None else float(KRL_FIXED)],
        ["meta", "N_GLOBAL_STARTS", N_GLOBAL_STARTS],
        ["meta", "N_STAGE1_KEEP", N_STAGE1_KEEP],
        ["meta", "STAGE1_LOSS", STAGE1_LOSS],
        ["meta", "PLQY_TARGET", PLQY_TARGET],
        ["meta", "PLQY_SIGMA", PLQY_SIGMA],
        ["meta", "SINK_TARGET", SINK_TARGET],
        ["meta", "SINK_SIGMA", SINK_SIGMA],
        ["fit_quality", "chi2_standardized", metrics["chi2"]],
        ["fit_quality", "AIC_relative", metrics["aic_relative"]],
        ["fit_quality", "BIC_relative", metrics["bic_relative"]],
        ["fit_quality", "n_residuals", metrics["n_residuals"]],
        ["fit_quality", "n_free_params", metrics["n_free_params"]],
        ["fit_quality", "R2_log_overall", r2_log(y_data, y_fit, t_valid)],
        ["fit_quality", "R2_linear_overall", r2_linear(y_data, y_fit, t_valid)],
        ["fit_quality", "R2_log_mid_window", r2_log(y_data, y_fit, t_valid, MID_TMIN, MID_TMAX)],
        ["fit_quality", "solver_success", bool(best_sol.success)],
        ["fit_quality", "solver_status", int(best_sol.status)],
        ["fit_quality", "solver_message", str(best_sol.message)],
        ["identifiability", "cond_JTJ", ident_diag["cond_JTJ"]],
        ["identifiability", "rank_J", ident_diag["rank_J"]],
        ["identifiability", "n_params", ident_diag["n_params"]],
        ["identifiability", "min_singular_value", ident_diag["min_singular_value"]],
        ["identifiability", "max_singular_value", ident_diag["max_singular_value"]],
        ["identifiability", "identifiability_note", ident_diag["identifiability_note"]],
        ["identifiability", "skip_uncertainty", ident_diag["skip_uncertainty"]],
        ["identifiability", "max_abs_parameter_correlation", max_abs_offdiag(corr)],
        ["identifiability", "n_stage2_candidates", len(stage2_rows)],
        ["identifiability", "n_candidates_deltaAIC_lt_2", int(np.sum(candidate_df["delta_aic"] < 2.0)) if "delta_aic" in candidate_df else np.nan],
        ["identifiability", "n_uncertainty_rows_flagged", int(np.sum(uncertainty_df["uncertainty_note"] != "ok")) if "uncertainty_note" in uncertainty_df else np.nan],
        ["identifiability", "high_correlation_flag", bool(np.isfinite(max_abs_offdiag(corr)) and max_abs_offdiag(corr) >= CORR_FLAG_ABS)],
    ]

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
