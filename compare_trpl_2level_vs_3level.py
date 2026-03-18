import math
import re
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
DATA_FILE = "Vacuum_EHBIPOAc_zeonex.txt"
OUTPUT_PREFIX = "compare_trpl_2level_vs_3level"

PLQY_TARGET = 0.87
PLQY_SIGMA = 0.01
SINK_TARGET = 1.0
SINK_SIGMA = 0.01

# Keep this at 0.0 to match your attached 3-level script,
# where 3LE is dynamically present but has no direct radiative channel.
KRL_FIXED_3L = 0.0

RANDOM_SEED = 1
N_GLOBAL_STARTS_3L = 512
N_GLOBAL_STARTS_2L = 512
N_STAGE1_KEEP = 24
STAGE1_MAX_NFEV = 12000
STAGE2_MAX_NFEV = 40000
STAGE1_LOSS = "soft_l1"
STAGE1_F_SCALE = 0.35

MID_TMIN = 3e4
MID_TMAX = 3e6
SAVE_TOP_N_SOLUTIONS = 50

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# Utilities
# ============================================================
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


def stable_linear_system(ts, M, x0):
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


# ============================================================
# 3-level model: 1CT - 3CT - 3LE
# ============================================================
def get_3l_free_spec():
    spec = [
        ("log10_krS", "log10"),
        ("log10_knrS", "log10"),
        ("log10_kISC", "log10"),
        ("log10_kRISC", "log10"),
        ("log10_knrC", "log10"),
        ("log10_kCTLE", "log10"),
        ("log10_kLECT", "log10"),
    ]
    if KRL_FIXED_3L is None:
        spec.append(("log10_krL", "log10"))
    spec.extend(
        [
            ("log10_knrL", "log10"),
            ("ln_scale", "ln"),
            ("ln_B", "ln"),
        ]
    )
    return spec


FREE_SPEC_3L = get_3l_free_spec()
FREE_NAMES_3L = [name for name, _ in FREE_SPEC_3L]

if KRL_FIXED_3L is None:
    LB_3L = np.array(
        [-8.0, -8.0, -8.0, -12.0, -12.0, -12.0, -12.0, -14.0, -12.0, math.log(1e-8), math.log(1e-10)],
        dtype=float,
    )
    UB_3L = np.array(
        [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, math.log(1e12), math.log(1e2)],
        dtype=float,
    )
else:
    LB_3L = np.array(
        [-8.0, -8.0, -8.0, -12.0, -12.0, -12.0, -12.0, -12.0, math.log(1e-8), math.log(1e-10)],
        dtype=float,
    )
    UB_3L = np.array(
        [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, math.log(1e12), math.log(1e2)],
        dtype=float,
    )

DEFAULT_START_BOX_3L = {
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


def rate_matrix_3l(krS, knrS, kISC, kRISC, knrC, kCTLE, kLECT, krL, knrL):
    return np.array(
        [
            [-(krS + knrS + kISC), kRISC, 0.0],
            [kISC, -(kRISC + knrC + kCTLE), kLECT],
            [0.0, kCTLE, -(krL + knrL + kLECT)],
        ],
        dtype=float,
    )


def unpack_3l(p):
    vals = {}
    i = 0
    for name, kind in FREE_SPEC_3L:
        raw = float(p[i])
        i += 1
        if kind == "log10":
            vals[name] = 10.0 ** raw
        elif kind == "ln":
            vals[name] = math.exp(raw)
        else:
            raise ValueError(f"Unknown transform kind: {kind}")

    krL = float(KRL_FIXED_3L) if KRL_FIXED_3L is not None else vals["log10_krL"]

    return {
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


def physical_signal_3l(ts, pars):
    M = rate_matrix_3l(
        pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"],
        pars["kCTLE"], pars["kLECT"], pars["krL"], pars["knrL"],
    )
    pops = stable_linear_system(ts, M, np.array([1.0, 0.0, 0.0], dtype=float))
    if not np.all(np.isfinite(pops)):
        raise ValueError("Population propagation generated non-finite values.")
    min_pop = float(np.min(pops))
    if min_pop < -1e-7:
        raise ValueError(f"Significant negative population encountered: {min_pop:.3e}")
    pops = np.maximum(pops, 0.0)
    S, _, L = pops[:, 0], pops[:, 1], pops[:, 2]
    eS = pars["krS"] * S
    eL = pars["krL"] * L
    iphys = eS + eL
    components = {
        "eS": eS,
        "eL": eL,
    }
    return iphys, pops, components


def yields_3l(pars):
    M = rate_matrix_3l(
        pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"],
        pars["kCTLE"], pars["kLECT"], pars["krL"], pars["knrL"],
    )
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

    phi_rad_S = pars["krS"] * TS
    phi_nr_S = pars["knrS"] * TS
    phi_nr_C = pars["knrC"] * TC
    phi_rad_L = pars["krL"] * TL
    phi_nr_L = pars["knrL"] * TL

    plqy = phi_rad_S + phi_rad_L
    total_sink = phi_rad_S + phi_nr_S + phi_nr_C + phi_rad_L + phi_nr_L

    out = {
        "plqy": plqy,
        "sum_all_sink_yields": total_sink,
        "phi_rad_1CT": phi_rad_S,
        "phi_rad_3LE": phi_rad_L,
        "phi_nr_1CT": phi_nr_S,
        "phi_nr_3CT": phi_nr_C,
        "phi_nr_3LE": phi_nr_L,
        "int_pop_1CT_ns": TS,
        "int_pop_3CT_ns": TC,
        "int_pop_3LE_ns": TL,
    }
    if not np.all(np.isfinite(list(out.values()))):
        return None
    return out


MODEL_3L = {
    "name": "3level_1CT_3CT_3LE",
    "title": "1CT-3CT-3LE",
    "free_spec": FREE_SPEC_3L,
    "free_names": FREE_NAMES_3L,
    "LB": LB_3L,
    "UB": UB_3L,
    "start_box": DEFAULT_START_BOX_3L,
    "unpack": unpack_3l,
    "physical_signal": physical_signal_3l,
    "yields": yields_3l,
}


# ============================================================
# 2-level model: 1CT - 3CT only
# ============================================================
FREE_SPEC_2L = [
    ("log10_krS", "log10"),
    ("log10_knrS", "log10"),
    ("log10_kISC", "log10"),
    ("log10_kRISC", "log10"),
    ("log10_knrC", "log10"),
    ("ln_scale", "ln"),
    ("ln_B", "ln"),
]
FREE_NAMES_2L = [name for name, _ in FREE_SPEC_2L]

LB_2L = np.array(
    [-8.0, -8.0, -8.0, -12.0, -12.0, math.log(1e-8), math.log(1e-10)],
    dtype=float,
)
UB_2L = np.array(
    [1.0, 1.0, 1.0, -1.0, -1.0, math.log(1e12), math.log(1e2)],
    dtype=float,
)

DEFAULT_START_BOX_2L = {
    "log10_krS": (-3.0, -0.2),
    "log10_knrS": (-4.0, -1.0),
    "log10_kISC": (-3.0, -0.2),
    "log10_kRISC": (-9.0, -4.0),
    "log10_knrC": (-9.0, -4.0),
    "ln_scale": None,
    "ln_B": None,
}


def rate_matrix_2l(krS, knrS, kISC, kRISC, knrC):
    return np.array(
        [
            [-(krS + knrS + kISC), kRISC],
            [kISC, -(kRISC + knrC)],
        ],
        dtype=float,
    )


def unpack_2l(p):
    vals = {}
    i = 0
    for name, kind in FREE_SPEC_2L:
        raw = float(p[i])
        i += 1
        if kind == "log10":
            vals[name] = 10.0 ** raw
        elif kind == "ln":
            vals[name] = math.exp(raw)
        else:
            raise ValueError(f"Unknown transform kind: {kind}")

    return {
        "krS": vals["log10_krS"],
        "knrS": vals["log10_knrS"],
        "kISC": vals["log10_kISC"],
        "kRISC": vals["log10_kRISC"],
        "knrC": vals["log10_knrC"],
        "scale": vals["ln_scale"],
        "B": vals["ln_B"],
    }


def physical_signal_2l(ts, pars):
    M = rate_matrix_2l(pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"])
    pops = stable_linear_system(ts, M, np.array([1.0, 0.0], dtype=float))
    if not np.all(np.isfinite(pops)):
        raise ValueError("Population propagation generated non-finite values.")
    min_pop = float(np.min(pops))
    if min_pop < -1e-7:
        raise ValueError(f"Significant negative population encountered: {min_pop:.3e}")
    pops = np.maximum(pops, 0.0)
    S = pops[:, 0]
    eS = pars["krS"] * S
    iphys = eS
    components = {
        "eS": eS,
    }
    return iphys, pops, components


def yields_2l(pars):
    M = rate_matrix_2l(pars["krS"], pars["knrS"], pars["kISC"], pars["kRISC"], pars["knrC"])
    eigvals = np.linalg.eigvals(M)
    if np.any(np.real(eigvals) >= -1e-15):
        return None

    x0 = np.array([1.0, 0.0], dtype=float)
    try:
        tint = -np.linalg.solve(M, x0)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(tint)):
        return None

    TS, TC = [float(v) for v in tint]
    if min(TS, TC) < -1e-10:
        return None

    TS = max(TS, 0.0)
    TC = max(TC, 0.0)

    phi_rad_S = pars["krS"] * TS
    phi_nr_S = pars["knrS"] * TS
    phi_nr_C = pars["knrC"] * TC

    plqy = phi_rad_S
    total_sink = phi_rad_S + phi_nr_S + phi_nr_C

    out = {
        "plqy": plqy,
        "sum_all_sink_yields": total_sink,
        "phi_rad_1CT": phi_rad_S,
        "phi_nr_1CT": phi_nr_S,
        "phi_nr_3CT": phi_nr_C,
        "int_pop_1CT_ns": TS,
        "int_pop_3CT_ns": TC,
    }
    if not np.all(np.isfinite(list(out.values()))):
        return None
    return out


MODEL_2L = {
    "name": "2level_1CT_3CT",
    "title": "1CT-3CT",
    "free_spec": FREE_SPEC_2L,
    "free_names": FREE_NAMES_2L,
    "LB": LB_2L,
    "UB": UB_2L,
    "start_box": DEFAULT_START_BOX_2L,
    "unpack": unpack_2l,
    "physical_signal": physical_signal_2l,
    "yields": yields_2l,
}


# ============================================================
# Generic fitting helpers
# ============================================================
def transformed_to_named_dict(model, p):
    return {name: float(v) for name, v in zip(model["free_names"], p)}


def standardized_residuals_model(p, model, data):
    pars = model["unpack"](p)
    n_trace = len(data["tf_fit"])

    try:
        iphys, _, _ = model["physical_signal"](data["tf_fit"], pars)
    except Exception:
        return np.full(n_trace + 2, 1e6, dtype=float)

    y = pars["B"] + pars["scale"] * iphys
    if np.any(~np.isfinite(y)) or np.any(y <= 0.0):
        return np.full(n_trace + 2, 1e6, dtype=float)

    r_trace = data["w"] * (safe_log(y) - safe_log(data["If_fit"]))

    yld = model["yields"](pars)
    if yld is None:
        return np.full(n_trace + 2, 1e6, dtype=float)

    r_plqy = np.array([(yld["plqy"] - PLQY_TARGET) / PLQY_SIGMA], dtype=float)
    r_sink = np.array([(yld["sum_all_sink_yields"] - SINK_TARGET) / SINK_SIGMA], dtype=float)
    return np.concatenate([r_trace, r_plqy, r_sink])


def candidate_metrics_model(model, p, data):
    r = standardized_residuals_model(p, model, data)
    chi2 = float(np.sum(r ** 2))
    m = len(r)
    k = len(p)
    aic = chi2 + 2.0 * k
    bic = chi2 + k * math.log(m)
    pars = model["unpack"](p)
    yld = model["yields"](pars)
    return {
        "chi2": chi2,
        "aic_relative": aic,
        "bic_relative": bic,
        "n_residuals": m,
        "n_free_params": k,
        "plqy_model": np.nan if yld is None else yld["plqy"],
        "sum_all_sink_yields": np.nan if yld is None else yld["sum_all_sink_yields"],
    }


def make_start_points_model(model, data, n_points):
    lo = model["LB"].copy()
    hi = model["UB"].copy()
    start_box = model["start_box"]

    for i, name in enumerate(model["free_names"]):
        if name in ("ln_scale", "ln_B"):
            continue
        box = start_box.get(name)
        if box is not None:
            lo[i] = max(lo[i], box[0])
            hi[i] = min(hi[i], box[1])

    scale_idx = model["free_names"].index("ln_scale")
    base_idx = model["free_names"].index("ln_B")
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

    starts = np.clip(starts, model["LB"], model["UB"])
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


def run_stage1_model(model, starts, data):
    rows = []
    for i, p0 in enumerate(starts):
        try:
            sol = least_squares(
                standardized_residuals_model,
                p0,
                args=(model, data),
                bounds=(model["LB"], model["UB"]),
                method="trf",
                loss=STAGE1_LOSS,
                f_scale=STAGE1_F_SCALE,
                x_scale="jac",
                max_nfev=STAGE1_MAX_NFEV,
            )
            metrics = candidate_metrics_model(model, sol.x, data)
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


def run_stage2_model(model, seeds, data):
    rows = []
    for i, seed in enumerate(seeds):
        try:
            sol = least_squares(
                standardized_residuals_model,
                seed["x"],
                args=(model, data),
                bounds=(model["LB"], model["UB"]),
                method="trf",
                loss="linear",
                x_scale="jac",
                max_nfev=STAGE2_MAX_NFEV,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            metrics = candidate_metrics_model(model, sol.x, data)
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


def build_candidate_dataframe(model, rows):
    flat = []
    for row in rows[:SAVE_TOP_N_SOLUTIONS]:
        pars = model["unpack"](row["x"])
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
        for k, v in transformed_to_named_dict(model, row["x"]).items():
            one[k] = v
        flat.append(one)
    return pd.DataFrame(flat)


def fit_model(model, data, n_global_starts):
    starts = make_start_points_model(model, data, n_global_starts)
    stage1_rows = run_stage1_model(model, starts, data)
    seeds = stage1_rows[:N_STAGE1_KEEP]
    stage2_rows = run_stage2_model(model, seeds, data)
    if not stage2_rows:
        raise RuntimeError(f"No stage-2 solutions were returned for model {model['name']}")

    valid_stage2 = [
        row for row in stage2_rows
        if row.get('sol') is not None and np.isfinite(row.get('chi2', np.inf))
    ]
    if not valid_stage2:
        raise RuntimeError(f"No successful finite stage-2 solution for model {model['name']}")

    best_row = valid_stage2[0]
    best_x = best_row["x"]
    best_pars = model["unpack"](best_x)

    t_valid = data["tf"][data["valid"]]
    y_data = data["If"][data["valid"]]
    iphys_fit, pops_fit, comps_fit = model["physical_signal"](t_valid, best_pars)
    y_fit = best_pars["B"] + best_pars["scale"] * iphys_fit

    tline = np.logspace(np.log10(t_valid.min()), np.log10(t_valid.max()), 1400)
    iphys_line, pops_line, comps_line = model["physical_signal"](tline, best_pars)
    y_line = best_pars["B"] + best_pars["scale"] * iphys_line

    yld = model["yields"](best_pars)
    metrics = candidate_metrics_model(model, best_x, data)
    metrics["R2_log_overall"] = r2_log(y_data, y_fit, t_valid)
    metrics["R2_linear_overall"] = r2_linear(y_data, y_fit, t_valid)
    metrics["R2_log_mid_window"] = r2_log(y_data, y_fit, t_valid, MID_TMIN, MID_TMAX)

    candidate_df = build_candidate_dataframe(model, stage2_rows)

    return {
        "model": model,
        "stage1_rows": stage1_rows,
        "stage2_rows": stage2_rows,
        "best_row": best_row,
        "best_x": best_x,
        "best_pars": best_pars,
        "yld": yld,
        "metrics": metrics,
        "t_valid": t_valid,
        "y_data": y_data,
        "y_fit": y_fit,
        "iphys_fit": iphys_fit,
        "pops_fit": pops_fit,
        "comps_fit": comps_fit,
        "tline": tline,
        "y_line": y_line,
        "iphys_line": iphys_line,
        "pops_line": pops_line,
        "comps_line": comps_line,
        "candidate_df": candidate_df,
    }


# ============================================================
# Saving outputs
# ============================================================
def save_model_outputs(result):
    model = result["model"]
    slug = model["name"]
    best_pars = result["best_pars"]
    yld = result["yld"]
    metrics = result["metrics"]

    fig_fit = f"{OUTPUT_PREFIX}_{slug}_fit.png"
    plt.figure(figsize=(8.2, 5.3))
    plt.loglog(result["t_valid"], result["y_data"], ".", markersize=2, label="Data")
    plt.loglog(result["tline"], result["y_line"], "-", linewidth=2.2, label=f"Best {model['title']} fit")
    plt.loglog(
        result["tline"],
        best_pars["B"] + best_pars["scale"] * result["comps_line"]["eS"],
        "--",
        linewidth=1.7,
        label="1CT radiative",
    )
    if "eL" in result["comps_line"]:
        plt.loglog(
            result["tline"],
            best_pars["B"] + best_pars["scale"] * result["comps_line"]["eL"],
            ":",
            linewidth=2.0,
            label="3LE radiative",
        )
    plt.xlabel("Time after peak (ns)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Rigorous TRPL fit: {model['title']} model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_fit, dpi=220)
    plt.close()

    fig_mid = f"{OUTPUT_PREFIX}_{slug}_midregion.png"
    mid = (result["t_valid"] >= MID_TMIN) & (result["t_valid"] <= MID_TMAX)
    tline_mid = (result["tline"] >= MID_TMIN) & (result["tline"] <= MID_TMAX)
    plt.figure(figsize=(8.2, 5.3))
    plt.loglog(result["t_valid"][mid], result["y_data"][mid], ".", markersize=3, label="Data")
    plt.loglog(result["tline"][tline_mid], result["y_line"][tline_mid], "-", linewidth=2.2, label=f"Best {model['title']} fit")
    plt.loglog(
        result["tline"][tline_mid],
        (best_pars["B"] + best_pars["scale"] * result["comps_line"]["eS"])[tline_mid],
        "--",
        linewidth=1.7,
        label="1CT radiative",
    )
    if "eL" in result["comps_line"]:
        plt.loglog(
            result["tline"][tline_mid],
            (best_pars["B"] + best_pars["scale"] * result["comps_line"]["eL"])[tline_mid],
            ":",
            linewidth=2.0,
            label="3LE radiative",
        )
    plt.xlabel("Time after peak (ns)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"TRPL fit in the {MID_TMIN:.1e}–{MID_TMAX:.1e} ns region: {model['title']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_mid, dpi=220)
    plt.close()

    curves = {
        "time_ns": result["tline"],
        "measured_fit_total": result["y_line"],
        "baseline": np.full_like(result["tline"], best_pars["B"]),
        "measured_1CT_radiative_component": best_pars["scale"] * result["comps_line"]["eS"],
        "physical_emission_total_per_ns": result["iphys_line"],
        "physical_1CT_radiative_per_ns": result["comps_line"]["eS"],
        "population_1CT": result["pops_line"][:, 0],
        "population_3CT": result["pops_line"][:, 1],
    }
    if result["pops_line"].shape[1] >= 3:
        curves["measured_3LE_radiative_component"] = best_pars["scale"] * result["comps_line"]["eL"]
        curves["physical_3LE_radiative_per_ns"] = result["comps_line"]["eL"]
        curves["population_3LE"] = result["pops_line"][:, 2]

    curves_csv = f"{OUTPUT_PREFIX}_{slug}_curves.csv"
    pd.DataFrame(curves).to_csv(curves_csv, index=False)

    candidate_csv = f"{OUTPUT_PREFIX}_{slug}_candidate_solutions.csv"
    result["candidate_df"].to_csv(candidate_csv, index=False)

    summary_rows = [
        ["meta", "model_name", model["name"]],
        ["meta", "model_title", model["title"]],
        ["meta", "data_file", DATA_FILE],
        ["meta", "PLQY_TARGET", PLQY_TARGET],
        ["meta", "PLQY_SIGMA", PLQY_SIGMA],
        ["meta", "SINK_TARGET", SINK_TARGET],
        ["meta", "SINK_SIGMA", SINK_SIGMA],
        ["meta", "N_GLOBAL_STARTS", N_GLOBAL_STARTS_3L if model['name'] == MODEL_3L['name'] else N_GLOBAL_STARTS_2L],
        ["meta", "N_STAGE1_KEEP", N_STAGE1_KEEP],
        ["fit_quality", "chi2_standardized", metrics["chi2"]],
        ["fit_quality", "AIC_relative", metrics["aic_relative"]],
        ["fit_quality", "BIC_relative", metrics["bic_relative"]],
        ["fit_quality", "n_residuals", metrics["n_residuals"]],
        ["fit_quality", "n_free_params", metrics["n_free_params"]],
        ["fit_quality", "R2_log_overall", metrics["R2_log_overall"]],
        ["fit_quality", "R2_linear_overall", metrics["R2_linear_overall"]],
        ["fit_quality", "R2_log_mid_window", metrics["R2_log_mid_window"]],
        ["fit_quality", "solver_success", bool(result['best_row']['sol'].success) if result['best_row'].get('sol') is not None else False],
        ["fit_quality", "solver_status", int(result['best_row']['sol'].status) if result['best_row'].get('sol') is not None else np.nan],
        ["fit_quality", "solver_message", str(result['best_row']['sol'].message) if result['best_row'].get('sol') is not None else 'not_available'],
    ]
    for pname, pval in best_pars.items():
        summary_rows.append(["best_parameters", pname, pval])
    if yld is not None:
        for key, val in yld.items():
            summary_rows.append(["yields", key, val])

    summary_csv = f"{OUTPUT_PREFIX}_{slug}_summary.csv"
    summary_df = pd.DataFrame(summary_rows, columns=["section", "quantity", "value"])
    summary_df.to_csv(summary_csv, index=False)

    return {
        "fig_fit": fig_fit,
        "fig_mid": fig_mid,
        "curves_csv": curves_csv,
        "candidate_csv": candidate_csv,
        "summary_csv": summary_csv,
        "summary_df": summary_df,
    }


def save_comparison_outputs(res3, res2):
    overlay_fig = f"{OUTPUT_PREFIX}_overlay_fit.png"
    plt.figure(figsize=(8.4, 5.4))
    plt.loglog(res3["t_valid"], res3["y_data"], ".", markersize=2, label="Data")
    plt.loglog(res3["tline"], res3["y_line"], "-", linewidth=2.2, label="3-level best fit")
    plt.loglog(res2["tline"], res2["y_line"], "--", linewidth=2.2, label="2-level best fit")
    plt.xlabel("Time after peak (ns)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("TRPL comparison: 2-level vs 3-level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(overlay_fig, dpi=220)
    plt.close()

    overlay_mid_fig = f"{OUTPUT_PREFIX}_overlay_midregion.png"
    mid3 = (res3["t_valid"] >= MID_TMIN) & (res3["t_valid"] <= MID_TMAX)
    tline3_mid = (res3["tline"] >= MID_TMIN) & (res3["tline"] <= MID_TMAX)
    tline2_mid = (res2["tline"] >= MID_TMIN) & (res2["tline"] <= MID_TMAX)
    plt.figure(figsize=(8.4, 5.4))
    plt.loglog(res3["t_valid"][mid3], res3["y_data"][mid3], ".", markersize=3, label="Data")
    plt.loglog(res3["tline"][tline3_mid], res3["y_line"][tline3_mid], "-", linewidth=2.2, label="3-level best fit")
    plt.loglog(res2["tline"][tline2_mid], res2["y_line"][tline2_mid], "--", linewidth=2.2, label="2-level best fit")
    plt.xlabel("Time after peak (ns)")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Mid-window comparison: {MID_TMIN:.1e}–{MID_TMAX:.1e} ns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(overlay_mid_fig, dpi=220)
    plt.close()

    comparison_rows = []
    for label, res in [("3level", res3), ("2level", res2)]:
        m = res["metrics"]
        comparison_rows.append(
            {
                "model": label,
                "model_title": res["model"]["title"],
                "chi2": m["chi2"],
                "AIC_relative": m["aic_relative"],
                "BIC_relative": m["bic_relative"],
                "n_free_params": m["n_free_params"],
                "R2_log_overall": m["R2_log_overall"],
                "R2_linear_overall": m["R2_linear_overall"],
                "R2_log_mid_window": m["R2_log_mid_window"],
                "plqy_model": m["plqy_model"],
                "sum_all_sink_yields": m["sum_all_sink_yields"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = f"{OUTPUT_PREFIX}_comparison_summary.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    return overlay_fig, overlay_mid_fig, comparison_csv, comparison_df


# ============================================================
# Main
# ============================================================
def main():
    data_path = ensure_path(DATA_FILE)
    data = load_data(data_path)

    print(f"Using data file: {data_path}")
    print("Fitting 3-level model ...")
    res3 = fit_model(MODEL_3L, data, N_GLOBAL_STARTS_3L)
    saved3 = save_model_outputs(res3)

    print("Fitting 2-level model ...")
    res2 = fit_model(MODEL_2L, data, N_GLOBAL_STARTS_2L)
    saved2 = save_model_outputs(res2)

    overlay_fig, overlay_mid_fig, comparison_csv, comparison_df = save_comparison_outputs(res3, res2)

    print(saved3["fig_fit"])
    print(saved3["fig_mid"])
    print(saved3["curves_csv"])
    print(saved3["candidate_csv"])
    print(saved3["summary_csv"])
    print(saved2["fig_fit"])
    print(saved2["fig_mid"])
    print(saved2["curves_csv"])
    print(saved2["candidate_csv"])
    print(saved2["summary_csv"])
    print(overlay_fig)
    print(overlay_mid_fig)
    print(comparison_csv)
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
