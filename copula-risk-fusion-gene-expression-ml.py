# Primary METABRIC copula-risk fusion analysis

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from pathlib import Path

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from scipy import stats

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed; XGBClassifier will be skipped.")


# ============================================================
# GPU CONFIG (Colab-ready). Auto-detects an NVIDIA GPU and routes
# XGBoost to CUDA. Set USE_GPU_XGB = False to force CPU if needed.
# Only XGBoost is GPU-accelerated; RandomForest / GradientBoosting /
# permutation importance run multi-core on CPU via n_jobs=-1.
# ============================================================
USE_GPU_XGB = True
import shutil as _shutil, subprocess as _subprocess
def _detect_gpu():
    if not USE_GPU_XGB or _shutil.which("nvidia-smi") is None:
        return False
    try:
        return _subprocess.run(["nvidia-smi"],
                               stdout=_subprocess.DEVNULL,
                               stderr=_subprocess.DEVNULL).returncode == 0
    except Exception:
        return False
_USE_GPU = _detect_gpu()
_XGB_KW = {"tree_method": "hist", "device": "cuda"} if _USE_GPU else {"tree_method": "hist"}
print(f"[config] XGBoost device: {'cuda (GPU)' if _USE_GPU else 'cpu (hist)'}")
STRICT_INPUT_CHECKS = True  # hard-stop guards against truncated / wrong input data


def _pkg_versions():
    import sklearn, scipy
    v = {"numpy": np.__version__, "pandas": pd.__version__,
         "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    try:
        import xgboost; v["xgboost"] = xgboost.__version__
    except Exception:
        v["xgboost"] = "not installed"
    try:
        import lifelines; v["lifelines"] = lifelines.__version__
    except Exception:
        v["lifelines"] = "not installed"
    return v


def xgb_actual_device(pipe):
    """Read the device the XGBoost booster actually trained on (config JSON),
    so a CUDA request that silently fell back to CPU is detected."""
    try:
        clf = pipe.named_steps.get("clf")
        cfg = json.loads(clf.get_booster().save_config())
        return cfg.get("learner", {}).get("generic_param", {}).get("device", "unknown")
    except Exception as e:
        return f"unknown ({e})"
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("lifelines not installed; KM curves will be skipped.")


# ======================================================================
# 1.  DATA LOADING & ENDPOINT
# ======================================================================

def load_metabric(csv_path="METABRIC_RNA_Mutation.csv"):
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded METABRIC: {df.shape[0]} patients, {df.shape[1]} columns")
    if STRICT_INPUT_CHECKS:
        assert df.shape == (1904, 693), (
            f"METABRIC shape {df.shape} != (1904, 693); the input file looks "
            f"truncated or wrong. Fix the data or set STRICT_INPUT_CHECKS = False.")
    return df


def build_5year_endpoint(df,
                         time_col="overall_survival_months",
                         status_col_cancer="death_from_cancer",
                         status_col_overall="overall_survival",
                         cutoff_months=60.0):
    """
    Binary endpoint: cancer-specific death within cutoff_months.

    Categories in death_from_cancer:
      'Died of Disease'    → event (status=1)
      'Living'             → alive (status=0)
      'Died of Other Causes' → competing risk, set to NaN → excluded

    Endpoint:
      Y=1  if status=1 AND time ≤ cutoff
      Y=0  if (status=1 AND time > cutoff) OR (status=0 AND time ≥ cutoff)
      Y=NaN otherwise (censored < cutoff, or competing death) → dropped
    """
    time = df[time_col].astype(float)

    if status_col_cancer in df.columns:
        raw = df[status_col_cancer]
        print(f"Using '{status_col_cancer}' for event status.")
    else:
        # Fallback: overall_survival.  In this dataset 0=dead, 1=alive.
        raw = df[status_col_overall]
        print(f"Fallback: using '{status_col_overall}' for event status.")

    # Parse status to binary
    if raw.dtype.kind in "ifb":
        # For overall_survival: 0=dead, 1=alive in METABRIC Kaggle CSV
        # So event = (raw == 0)
        status = (raw.astype(int) == 0).astype(int)
    else:
        s = raw.astype(str).str.strip().str.lower()
        died_labels = {"died of disease", "died", "dead", "deceased"}
        alive_labels = {"living", "alive"}
        status = np.where(s.isin(died_labels), 1,
                          np.where(s.isin(alive_labels), 0, np.nan))
        status = pd.Series(status, index=df.index)

    # Report exclusions explicitly
    if raw.dtype.kind not in "ifb":
        s_lower = raw.astype(str).str.strip().str.lower()
        n_doc = (s_lower == "died of other causes").sum()
        n_nan = raw.isna().sum()
        print(f"  → {n_doc} patients 'Died of Other Causes' excluded "
              f"(competing risk, not cancer-specific)")
        if n_nan > 0:
            print(f"  → {n_nan} patients with missing status excluded")

    # Build 5-year endpoint
    event_5y = pd.Series(np.nan, index=df.index, dtype=float)
    event_5y.loc[(status == 1) & (time <= cutoff_months)] = 1.0
    event_5y.loc[((status == 1) & (time > cutoff_months)) |
                 ((status == 0) & (time >= cutoff_months))] = 0.0

    n_ev = int((event_5y == 1).sum())
    n_ne = int((event_5y == 0).sum())
    n_na = int(event_5y.isna().sum())
    print(f"5-year endpoint: {n_ev} events, {n_ne} non-events, "
          f"{n_na} excluded (censored <5y or competing)")
    return event_5y


# ======================================================================
# 2.  FEATURE VIEWS
# ======================================================================

def split_views(df):
    """
    Split columns into clinical vs gene-expression views.

    Clinical: 31 variables from the Kaggle METABRIC description.
    Gene-expression: mRNA z-score columns (numeric, not ending in _mut).
    Mutation indicators (_mut) are catalogued but excluded from the
    gene-expression view to match the paper's scope.
    """
    clinical_candidates = [
        "patient_id", "age_at_diagnosis", "type_of_breast_surgery",
        "cancer_type", "cancer_type_detailed", "cellularity", "chemotherapy",
        "pam50_+_claudin-low_subtype", "cohort", "er_status_measured_by_ihc",
        "er_status", "neoplasm_histologic_grade", "her2_status_measured_by_snp6",
        "her2_status", "tumor_other_histologic_subtype", "hormone_therapy",
        "inferred_menopausal_state", "integrative_cluster",
        "primary_tumor_laterality", "lymph_nodes_examined_positive",
        "mutation_count", "nottingham_prognostic_index", "oncotree_code",
        "overall_survival_months", "overall_survival", "pr_status",
        "radio_therapy", "3-gene_classifier_subtype", "tumor_size",
        "tumor_stage", "death_from_cancer"
    ]
    clinical_cols = [c for c in clinical_candidates if c in df.columns]

    exclude = set(clinical_cols) | {"patient_id", "overall_survival_months",
                                     "overall_survival", "death_from_cancer"}

    # Gene-expression = numeric, not _mut, not clinical/survival/ID
    expr_cols = [c for c in df.columns
                 if c not in exclude
                 and not c.endswith("_mut")
                 and df[c].dtype.kind in "if"]

    mut_cols = [c for c in df.columns
                if c not in exclude and c.endswith("_mut")]

    # Sanity check: expression z-scores should have mean ≈ 0, std ≈ 1
    if len(expr_cols) > 0:
        sample_means = df[expr_cols].mean().abs()
        sample_stds = df[expr_cols].std()
        n_suspicious = int((sample_means > 1.0).sum() + (sample_stds < 0.5).sum())
        if n_suspicious > 0:
            print(f"  WARNING: {n_suspicious} expression columns have "
                  f"non-z-score-like distributions — verify data schema.")

    print(f"Clinical cols: {len(clinical_cols)}, "
          f"expression cols: {len(expr_cols)}, "
          f"mutation indicator cols: {len(mut_cols)}")
    return clinical_cols, expr_cols, mut_cols


# ======================================================================
# 3.  ML PIPELINE  (imputation inside pipeline — no leakage)
# ======================================================================

def build_ml_view(df, cols, y, view_name, random_state=42):
    """
    Train ML classifiers with proper pipeline (impute → transform → model).
    Returns best model name, fitted pipeline, CV risk scores, metrics dict.
    """
    X = df[cols].copy()
    print(f"\n[{view_name}] {X.shape[1]} features")

    num_cols = [c for c in cols if df[c].dtype.kind in "if"]
    cat_cols = [c for c in cols if c not in num_cols]


    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    # sparse_output (sklearn ≥1.2) or sparse (sklearn <1.2)
    import sklearn
    ohe_kwargs = {"handle_unknown": "ignore"}
    major, minor = map(int, sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 2):
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe",    OneHotEncoder(**ohe_kwargs)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Models
    estimators = [
        ("logistic_en", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            max_iter=5000, class_weight="balanced",
            n_jobs=-1, random_state=random_state)),
        ("random_forest", RandomForestClassifier(
            n_estimators=500, min_samples_split=5, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=random_state)),
        ("grad_boost", GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            random_state=random_state)),
    ]
    if HAS_XGB:
        estimators.append(("xgboost", XGBClassifier(**_XGB_KW,
            
            n_estimators=500, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=-1, random_state=random_state)))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    metrics = {}
    best_auc, best_name, best_model, best_cv = -np.inf, None, None, None

    for name, clf in estimators:
        pipe = Pipeline([("pre", preprocessor), ("clf", clf)])
        cv_njobs = 1 if (name == "xgboost" and _USE_GPU) else -1
        cv_probs = cross_val_predict(
            pipe, X, y, cv=skf, method="predict_proba", n_jobs=cv_njobs
        )[:, 1]
        a = roc_auc_score(y, cv_probs)
        print(f"  {name}: AUC = {a:.4f}")
        metrics[name] = a
        if a > best_auc:
            best_auc, best_name, best_cv = a, name, cv_probs
            pipe.fit(X, y)
            best_model = pipe

    print(f"  → Best: {best_name} (AUC = {best_auc:.4f})")
    return best_name, best_model, best_cv, metrics


def bootstrap_auc_ci(y, scores, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    aucs = np.array(aucs)
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return roc_auc_score(y, scores), lo, hi


# ======================================================================
# 4.  SENSITIVITY ANALYSIS: genomic feature count (REV1-4)
# ======================================================================

def sensitivity_feature_count(df, expr_cols, y, outdir, random_state=42):
    """
    Sensitivity analysis: train genomic RF at different feature counts.
    Feature selection (ANOVA F-test via SelectKBest) is performed INSIDE
    each CV fold to avoid information leakage.
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    counts = [25, 50, 100, 200, len(expr_cols)]
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for k in counts:
        # SelectKBest inside the pipeline → no leakage
        rf = RandomForestClassifier(
            n_estimators=500, min_samples_split=5, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=random_state)
        if k >= len(expr_cols):
            # Use all features — no selection step needed
            pipe = Pipeline([
                ("pre", ColumnTransformer([
                    ("num", Pipeline([
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc",  StandardScaler()),
                    ]), expr_cols)
                ])),
                ("clf", rf),
            ])
        else:
            pipe = Pipeline([
                ("pre", ColumnTransformer([
                    ("num", Pipeline([
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc",  StandardScaler()),
                    ]), expr_cols)
                ])),
                ("select", SelectKBest(f_classif, k=k)),
                ("clf", rf),
            ])
        probs = cross_val_predict(pipe, df, y, cv=skf,
                                  method="predict_proba", n_jobs=-1)[:, 1]
        a = roc_auc_score(y, probs)
        _, lo, hi = bootstrap_auc_ci(y.values, probs)
        results.append({"n_features": k, "auc": a, "ci_lo": lo, "ci_hi": hi})
        print(f"  Sensitivity: top-{k} features → AUC={a:.4f} [{lo:.4f}, {hi:.4f}]")

    pd.DataFrame(results).to_csv(outdir / "sensitivity_feature_count.csv",
                                 index=False)
    return results


# ======================================================================
# 5.  COPULA FUNCTIONS  (+ Frank copula for REV1-7)
# ======================================================================

EPS = 1e-10

def empirical_copula_C(u, v):
    n = len(u)
    Cn = np.empty(n)
    for i in range(n):
        Cn[i] = np.mean((u <= u[i]) & (v <= v[i]))
    return Cn

# ---- Gaussian ----
def copula_gaussian_cdf(u, v, rho):
    u, v = np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)
    x, y_ = stats.norm.ppf(u), stats.norm.ppf(v)
    return stats.multivariate_normal.cdf(
        np.column_stack([x, y_]), mean=[0,0], cov=[[1,rho],[rho,1]])

def fit_gaussian_copula(u, v):
    tau = stats.kendalltau(u, v)[0]
    return np.sin(np.pi * tau / 2.0)

def simulate_gaussian_copula(n, rho, random_state=None):
    rng = np.random.default_rng(random_state)
    x = rng.multivariate_normal([0,0], [[1,rho],[rho,1]], size=n)
    return stats.norm.cdf(x[:,0]), stats.norm.cdf(x[:,1])

def tail_dep_gaussian(rho):
    return (0.0, 0.0) if abs(rho) < 1-1e-6 else (1.0, 1.0)

# ---- Clayton ----
def copula_clayton_cdf(u, v, theta):
    u, v = np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)
    return np.maximum((u**(-theta) + v**(-theta) - 1.0)**(-1.0/theta), EPS)

def fit_clayton_copula(u, v):
    tau = max(stats.kendalltau(u, v)[0], 1e-6)
    return 2*tau / (1-tau)

def tail_dep_clayton(theta):
    return (2**(-1.0/theta), 0.0) if theta > 0 else (0.0, 0.0)

# ---- Gumbel ----
def copula_gumbel_cdf(u, v, theta):
    u, v = np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)
    t = ((-np.log(u))**theta + (-np.log(v))**theta)**(1.0/theta)
    return np.exp(-t)

def fit_gumbel_copula(u, v):
    tau = max(stats.kendalltau(u, v)[0], 0.0)
    return max(1.0 / (1.0 - tau + 1e-6), 1.0)

def tail_dep_gumbel(theta):
    return (0.0, 2 - 2**(1.0/max(theta, 1.0)))

# ---- Frank (NEW — REV1-7) ----
def copula_frank_cdf(u, v, theta):
    u, v = np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)
    if abs(theta) < 1e-8:
        return u * v  # independence
    num = (np.exp(-theta*u) - 1) * (np.exp(-theta*v) - 1)
    den = np.exp(-theta) - 1
    return -np.log(1 + num / den) / theta

def fit_frank_copula(u, v):
    """Estimate Frank theta from Kendall's tau via numerical inversion."""
    from scipy.optimize import brentq
    from scipy.integrate import quad

    tau_obs = stats.kendalltau(u, v)[0]

    def debye1(t):
        if abs(t) < 1e-8:
            return 1.0
        val, _ = quad(lambda x: x / (np.exp(x) - 1), 0, t)
        return val / t

    def tau_frank(theta):
        if abs(theta) < 1e-8:
            return 0.0
        return 1 - 4/theta * (1 - debye1(theta))

    # Search in a wide range
    try:
        theta = brentq(lambda th: tau_frank(th) - tau_obs, -50, 50)
    except ValueError:
        theta = 0.0
    return theta

def simulate_frank_copula(n, theta, random_state=None):
    rng = np.random.default_rng(random_state)
    u = rng.uniform(size=n)
    p = rng.uniform(size=n)
    if abs(theta) < 1e-8:
        return u, p
    # Conditional inversion: v = C^{-1}_{2|1}(p | u)
    # Derived from dC/du = p, solving for v.
    # v = -log(1 + p*(e^{-θ} - 1) / (e^{-θu}*(1-p) + p)) / θ
    a = np.exp(-theta * u)
    b = np.exp(-theta)
    v = -np.log(1 + p * (b - 1) / (a * (1 - p) + p)) / theta
    v = np.clip(v, EPS, 1 - EPS)
    return u, v

def tail_dep_frank(theta):
    return (0.0, 0.0)  # Frank has no tail dependence

# ---- Archimedean simulation (Clayton, Gumbel) via Genest-Rivest ----
def _simulate_archimedean(n, theta, family, random_state=None):
    rng = np.random.default_rng(random_state)
    s = rng.uniform(size=n)
    t = rng.uniform(size=n)

    if family == "clayton":
        phi     = lambda x: (x**(-theta) - 1) / theta
        phi_inv = lambda w: (1 + theta*w)**(-1/theta)
        phi_p   = lambda x: -x**(-(theta+1))
    elif family == "gumbel":
        phi     = lambda x: (-np.log(x))**theta
        phi_inv = lambda w: np.exp(-w**(1/theta))
        phi_p   = lambda x: -theta * (-np.log(x))**(theta-1) / x
    else:
        raise ValueError(family)

    K = lambda x: x - phi(x) / phi_p(x)
    xg = np.linspace(1e-6, 1-1e-6, 5000)
    Kg = K(xg)
    idx = np.argsort(Kg)
    Ks, xs = Kg[idx], xg[idx]
    w = np.interp(np.clip(t, Ks[0], Ks[-1]), Ks, xs)
    w = np.clip(w, EPS, 1-EPS)
    pw = phi(w)
    u = phi_inv(s * pw)
    v = phi_inv((1-s) * pw)
    return np.clip(u, EPS, 1-EPS), np.clip(v, EPS, 1-EPS)

def simulate_clayton_copula(n, theta, random_state=None):
    return _simulate_archimedean(n, theta, "clayton", random_state)

def simulate_gumbel_copula(n, theta, random_state=None):
    return _simulate_archimedean(n, theta, "gumbel", random_state)


# ======================================================================
# 6.  GOF + BOOTSTRAP CIs for copula params  (REV1-7)
# ======================================================================

def cvm_statistic(u, v, copula_cdf, params):
    Cn = empirical_copula_C(u, v)
    Ct = copula_cdf(u, v, *params)
    return np.mean((Cn - Ct)**2)

def cvm_gof_bootstrap(u, v, copula_cdf, copula_sim, fit_fn, params,
                      n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    obs = cvm_statistic(u, v, copula_cdf, params)
    boot = []
    for _ in range(n_bootstrap):
        u_b, v_b = copula_sim(len(u), *params,
                              random_state=rng.integers(1e9))
        # Re-estimate parameter on bootstrap sample
        param_b = fit_fn(u_b, v_b)
        stat_b = cvm_statistic(u_b, v_b, copula_cdf, (param_b,))
        boot.append(stat_b)
    boot = np.array(boot)
    p = (1 + np.sum(boot >= obs)) / (n_bootstrap + 1)
    return obs, p, boot

def bootstrap_copula_param_ci(u, v, fit_fn, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap CI for a copula parameter by resampling (u,v) pairs."""
    rng = np.random.default_rng(seed)
    n = len(u)
    params = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        params.append(fit_fn(u[idx], v[idx]))
    params = np.array(params)
    lo = np.percentile(params, 100*alpha/2)
    hi = np.percentile(params, 100*(1-alpha/2))
    return fit_fn(u, v), lo, hi


# ======================================================================
# 7.  PLOTTING
# ======================================================================

def plot_histograms(risk_clin, risk_gen, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(risk_clin, bins=30, alpha=0.7)
    ax1.set_title("Clinical risk score distribution")
    ax1.set_xlabel("Risk score"); ax1.set_ylabel("Count")
    ax2.hist(risk_gen, bins=30, alpha=0.7)
    ax2.set_title("Gene-expression risk score distribution")
    ax2.set_xlabel("Risk score")
    plt.tight_layout(); plt.savefig(outdir/"risk_histograms.png", dpi=300); plt.close()

def plot_risk_scatter(risk_clin, risk_gen, y, outdir):
    plt.figure(figsize=(6,6))
    sc = plt.scatter(risk_clin, risk_gen, c=y, cmap="coolwarm", alpha=0.7)
    plt.colorbar(sc, label="5-year cancer death")
    plt.xlabel("Clinical risk score"); plt.ylabel("Gene-expression risk score")
    plt.title("Clinical vs Gene-expression risk")
    plt.tight_layout(); plt.savefig(outdir/"risk_scatter.png", dpi=300); plt.close()

def plot_roc_curves(y, scores_dict, outdir, ci_dict=None):
    plt.figure(figsize=(6,6))
    for label, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y, scores)
        a = auc(fpr, tpr)
        ci_str = ""
        if ci_dict and label in ci_dict:
            lo, hi = ci_dict[label]
            ci_str = f" [{lo:.3f}–{hi:.3f}]"
        plt.plot(fpr, tpr, label=f"{label} (AUC={a:.3f}{ci_str})")
    plt.plot([0,1],[0,1],"k--",alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC curves"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(outdir/"roc_curves.png", dpi=300); plt.close()

def plot_km_joint_risk(time, status, risk_clin, risk_gen, outdir):
    if not HAS_LIFELINES:
        print("  lifelines not available — skipping KM plot")
        return

    med_c = np.median(risk_clin)
    med_g = np.median(risk_gen)
    hc = risk_clin > med_c
    hg = risk_gen > med_g

    groups = {
        "low-low":               (~hc) & (~hg),
        "high-clinical-only":     hc & (~hg),
        "high-expression-only":  (~hc) &  hg,
        "high-both":              hc &  hg,
    }

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7,6))
    for label, mask in groups.items():
        if mask.sum() < 10:
            continue
        kmf.fit(time[mask], event_observed=status[mask], label=f"{label} (n={mask.sum()})")
        kmf.plot(ci_show=False)

    # Log-rank: high-both vs low-low
    m_hb = groups["high-both"]
    m_ll = groups["low-low"]
    if m_hb.sum() >= 10 and m_ll.sum() >= 10:
        lr = logrank_test(time[m_hb], time[m_ll],
                          event_observed_A=status[m_hb],
                          event_observed_B=status[m_ll])
        plt.title(f"KM by joint risk strata\n"
                  f"(median split; log-rank high-both vs low-low p={lr.p_value:.2e})")
    else:
        plt.title("KM by joint risk strata (median split)")

    plt.xlabel("Time (months)"); plt.ylabel("Survival probability")
    plt.tight_layout(); plt.savefig(outdir/"km_joint_risk.png", dpi=300); plt.close()

    # Save group sizes
    info = {k: int(v.sum()) for k, v in groups.items()}
    info["median_clinical"] = float(med_c)
    info["median_expression"] = float(med_g)
    with open(outdir / "km_group_info.json", "w") as f:
        json.dump(info, f, indent=2)


def plot_km_overall_survival_fullcohort(df_full, model_clin, clinical_ml_cols,
                                        model_gen, expr_cols, outdir,
                                        oof_pid=None, oof_clin=None, oof_gen=None):
    """Kaplan-Meier for TRUE overall survival on the FULL METABRIC cohort.

    Every death (any cause) is an event, coded from overall_survival == 0
    (METABRIC Kaggle: 0 = dead, 1 = alive). Analytic-cohort patients keep their
    out-of-fold scores from the selected pipeline; patients excluded from the
    binary endpoint receive out-of-sample predictions from the final models
    fitted on the analytic cohort. Risk-group thresholds are the medians of the
    analytic out-of-fold scores.
    """
    if not HAS_LIFELINES:
        print("  lifelines not available - skipping overall-survival KM")
        return
    miss_c = [c for c in clinical_ml_cols if c not in df_full.columns]
    miss_g = [c for c in expr_cols if c not in df_full.columns]
    if miss_c or miss_g:
        print("  Schema mismatch - skipping overall-survival KM.")
        return
    risk_clin = model_clin.predict_proba(df_full[clinical_ml_cols])[:, 1]
    risk_gen = model_gen.predict_proba(df_full[expr_cols])[:, 1]
    if oof_pid is not None and "patient_id" in df_full.columns:
        cmap = {str(p): float(x) for p, x in zip(oof_pid, oof_clin)}
        gmap = {str(p): float(x) for p, x in zip(oof_pid, oof_gen)}
        pid_full = df_full["patient_id"].astype(str).values
        n_oof = 0
        for i, pid in enumerate(pid_full):
            if pid in cmap:
                risk_clin[i] = cmap[pid]; risk_gen[i] = gmap[pid]; n_oof += 1
        print(f"  KM scores: {n_oof} analytic patients use out-of-fold scores; "
              f"{len(pid_full) - n_oof} excluded patients use out-of-sample predictions.")
    time = df_full["overall_survival_months"].astype(float).values
    # All-cause death: overall_survival == 0 (dead) in the METABRIC Kaggle CSV.
    status = (pd.to_numeric(df_full["overall_survival"], errors="coerce").values == 0).astype(int)
    valid = ~np.isnan(time)
    time, status = time[valid], status[valid]
    risk_clin, risk_gen = risk_clin[valid], risk_gen[valid]
    # Thresholds from the ANALYTIC out-of-fold scores (not full-cohort medians).
    med_c = float(np.median(oof_clin)); med_g = float(np.median(oof_gen))
    hc = risk_clin > med_c; hg = risk_gen > med_g
    groups = {
        "low-low": (~hc) & (~hg),
        "high-clinical-only": hc & (~hg),
        "high-expression-only": (~hc) & hg,
        "high-both": hc & hg,
    }
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7, 6))
    counts = {k: int(m.sum()) for k, m in groups.items()}
    for label, mask in groups.items():
        if mask.sum() < 10:
            continue
        kmf.fit(time[mask], event_observed=status[mask], label=f"{label} (n={int(mask.sum())})")
        kmf.plot(ci_show=False)
    m_hb, m_ll = groups["high-both"], groups["low-low"]
    lr_p = float("nan")
    if m_hb.sum() >= 10 and m_ll.sum() >= 10:
        lr = logrank_test(time[m_hb], time[m_ll],
                          event_observed_A=status[m_hb], event_observed_B=status[m_ll])
        lr_p = float(lr.p_value)
        plt.title("KM by joint risk strata - overall survival (full cohort)\n"
                  f"(median split; log-rank high-both vs low-low p={lr_p:.2e})")
    else:
        plt.title("KM by joint risk strata - overall survival (full cohort)")
    plt.xlabel("Time (months)"); plt.ylabel("Overall survival probability")
    plt.tight_layout()
    plt.savefig(outdir / "km_joint_risk_overall_survival.png", dpi=300); plt.close()
    pd.DataFrame([{"group": k, "n": v} for k, v in counts.items()]).to_csv(
        outdir / "km_overall_survival_group_counts.csv", index=False)
    with open(outdir / "km_overall_survival_summary.json", "w") as f:
        json.dump({"event_definition": "all-cause death (overall_survival == 0)",
                   "cohort": "full METABRIC (1904)",
                   "n_scored": int(len(time)),
                   "n_events": int(status.sum()),
                   "threshold_clinical_from_analytic_oof": med_c,
                   "threshold_gene_from_analytic_oof": med_g,
                   "group_counts": counts,
                   "logrank_high_both_vs_low_low_p": lr_p}, f, indent=2)
    print(f"  Overall-survival KM group counts: {counts}; log-rank p={lr_p:.2e}")


def compute_copula_joint_risk(u, v, copula_cdf, params, y, outdir):
    """
    Copula-based continuous joint risk score.

    We use C(u, v) — the copula CDF evaluated at each patient's
    pseudo-observations — as a concordant high-risk score.
    C(u, v) = Pr(U ≤ u, V ≤ v): patients with high u AND high v
    (i.e., high clinical and high expression risk) get the largest
    C(u, v) values, making it a natural joint risk metric.
    """
    joint_score = copula_cdf(u, v, *params)

    a = roc_auc_score(y, joint_score)
    _, lo, hi = bootstrap_auc_ci(y.values, joint_score)
    print(f"  Copula joint risk score AUC: {a:.4f} [{lo:.4f}, {hi:.4f}]")

    pd.DataFrame({
        "u": u, "v": v, "joint_risk_C_uv": joint_score, "y_5y": y
    }).to_csv(outdir / "copula_joint_risk_scores.csv", index=False)

    return joint_score, a

def plot_copula_contours(u, v, copula_cdf, params, outdir, name):
    plt.figure(figsize=(6,6))
    plt.scatter(u, v, alpha=0.4, s=10)
    grid = np.linspace(0.01, 0.99, 50)
    U, V = np.meshgrid(grid, grid)
    C = copula_cdf(U.ravel(), V.ravel(), *params).reshape(U.shape)
    cs = plt.contour(U, V, C, levels=10, colors="k", linewidths=0.5)
    plt.clabel(cs, inline=1, fontsize=8)
    plt.xlabel("U (clinical risk rank)"); plt.ylabel("V (expression risk rank)")
    plt.title(f"{name} copula contours")
    plt.tight_layout(); plt.savefig(outdir/f"copula_{name}_contours.png", dpi=300); plt.close()

def plot_copula_heatmap(u, v, copula_cdf, params, outdir, name):
    grid = np.linspace(0.01, 0.99, 30)
    U, V = np.meshgrid(grid, grid)
    emp = np.array([np.mean((u <= U[i,j]) & (v <= V[i,j]))
                    for i in range(len(grid)) for j in range(len(grid))]).reshape(U.shape)
    fit = copula_cdf(U.ravel(), V.ravel(), *params).reshape(U.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    im0 = ax1.imshow(emp, origin="lower", extent=[0,1,0,1], aspect="auto")
    ax1.set_title("Empirical copula"); fig.colorbar(im0, ax=ax1)
    im1 = ax2.imshow(fit, origin="lower", extent=[0,1,0,1], aspect="auto")
    ax2.set_title(f"Fitted {name} copula"); fig.colorbar(im1, ax=ax2)
    plt.tight_layout(); plt.savefig(outdir/f"copula_{name}_emp_vs_fit.png", dpi=300); plt.close()


# ======================================================================
# 8.  COMPETING RISKS (REV1-9)
# ======================================================================

def competing_risks_cif(df_full, model_clin, clinical_ml_cols,
                        model_gen, expr_cols, outdir,
                        oof_pid=None, oof_clin=None, oof_gen=None):
    """
    Cumulative incidence functions treating 'Died of Other Causes'
    as a competing event, using the Aalen-Johansen estimator.

    Uses the FULL dataset (not the analytic cohort) so that competing
    deaths are present in the event coding.

    Risk scores are constructed to avoid in-sample optimism in the risk-group
    definition. Patients in the analytic cohort receive their out-of-fold
    out-of-fold (cross-validated) risk scores; the remaining patients (those
    excluded from the endpoint, e.g. other-cause deaths, who were never used to
    train any model) receive genuine out-of-sample predictions from the fitted
    models. If out-of-fold scores are not supplied, the function falls back to
    scoring all patients with the fitted models.
    """
    try:
        from lifelines import AalenJohansenFitter
    except ImportError:
        print("  AalenJohansenFitter not available — skipping CIF.")
        print("  Install lifelines >= 0.27 for competing-risks support.")
        return

    # Schema check: verify all required columns exist in df_full
    missing_clin = [c for c in clinical_ml_cols if c not in df_full.columns]
    missing_expr = [c for c in expr_cols if c not in df_full.columns]
    if missing_clin or missing_expr:
        print(f"  Schema mismatch — missing clinical: {missing_clin}, "
              f"expression: {missing_expr}")
        print("  Skipping competing-risks plot.")
        return

    # Predict risk scores for the full dataset using fitted models. For patients
    # who were excluded from the analytic cohort (never in any training fold),
    # these are genuine out-of-sample predictions.
    try:
        risk_clin = model_clin.predict_proba(df_full[clinical_ml_cols])[:, 1]
        risk_gen = model_gen.predict_proba(df_full[expr_cols])[:, 1]
    except Exception as e:
        print(f"  Could not score full dataset for CIF: {e}")
        print("  Skipping competing-risks plot.")
        return

    # Overwrite analytic-cohort patients with their out-of-fold
    # scores so that the risk-group definition is not circular.
    if oof_pid is not None and "patient_id" in df_full.columns:
        cmap = {str(p): float(s) for p, s in zip(oof_pid, oof_clin)}
        gmap = {str(p): float(s) for p, s in zip(oof_pid, oof_gen)}
        pid_full = df_full["patient_id"].astype(str).values
        n_oof = 0
        for i, pid in enumerate(pid_full):
            if pid in cmap:
                risk_clin[i] = cmap[pid]; risk_gen[i] = gmap[pid]; n_oof += 1
        print(f"  CIF risk scores: {n_oof} analytic patients use out-of-fold "
              f"scores; {len(pid_full) - n_oof} excluded patients use "
              f"out-of-sample predictions.")

    time = df_full["overall_survival_months"].astype(float).values
    raw = df_full["death_from_cancer"].astype(str).str.strip().str.lower()

    # 3-level event: 0=censored/living, 1=cancer death, 2=other cause death
    event = np.where(raw == "died of disease", 1,
                     np.where(raw == "died of other causes", 2, 0))

    # Drop rows where time is NaN
    valid = ~np.isnan(time)
    time, event = time[valid], event[valid]
    risk_clin, risk_gen = risk_clin[valid], risk_gen[valid]

    # Use the SAME strata cutoffs as the overall-survival KM: medians of the
    # analytic out-of-fold scores, NOT full-cohort medians, so that Figure 6
    # (KM) and Figure 7 (competing risks) define identical risk groups.
    if oof_clin is not None and oof_gen is not None:
        med_c, med_g = float(np.median(oof_clin)), float(np.median(oof_gen))
    else:
        med_c, med_g = np.median(risk_clin), np.median(risk_gen)
    hc = risk_clin > med_c
    hg = risk_gen > med_g

    groups = {
        "low-low":            (~hc) & (~hg),
        "high-clinical-only":  hc & (~hg),
        "high-expression-only":  (~hc) &  hg,
        "high-both":           hc &  hg,
    }

    plt.figure(figsize=(7, 6))
    for label, mask in groups.items():
        if mask.sum() < 10:
            continue
        ajf = AalenJohansenFitter(calculate_variance=False)
        ajf.fit(time[mask], event[mask], event_of_interest=1)
        plt.step(ajf.cumulative_density_.index,
                 ajf.cumulative_density_.values.ravel(),
                 label=f"{label} (n={mask.sum()})", where="post")

    plt.xlabel("Time (months)")
    plt.ylabel("Cumulative incidence of cancer death")
    plt.title("Cumulative incidence (competing risks)\n"
              "Event=cancer death, Competing=other-cause death")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "competing_risks_cif.png", dpi=300)
    plt.close()
    print(f"  CIF plot saved. Events: cancer={int((event==1).sum())}, "
          f"competing={int((event==2).sum())}, censored={int((event==0).sum())}")

    # Gray's test: high-both vs low-low (REV1-9)
    # Gray's test compares CIFs in the presence of competing risks.
    # We use a permutation-based approach since lifelines does not
    # include Gray's test natively.
    m_hb = groups["high-both"]
    m_ll = groups["low-low"]
    if m_hb.sum() >= 10 and m_ll.sum() >= 10:
        try:
            from lifelines.statistics import logrank_test
            # Cause-specific log-rank test: treats competing events
            # (death from other causes) as censored observations.
            # This is NOT Gray's test, but the standard cause-specific
            # comparison used when Gray's test is unavailable.
            cs_status_hb = (event[m_hb] == 1).astype(int)
            cs_status_ll = (event[m_ll] == 1).astype(int)
            lr = logrank_test(time[m_hb], time[m_ll],
                              event_observed_A=cs_status_hb,
                              event_observed_B=cs_status_ll)
            print(f"  Cause-specific log-rank (high-both vs low-low): "
                  f"p = {lr.p_value:.2e}")

            pd.DataFrame([{
                "test": "cause-specific log-rank",
                "groups": "high-both vs low-low",
                "statistic": lr.test_statistic,
                "p_value": lr.p_value,
            }]).to_csv(outdir / "competing_risks_test.csv", index=False)
        except Exception as e:
            print(f"  Could not run cause-specific log-rank: {e}")


# ======================================================================
# 9.  MAIN PIPELINE
# ======================================================================

def _fit_unpenalized_logistic(X, y):
    """Support both current and older scikit-learn penalty syntax."""
    last_error = None
    for penalty in (None, "none"):
        try:
            model = LogisticRegression(
                penalty=penalty, solver="lbfgs", max_iter=5000
            )
            model.fit(X, y)
            return model
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError("Unable to fit unpenalized logistic regression.") from last_error


# ======================================================================
# 8b.  CALIBRATION ASSESSMENT  (Editor point 1: Hosmer-Lemeshow etc.)
# ======================================================================

def hosmer_lemeshow(y, p, g=10):
    """Hosmer-Lemeshow goodness-of-fit test using g equal-count bins of
    predicted risk. Returns chi-square statistic, degrees of freedom (bins-2),
    and p-value. Rank-based binning avoids degenerate quantile edges."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    grp = np.empty(n, int)
    grp[order] = np.floor(np.arange(n) * g / n).astype(int)
    chi2 = 0.0; used = 0
    for b in range(g):
        m = grp == b
        if m.sum() == 0:
            continue
        obs1, exp1 = y[m].sum(), p[m].sum()
        obs0, exp0 = m.sum() - obs1, m.sum() - exp1
        for obs, exp in ((obs1, exp1), (obs0, exp0)):
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
        used += 1
    dof = max(used - 2, 1)
    return {"chi2": float(chi2), "dof": int(dof),
            "p_value": float(stats.chi2.sf(chi2, dof)), "n_bins": used}


def calibration_slope_intercept(y, p, eps=1e-6):
    """Calibration slope (logistic reg of y on logit(p)) and calibration
    intercept-in-the-large (Newton step on intercept with slope fixed at 1).
    Ideal slope=1, intercept=0; slope<1 indicates over-dispersed risks."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = _fit_unpenalized_logistic(logit, y)
    slope = float(lr.coef_[0, 0])
    a = 0.0
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(a + logit.ravel())))
        grad = np.sum(y - mu); hess = -np.sum(mu * (1 - mu))
        if abs(hess) < 1e-12:
            break
        step = grad / hess; a -= step
        if abs(step) < 1e-10:
            break
    return {"slope": slope, "intercept": float(a)}


def integrated_calibration_index(y, p):
    """ICI: mean |predicted - isotonic-smoothed observed| risk."""
    from sklearn.isotonic import IsotonicRegression
    y = np.asarray(y, float); p = np.asarray(p, float)
    obs = IsotonicRegression(out_of_bounds="clip").fit_transform(p, y)
    return float(np.mean(np.abs(obs - p)))


def safe_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _calibration_predictor(score, score_type):
    score = np.asarray(score, dtype=float)
    if score_type == "probability":
        score = safe_logit(score)
    elif score_type != "continuous":
        raise ValueError("score_type must be 'probability' or 'continuous'")
    return score.reshape(-1, 1)


def fit_recalibration_model(y, score, score_type):
    return _fit_unpenalized_logistic(
        _calibration_predictor(score, score_type), np.asarray(y, dtype=int))


def apply_recalibration_model(model, score, score_type):
    return model.predict_proba(_calibration_predictor(score, score_type))[:, 1]


def cross_fitted_recalibration(y, score, score_type, n_splits=5, random_state=42):
    """Second-stage cross-fitted recalibration of out-of-fold scores.

    Returns cross-fitted calibrated probabilities (for honest internal estimates)
    and a final map fitted on all scores for transport. The raw copula CDF must
    be passed with score_type='continuous' so it is mapped to a probability
    BEFORE any Brier/HL/slope/ICI calculation.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score) or np.isnan(score).any():
        raise ValueError("Invalid score vector supplied for recalibration.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    calibrated_oof = np.full(len(y), np.nan, dtype=float)
    for tr, te in splitter.split(np.zeros(len(y)), y):
        m = fit_recalibration_model(y[tr], score[tr], score_type)
        calibrated_oof[te] = apply_recalibration_model(m, score[te], score_type)
    if np.isnan(calibrated_oof).any():
        raise RuntimeError("Cross-fitted recalibration produced missing predictions.")
    final_model = fit_recalibration_model(y, score, score_type)
    return calibrated_oof, final_model


def format_p(p):
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def calibration_report(scores_dict, y, outdir, tag="metabric"):
    """Compute HL test, calibration slope/intercept, Brier, and ICI for each
    named probability score, and save a reliability-curve figure + CSV."""
    from sklearn.calibration import calibration_curve
    y = np.asarray(y, float)
    rows = []
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    for name, p in scores_dict.items():
        p = np.asarray(p, float)
        hl = hosmer_lemeshow(y, p)
        si = calibration_slope_intercept(y, p)
        brier = float(np.mean((p - y) ** 2))
        ici = integrated_calibration_index(y, p)
        rows.append({"score": name, "brier": brier,
                     "hl_chi2": hl["chi2"], "hl_dof": hl["dof"],
                     "hl_p_value": hl["p_value"],
                     "calibration_slope": si["slope"],
                     "calibration_intercept": si["intercept"],
                     "ici": ici})
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=f"{name} (HL {format_p(hl['p_value'])})")
        print(f"  [{name}] Brier={brier:.4f}  HL p={hl['p_value']:.3f}  "
              f"slope={si['slope']:.3f}  intercept={si['intercept']:.3f}  ICI={ici:.4f}")
    plt.xlabel("Mean predicted risk"); plt.ylabel("Observed event fraction")
    plt.title(f"Calibration ({tag})"); plt.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(outdir / f"calibration_{tag}.png", dpi=300); plt.close()
    df_out = pd.DataFrame(rows)
    df_out.to_csv(outdir / f"calibration_{tag}.csv", index=False)
    return df_out


# ======================================================================
# 8c.  FEATURE IMPORTANCE  (Editor point 3: key genes + interpretability)
# ======================================================================

def feature_importance_report(
    fitted_model, X, y, cols, view_name, outdir,
    topn=20, n_splits=5, n_repeats=5, random_state=42,
    importance_n_jobs=-1
):
    """Repeated cross-validated permutation importance over original columns.

    The primary ranking is the mean held-out ROC-AUC reduction after permutation.
    Tree impurity importance is retained only as a secondary descriptive column.
    """
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from joblib import parallel_backend

    X_view = X[cols].copy()
    y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    permutation_values = {feature: [] for feature in cols}
    impurity_values = []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X_view, y_array), start=1
    ):
        estimator = clone(fitted_model)
        # Avoid nested process pools. The outer permutation work uses threads,
        # while the fitted classifier uses one thread during this section.
        if "clf__n_jobs" in estimator.get_params(deep=True):
            estimator.set_params(clf__n_jobs=1)
        X_train, X_test = X_view.iloc[train_idx], X_view.iloc[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]
        estimator.fit(X_train, y_train)

        probability = estimator.predict_proba(X_test)[:, 1]
        fold_auc = roc_auc_score(y_test, probability)
        fold_rows.append({
            "fold": fold, "test_auc": fold_auc, "n_test": len(test_idx)
        })

        with parallel_backend("threading", n_jobs=importance_n_jobs):
            result = permutation_importance(
                estimator, X_test, y_test,
                scoring="roc_auc", n_repeats=n_repeats,
                random_state=random_state + fold,
                n_jobs=importance_n_jobs
            )
        for j, feature in enumerate(cols):
            permutation_values[feature].extend(result.importances[j, :].tolist())

        try:
            classifier = estimator.named_steps["clf"]
            fold_impurity = getattr(classifier, "feature_importances_", None)
            if fold_impurity is not None and len(fold_impurity) == len(cols):
                impurity_values.append(np.asarray(fold_impurity, dtype=float))
        except Exception:
            pass

    rows = []
    for feature in cols:
        values = np.asarray(permutation_values[feature], dtype=float)
        rows.append({
            "feature": feature,
            "perm_importance_mean": float(values.mean()),
            "perm_importance_sd": float(values.std(ddof=1)),
            "perm_importance_median": float(np.median(values)),
            "perm_importance_q025": float(np.quantile(values, 0.025)),
            "perm_importance_q975": float(np.quantile(values, 0.975)),
            "positive_importance_fraction": float(np.mean(values > 0)),
            "n_importance_estimates": int(len(values)),
        })

    importance = pd.DataFrame(rows)
    if impurity_values:
        impurity_matrix = np.vstack(impurity_values)
        importance["impurity_importance_mean"] = impurity_matrix.mean(axis=0)
        importance["impurity_importance_sd"] = impurity_matrix.std(axis=0, ddof=1)

    importance["stable_positive"] = (
        (importance["perm_importance_q025"] > 0) &
        (importance["positive_importance_fraction"] >= 0.80)
    )
    importance = importance.sort_values(
        "perm_importance_mean", ascending=False
    ).reset_index(drop=True)

    importance.to_csv(
        outdir / f"feature_importance_{view_name}.csv", index=False
    )
    importance.loc[importance["stable_positive"]].to_csv(
        outdir / f"feature_importance_{view_name}_stable_positive.csv",
        index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        outdir / f"feature_importance_{view_name}_fold_performance.csv",
        index=False
    )

    top = importance.head(topn).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.35 * len(top))))
    plt.barh(
        top["feature"], top["perm_importance_mean"],
        xerr=top["perm_importance_sd"]
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel(
        "Repeated cross-validated permutation importance\n"
        "(mean held-out ROC-AUC reduction)"
    )
    plt.ylabel("Feature")
    plt.title(f"Top {topn} features: {view_name}")
    plt.tight_layout()
    plt.savefig(
        outdir / f"feature_importance_{view_name}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"  [{view_name}] primary ranking: repeated CV permutation importance")
    print("  Top features: " + ", ".join(importance["feature"].head(15)))
    print(
        f"  Stable positive features: {int(importance['stable_positive'].sum())}"
    )

    # Permutation-vs-impurity concordance (tests the "consistent with impurity"
    # claim rather than assuming it): Spearman rho and top-20 overlap.
    concordance = {"view": view_name}
    if "impurity_importance_mean" in importance.columns:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(importance["perm_importance_mean"],
                              importance["impurity_importance_mean"])
        perm_top = set(importance.sort_values("perm_importance_mean", ascending=False).head(20)["feature"])
        imp_top = set(importance.sort_values("impurity_importance_mean", ascending=False).head(20)["feature"])
        overlap = len(perm_top & imp_top)
        concordance.update({"spearman_perm_vs_impurity": float(rho),
                            "spearman_p_value": float(pval),
                            "top20_overlap_count": int(overlap),
                            "top20_overlap_fraction": overlap / 20.0})
        print(f"  [{view_name}] permutation-vs-impurity Spearman={rho:.3f} "
              f"(p={pval:.3g}); top-20 overlap={overlap}/20")
    else:
        concordance.update({"spearman_perm_vs_impurity": None,
                            "note": "impurity importance unavailable for this view"})
    pd.DataFrame([concordance]).to_csv(
        outdir / f"feature_importance_{view_name}_concordance.csv", index=False)
    return importance


# ======================================================================
# 8d.  SIMPLE FUSION BASELINES  (referee-proofing: is the copula needed?)
# ======================================================================

def fusion_baselines(risk_clin, risk_gen, u, v, y, copula_fused, outdir,
                     random_state=42):
    """Compare the copula-fused score against simple score-level fusion
    baselines, so that the added value (or not) of the copula is explicit:

      * rank-average fusion:  mean of the two pseudo-observations (u, v);
      * probability-average fusion:  mean of the two predicted probabilities;
      * logistic stacking:  logistic regression of the outcome on the two
        out-of-fold risk scores, itself evaluated with 5-fold cross-validation
        (so the stacker is never trained and tested on the same patient).

    All scores are ranking scores, so we compare them by ROC-AUC with matched
    bootstrap confidence intervals."""
    rank_avg = (np.asarray(u) + np.asarray(v)) / 2.0
    prob_avg = (np.asarray(risk_clin) + np.asarray(risk_gen)) / 2.0

    Xs = np.column_stack([np.asarray(risk_clin), np.asarray(risk_gen)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    stack = cross_val_predict(
        LogisticRegression(max_iter=1000), Xs, y, cv=skf,
        method="predict_proba")[:, 1]

    scores = {
        "clinical": np.asarray(risk_clin),
        "gene-expression": np.asarray(risk_gen),
        "copula-fused": np.asarray(copula_fused),
        "rank-average": rank_avg,
        "probability-average": prob_avg,
        "logistic-stacking (CV)": stack,
    }
    rows = []
    for name, s in scores.items():
        a, lo, hi = bootstrap_auc_ci(y.values if hasattr(y, "values") else y, s)
        rows.append({"score": name, "auc": a, "ci_lo": lo, "ci_hi": hi})
        print(f"  {name:24s} AUC = {a:.4f} [{lo:.4f}, {hi:.4f}]")
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "fusion_baselines.csv", index=False)
    return out


def main():
    outdir = Path("study1_outputs")
    outdir.mkdir(exist_ok=True)

    # ---- Load & endpoint ----
    df = load_metabric()
    y_5y = build_5year_endpoint(df)
    if STRICT_INPUT_CHECKS:
        _e = int((y_5y == 1).sum()); _n = int((y_5y == 0).sum()); _x = int(y_5y.isna().sum())
        assert (_e, _n, _x) == (325, 1038, 541), (
            f"Primary cancer-specific endpoint {(_e, _n, _x)} != (325, 1038, 541); "
            f"input may be truncated or wrong.")

    mask_valid = ~y_5y.isna()
    df = df.loc[mask_valid].reset_index(drop=True)
    y = y_5y.loc[mask_valid].astype(int).reset_index(drop=True)
    print(f"\nAnalytic cohort: {df.shape[0]} patients "
          f"({int(y.sum())} events, {int((1-y).sum())} non-events)")

    # ---- Feature views ----
    clinical_cols, expr_cols, mut_cols = split_views(df)
    if STRICT_INPUT_CHECKS:
        assert len(expr_cols) == 489, (
            f"Expression feature count {len(expr_cols)} != 489; input may be wrong.")

    # BUG-1 FIX: exclude patient_id and survival columns
    exclude_from_clinical = {"overall_survival_months", "overall_survival",
                             "death_from_cancer", "patient_id"}
    clinical_ml_cols = [c for c in clinical_cols if c not in exclude_from_clinical]
    print(f"Clinical ML features: {len(clinical_ml_cols)}")

    # ---- Clinical ML ----
    print("\n" + "="*60)
    print("CLINICAL MODEL")
    print("="*60)
    best_c, model_c, risk_clin, met_c = build_ml_view(
        df, clinical_ml_cols, y, "clinical")

    # ---- Gene-expression ML (BUG-2 FIX: all expression, not top-50) ----
    print("\n" + "="*60)
    print("GENE-EXPRESSION MODEL")
    print("="*60)
    best_g, model_g, risk_gen, met_g = build_ml_view(
        df, expr_cols, y, "gene-expression")

    # ---- Verify actual XGBoost device + record environment (Phoenix point 7) ----
    xgb_device_used = None
    for _nm, _mdl in [(best_c, model_c), (best_g, model_g)]:
        if _nm == "xgboost":
            xgb_device_used = xgb_actual_device(_mdl)
            print(f"  [device check] XGBoost booster trained on: {xgb_device_used}")
    with open(outdir / "run_environment.json", "w") as _f:
        import json as _json
        _json.dump({"xgboost_config_device": ("cuda" if _USE_GPU else "cpu"),
                    "xgboost_actual_booster_device": xgb_device_used,
                    "package_versions": _pkg_versions(), "seed": 42,
                    "strict_input_checks": STRICT_INPUT_CHECKS}, _f, indent=2)

    # ---- Bootstrap AUC CIs (REV2) ----
    print("\n--- Bootstrap AUC 95% CIs ---")
    auc_c, lo_c, hi_c = bootstrap_auc_ci(y.values, risk_clin)
    auc_g, lo_g, hi_g = bootstrap_auc_ci(y.values, risk_gen)
    print(f"  Clinical:         {auc_c:.4f} [{lo_c:.4f}, {hi_c:.4f}]")
    print(f"  Gene-expression:  {auc_g:.4f} [{lo_g:.4f}, {hi_g:.4f}]")

    # ---- Sensitivity analysis (REV1-4) ----
    print("\n--- Sensitivity: gene-expression feature count ---")
    sens = sensitivity_feature_count(df, expr_cols, y, outdir)

    # ---- Save performance ----
    perf_rows = []
    for k, v in met_c.items():
        perf_rows.append({"view": "clinical", "model": k, "auc": v})
    for k, v in met_g.items():
        perf_rows.append({"view": "gene-expression", "model": k, "auc": v})
    pd.DataFrame(perf_rows).to_csv(outdir/"model_performance.csv", index=False)

    # ---- Save risk scores ----
    pd.DataFrame({
        "clinical_risk": risk_clin, "expression_risk": risk_gen, "y_5y": y
    }).to_csv(outdir/"risk_scores.csv", index=False)

    # ---- Plots ----
    plot_histograms(risk_clin, risk_gen, outdir)
    plot_risk_scatter(risk_clin, risk_gen, y, outdir)
    ci_dict = {"clinical": (lo_c, hi_c), "gene-expression": (lo_g, hi_g)}
    plot_roc_curves(y, {"clinical": risk_clin, "gene-expression": risk_gen},
                    outdir, ci_dict=ci_dict)

    # ---- KM: TRUE overall survival on the FULL 1904 cohort (all-cause death) ----
    print("\n--- Kaplan-Meier (overall survival, full cohort) by joint risk strata ---")
    df_full_km = load_metabric()
    plot_km_overall_survival_fullcohort(
        df_full_km, model_c, clinical_ml_cols, model_g, expr_cols, outdir,
        oof_pid=df["patient_id"].values, oof_clin=risk_clin, oof_gen=risk_gen)

    # ---- Competing risks CIF (REV1-9) ----
    print("\n--- Competing risks analysis ---")
    # CIF uses the FULL dataset (all 1904 patients, including those who
    # died of other causes) so that competing events are properly coded.
    # Risk scores are generated by applying the fitted models to all patients.
    df_full = load_metabric()
    competing_risks_cif(df_full, model_c, clinical_ml_cols,
                        model_g, expr_cols, outdir,
                        oof_pid=df["patient_id"].values,
                        oof_clin=risk_clin, oof_gen=risk_gen)

    # ==================================================================
    # COPULA ANALYSIS
    # ==================================================================
    print("\n" + "="*60)
    print("COPULA ANALYSIS")
    print("="*60)

    n = len(risk_clin)
    u = stats.rankdata(risk_clin, method="average") / (n + 1)
    v = stats.rankdata(risk_gen, method="average") / (n + 1)

    # ---- Fit all four copulas ----
    copula_specs = {
        "Gaussian": {
            "fit": fit_gaussian_copula,
            "cdf": copula_gaussian_cdf,
            "sim": simulate_gaussian_copula,
            "tail": tail_dep_gaussian,
        },
        "Clayton": {
            "fit": fit_clayton_copula,
            "cdf": copula_clayton_cdf,
            "sim": simulate_clayton_copula,
            "tail": tail_dep_clayton,
        },
        "Gumbel": {
            "fit": fit_gumbel_copula,
            "cdf": copula_gumbel_cdf,
            "sim": simulate_gumbel_copula,
            "tail": tail_dep_gumbel,
        },
        "Frank": {
            "fit": fit_frank_copula,
            "cdf": copula_frank_cdf,
            "sim": simulate_frank_copula,
            "tail": tail_dep_frank,
        },
    }

    fit_results = []
    gof_results = []

    for i, (name, spec) in enumerate(copula_specs.items()):
        print(f"\n  {name} copula:")
        param = spec["fit"](u, v)
        lamL, lamU = spec["tail"](param)

        # Bootstrap CI for parameter (REV1-7)
        _, p_lo, p_hi = bootstrap_copula_param_ci(u, v, spec["fit"])

        print(f"    param = {param:.4f} [{p_lo:.4f}, {p_hi:.4f}]")
        print(f"    λ_L = {lamL:.4f}, λ_U = {lamU:.4f}")

        fit_results.append({
            "copula": name, "param": param,
            "param_ci_lo": p_lo, "param_ci_hi": p_hi,
            "lambda_L": lamL, "lambda_U": lamU,
        })

        # GOF
        stat, pval, _ = cvm_gof_bootstrap(
            u, v, spec["cdf"], spec["sim"], spec["fit"], (param,),
            n_bootstrap=1000, random_state=42 + i)
        print(f"    CvM stat = {stat:.6f}, p = {pval:.4f}")
        gof_results.append({"copula": name, "cvm_stat": stat, "p_value": pval})

        # Contour + heatmap for ALL copulas (REV1-7)
        plot_copula_contours(u, v, spec["cdf"], (param,), outdir, name)
        plot_copula_heatmap(u, v, spec["cdf"], (param,), outdir, name)

    fit_df = pd.DataFrame(fit_results)
    fit_df.to_csv(outdir/"copula_parameters.csv", index=False)

    gof_df = pd.DataFrame(gof_results)
    gof_df.to_csv(outdir/"copula_gof_cvm.csv", index=False)
    print(f"\n  GOF summary:\n{gof_df.to_string(index=False)}")

    best_cop = gof_df.loc[gof_df["cvm_stat"].idxmin(), "copula"]
    selected_cvm = float(
        gof_df.loc[gof_df["copula"] == best_cop, "cvm_stat"].iloc[0]
    )
    selected_gof_p = float(
        gof_df.loc[gof_df["copula"] == best_cop, "p_value"].iloc[0]
    )
    print(f"\n  Selected copula: {best_cop}")
    print("  Selection criterion: minimum Cramer-von Mises statistic")
    print(f"  Selected CvM statistic = {selected_cvm:.6f}")
    print(f"  Bootstrap GOF p-value = {selected_gof_p:.4f}")

    # ---- REV1-8: Copula-based continuous joint risk score ----
    best_spec = copula_specs[best_cop]
    best_param = fit_df[fit_df.copula == best_cop].iloc[0]["param"]
    print(f"\n--- Copula-based joint risk score (using {best_cop}) ---")
    joint_score, joint_auc = compute_copula_joint_risk(
        u, v, best_spec["cdf"], (best_param,), y, outdir)

    # Also add joint score to ROC comparison
    plot_roc_curves(y,
                    {"clinical": risk_clin, "gene-expression": risk_gen,
                     "copula-fused": joint_score},
                    outdir,
                    ci_dict={"clinical": (lo_c, hi_c),
                             "gene-expression": (lo_g, hi_g)})

    # ---- Simple fusion baselines (is the copula needed?) ----
    print("\n" + "="*60)
    print("FUSION BASELINES (copula vs simple score-level fusion)")
    print("="*60)
    fusion_df = fusion_baselines(risk_clin, risk_gen, u, v, y, joint_score, outdir)

    # ---- Primary cancer-specific calibration (fair: same treatment for all three) ----
    print("\n" + "=" * 60)
    print("CALIBRATION (primary cancer-specific endpoint)")
    print("=" * 60)
    yv = y.values
    # Raw-probability diagnostics are valid ONLY for the two model probabilities.
    # The copula-fused score is a rank-based CDF value, NOT a probability, so it is
    # not given Brier/HL/slope/ICI until mapped to a probability by recalibration.
    print("Raw model-probability calibration (clinical, gene-expression only):")
    calibration_report({"clinical (raw)": risk_clin,
                        "gene-expression (raw)": risk_gen},
                       yv, outdir, tag="metabric_cancer_specific_raw")
    # Fair comparison: same second-stage cross-fitted recalibration of OOF scores
    # for all three, including the copula mapped to a probability.
    clin_cal, _ = cross_fitted_recalibration(yv, risk_clin, "probability")
    gene_cal, _ = cross_fitted_recalibration(yv, risk_gen, "probability")
    fused_cal, _ = cross_fitted_recalibration(yv, joint_score, "continuous")
    print("Cross-fitted recalibrated calibration (all three, fair comparison):")
    calibration_report({"clinical (recal.)": clin_cal,
                        "gene-expression (recal.)": gene_cal,
                        "copula-fused (recal.)": fused_cal},
                       yv, outdir, tag="metabric_cancer_specific_recalibrated")


    # ---- Feature importance (Editor point 3: key genes + interpretability) ----
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (permutation, ROC-AUC)")
    print("="*60)
    imp_gene = feature_importance_report(
        model_g, df, y, expr_cols, "gene-expression", outdir, topn=20)
    imp_clin = feature_importance_report(
        model_c, df, y, clinical_ml_cols, "clinical", outdir, topn=15)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE — outputs in 'study1_outputs/'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
