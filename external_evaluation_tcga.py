# Harmonized METABRIC-to-TCGA external evaluation

import warnings

warnings.filterwarnings('ignore')

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.integrate import quad

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

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
        import xgboost;
        v["xgboost"] = xgboost.__version__
    except Exception:
        v["xgboost"] = "not installed"
    try:
        import lifelines;
        v["lifelines"] = lifelines.__version__
    except Exception:
        v["lifelines"] = "not installed"
    return v


def xgb_actual_device(pipe):
    """Read the device the XGBoost booster actually trained on (config JSON)."""
    try:
        clf = pipe.named_steps.get("clf")
        cfg = json.loads(clf.get_booster().save_config())
        return cfg.get("learner", {}).get("generic_param", {}).get("device", "unknown")
    except Exception as e:
        return f"unknown ({e})"


try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

RANDOM_STATE = 42
EPS = 1e-10
METABRIC_DEAD_VALUE = 0
TCGA_DEAD_VALUE = 1

# This analysis evaluates a harmonized model specification. It is not a direct
# external validation of the richer primary METABRIC model because the endpoint,
# predictor set, and expression platform differ between cohorts.

# ============================================================
# 1. INPUT PATHS
# ============================================================
METABRIC_PATH = 'METABRIC_RNA_Mutation.csv'
TCGA_PATH = 'tcga_final_dataset.csv'
OUTDIR = Path('harmonized_external_evaluation_outputs')


# ============================================================
# 2. HELPERS
# ============================================================
def clean_names(x: str) -> str:
    x = str(x).strip().upper()
    x = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in x)
    if x and x[0].isdigit():
        x = 'X' + x
    while '__' in x:
        x = x.replace('__', '_')
    return x


KNOWN_NON_GENE_COLUMNS = {
    'PATIENT_ID', 'AGE_AT_DIAGNOSIS', 'TYPE_OF_BREAST_SURGERY',
    'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'CELLULARITY', 'CHEMOTHERAPY',
    'PAM50___CLAUDIN_LOW_SUBTYPE', 'COHORT', 'ER_STATUS_MEASURED_BY_IHC',
    'ER_STATUS', 'NEOPLASM_HISTOLOGIC_GRADE', 'HER2_STATUS_MEASURED_BY_SNP6',
    'HER2_STATUS', 'TUMOR_OTHER_HISTOLOGIC_SUBTYPE', 'HORMONE_THERAPY',
    'INFERRED_MENOPAUSAL_STATE', 'INTEGRATIVE_CLUSTER',
    'PRIMARY_TUMOR_LATERALITY', 'LYMPH_NODES_EXAMINED_POSITIVE',
    'MUTATION_COUNT', 'NOTTINGHAM_PROGNOSTIC_INDEX', 'ONCOTREE_CODE',
    'OVERALL_SURVIVAL_MONTHS', 'OVERALL_SURVIVAL', 'PR_STATUS',
    'RADIO_THERAPY', 'X3_GENE_CLASSIFIER_SUBTYPE', 'TUMOR_SIZE',
    'TUMOR_STAGE', 'DEATH_FROM_CANCER', 'LYMPH_NODES', 'METASTASIS_STAGE',
    'SUBTYPE', 'RACE', 'RADIATION_THERAPY', 'SEX', 'GENDER', 'VITAL_STATUS'
}


def validate_binary_status(df, dataset_name, status_col, dead_value):
    """Fail loudly when the assumed binary endpoint coding is not present."""
    if status_col not in df.columns:
        raise ValueError(f'{dataset_name} is missing required status column: {status_col}')

    status = pd.to_numeric(df[status_col], errors='coerce')
    observed = set(status.dropna().unique())
    if not observed.issubset({0, 1}):
        raise ValueError(
            f'{dataset_name} {status_col} contains unexpected values: {sorted(observed)}'
        )
    if dead_value not in observed:
        raise ValueError(
            f'Configured death value {dead_value} is absent from {dataset_name} {status_col}'
        )

    print('\n' + '=' * 60)
    print(f'{dataset_name} endpoint coding check')
    print('=' * 60)
    print(status.value_counts(dropna=False).sort_index())
    print(f'Configured death value: {dead_value}')
    return status


def validate_metabric_status_consistency(df, minimum_agreement=0.95):
    """Cross-check METABRIC overall-survival coding against cause-of-death text."""
    required = {'OVERALL_SURVIVAL', 'DEATH_FROM_CANCER'}
    if not required.issubset(df.columns):
        print('METABRIC cause-of-death text is unavailable; cross-check skipped.')
        return np.nan

    overall = pd.to_numeric(df['OVERALL_SURVIVAL'], errors='coerce')
    cause = df['DEATH_FROM_CANCER'].astype(str).str.strip().str.lower()
    informative = cause.isin({'died of disease', 'living'}) & overall.notna()
    if informative.sum() == 0:
        print('No informative METABRIC records were available for coding cross-check.')
        return np.nan

    expected_dead = cause.loc[informative].eq('died of disease').astype(int).to_numpy()
    coded_dead = overall.loc[informative].eq(METABRIC_DEAD_VALUE).astype(int).to_numpy()
    agreement = float(np.mean(expected_dead == coded_dead))

    print('\nMETABRIC survival-status cross-tabulation:')
    print(pd.crosstab(overall, cause, dropna=False))
    print(f'Agreement with configured death coding: {agreement:.4f}')
    if agreement < minimum_agreement:
        raise ValueError(
            'METABRIC endpoint coding failed the cause-of-death cross-check. '
            'Do not run the analysis until the coding is verified.'
        )
    return agreement


def expression_shift_audit(train_df, test_df, genes, outdir):
    """Quantify cross-platform distribution shift without using outcome labels."""
    rows = []
    for gene in genes:
        a = pd.to_numeric(train_df[gene], errors='coerce')
        b = pd.to_numeric(test_df[gene], errors='coerce')
        mean_a, mean_b = a.mean(), b.mean()
        sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)
        pooled = np.sqrt((sd_a ** 2 + sd_b ** 2) / 2.0) if np.isfinite(sd_a) and np.isfinite(sd_b) else np.nan
        smd = (mean_b - mean_a) / pooled if np.isfinite(pooled) and pooled > 0 else np.nan
        iqr_a = a.quantile(0.75) - a.quantile(0.25)
        iqr_b = b.quantile(0.75) - b.quantile(0.25)
        rows.append({
            'gene': gene,
            'metabric_mean': mean_a,
            'metabric_sd': sd_a,
            'metabric_median': a.median(),
            'metabric_iqr': iqr_a,
            'metabric_missing_fraction': a.isna().mean(),
            'tcga_mean': mean_b,
            'tcga_sd': sd_b,
            'tcga_median': b.median(),
            'tcga_iqr': iqr_b,
            'tcga_missing_fraction': b.isna().mean(),
            'standardized_mean_difference': smd,
            'absolute_standardized_mean_difference': abs(smd) if np.isfinite(smd) else np.nan,
        })
    audit = pd.DataFrame(rows).sort_values(
        'absolute_standardized_mean_difference', ascending=False, na_position='last'
    )
    audit.to_csv(outdir / 'expression_platform_shift_audit.csv', index=False)
    finite = audit['absolute_standardized_mean_difference'].dropna()
    summary = {
        'n_shared_genes': int(len(genes)),
        'median_absolute_smd': float(finite.median()) if len(finite) else None,
        'fraction_absolute_smd_gt_0_5': float((finite > 0.5).mean()) if len(finite) else None,
        'fraction_absolute_smd_gt_1': float((finite > 1.0).mean()) if len(finite) else None,
    }
    with open(outdir / 'expression_platform_shift_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nExpression-platform shift audit:')
    print(json.dumps(summary, indent=2))
    return audit, summary


def _one_hot_encoder():
    import sklearn
    major, minor = map(int, sklearn.__version__.split('.')[:2])
    kwargs = {'handle_unknown': 'ignore'}
    if (major, minor) >= (1, 2):
        kwargs['sparse_output'] = False
    else:
        kwargs['sparse'] = False
    return OneHotEncoder(**kwargs)


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_score[idx]))
    vals = np.asarray(vals)
    point = roc_auc_score(y_true, y_score)
    lo = np.percentile(vals, 100 * alpha / 2)
    hi = np.percentile(vals, 100 * (1 - alpha / 2))
    return point, lo, hi


def fit_and_oof(X, y, num_cols, cat_cols, random_state=42):
    numeric_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', _one_hot_encoder()),
    ])

    transformers = []
    if num_cols:
        transformers.append(('num', numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(('cat', categorical_pipe, cat_cols))
    pre = ColumnTransformer(transformers=transformers)

    models = [
        ('logistic_en', LogisticRegression(
            penalty='elasticnet', solver='saga', l1_ratio=0.5,
            max_iter=5000, class_weight='balanced',
            random_state=random_state, n_jobs=-1
        )),
        ('random_forest', RandomForestClassifier(
            n_estimators=500, min_samples_split=5, min_samples_leaf=5,
            class_weight='balanced', random_state=random_state, n_jobs=-1
        )),
        ('grad_boost', GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            random_state=random_state
        )),
    ]
    if HAS_XGB:
        models.append(('xgboost', XGBClassifier(**_XGB_KW,

                                                n_estimators=500, learning_rate=0.05, max_depth=3,
                                                subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', eval_metric='logloss',
                                                random_state=random_state, n_jobs=-1
                                                )))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    best_auc = -np.inf
    best_name = None
    best_pipe = None
    best_oof = None
    all_metrics = []

    for name, clf in models:
        pipe = Pipeline([('pre', pre), ('clf', clf)])
        cv_njobs = 1 if (name == 'xgboost' and _USE_GPU) else -1
        oof = cross_val_predict(pipe, X, y, cv=skf, method='predict_proba', n_jobs=cv_njobs)[:, 1]
        auc_val = roc_auc_score(y, oof)
        all_metrics.append({'model': name, 'cv_auc': auc_val})
        if auc_val > best_auc:
            best_auc = auc_val
            best_name = name
            best_oof = oof
            pipe.fit(X, y)
            best_pipe = pipe

    return best_name, best_pipe, best_oof, pd.DataFrame(all_metrics)


def ecdf_from_train(train_scores, test_scores):
    train_scores = np.asarray(train_scores)
    test_scores = np.asarray(test_scores)
    sorter = np.sort(train_scores)
    ranks = np.searchsorted(sorter, test_scores, side='right')
    return np.clip(ranks / (len(sorter) + 1.0), EPS, 1 - EPS)


# ============================================================
# 3. COPULAS
# ============================================================
def empirical_copula_C(u, v):
    n = len(u)
    out = np.empty(n)
    for i in range(n):
        out[i] = np.mean((u <= u[i]) & (v <= v[i]))
    return out


def copula_gaussian_cdf(u, v, rho):
    u = np.clip(np.asarray(u), EPS, 1 - EPS)
    v = np.clip(np.asarray(v), EPS, 1 - EPS)
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    pts = np.column_stack([x, y])
    return stats.multivariate_normal.cdf(pts, mean=[0, 0], cov=[[1, rho], [rho, 1]])


def fit_gaussian(u, v):
    tau = stats.kendalltau(u, v)[0]
    return float(np.sin(np.pi * tau / 2.0))


def tail_gaussian(rho):
    if abs(rho) < 1 - 1e-8:
        return 0.0, 0.0
    return 1.0, 1.0


def simulate_gaussian(n, rho, random_state=None):
    rng = np.random.default_rng(random_state)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return stats.norm.cdf(z[:, 0]), stats.norm.cdf(z[:, 1])


def copula_clayton_cdf(u, v, theta):
    u = np.clip(np.asarray(u), EPS, 1 - EPS)
    v = np.clip(np.asarray(v), EPS, 1 - EPS)
    return np.maximum((u ** (-theta) + v ** (-theta) - 1.0) ** (-1.0 / theta), EPS)


def fit_clayton(u, v):
    tau = max(stats.kendalltau(u, v)[0], 1e-6)
    return float(2 * tau / (1 - tau))


def tail_clayton(theta):
    if theta > 0:
        return float(2 ** (-1.0 / theta)), 0.0
    return 0.0, 0.0


def _simulate_archimedean(n, theta, family, random_state=None):
    rng = np.random.default_rng(random_state)
    s = rng.uniform(size=n)
    t = rng.uniform(size=n)
    if family == 'clayton':
        phi = lambda x: (x ** (-theta) - 1) / theta
        phi_inv = lambda w: (1 + theta * w) ** (-1 / theta)
        phi_p = lambda x: -x ** (-(theta + 1))
    elif family == 'gumbel':
        phi = lambda x: (-np.log(x)) ** theta
        phi_inv = lambda w: np.exp(-w ** (1 / theta))
        phi_p = lambda x: -theta * (-np.log(x)) ** (theta - 1) / x
    else:
        raise ValueError('unknown family')
    K = lambda x: x - phi(x) / phi_p(x)
    xg = np.linspace(1e-6, 1 - 1e-6, 5000)
    Kg = K(xg)
    idx = np.argsort(Kg)
    Ks = Kg[idx]
    xs = xg[idx]
    w = np.interp(np.clip(t, Ks[0], Ks[-1]), Ks, xs)
    w = np.clip(w, EPS, 1 - EPS)
    pw = phi(w)
    u = phi_inv(s * pw)
    v = phi_inv((1 - s) * pw)
    return np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)


def simulate_clayton(n, theta, random_state=None):
    return _simulate_archimedean(n, theta, 'clayton', random_state)


def copula_gumbel_cdf(u, v, theta):
    u = np.clip(np.asarray(u), EPS, 1 - EPS)
    v = np.clip(np.asarray(v), EPS, 1 - EPS)
    t = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1.0 / theta)
    return np.exp(-t)


def fit_gumbel(u, v):
    tau = max(stats.kendalltau(u, v)[0], 0.0)
    return float(max(1.0 / (1.0 - tau + 1e-6), 1.0))


def tail_gumbel(theta):
    return 0.0, float(2 - 2 ** (1.0 / max(theta, 1.0)))


def simulate_gumbel(n, theta, random_state=None):
    return _simulate_archimedean(n, theta, 'gumbel', random_state)


def copula_frank_cdf(u, v, theta):
    u = np.clip(np.asarray(u), EPS, 1 - EPS)
    v = np.clip(np.asarray(v), EPS, 1 - EPS)
    if abs(theta) < 1e-8:
        return u * v
    num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
    den = np.exp(-theta) - 1
    return -np.log(1 + num / den) / theta


def _debye1(t):
    if abs(t) < 1e-8:
        return 1.0
    val, _ = quad(lambda x: x / (np.exp(x) - 1), 0, t)
    return val / t


def _tau_frank(theta):
    if abs(theta) < 1e-8:
        return 0.0
    return 1 - 4 / theta * (1 - _debye1(theta))


def fit_frank(u, v):
    tau_obs = stats.kendalltau(u, v)[0]
    if abs(tau_obs) < 1e-8:
        return 0.0
    try:
        return float(brentq(lambda th: _tau_frank(th) - tau_obs, -50, 50))
    except Exception:
        return 0.0


def tail_frank(theta):
    return 0.0, 0.0


def simulate_frank(n, theta, random_state=None):
    rng = np.random.default_rng(random_state)
    u = rng.uniform(size=n)
    p = rng.uniform(size=n)
    if abs(theta) < 1e-8:
        return u, p
    a = np.exp(-theta * u)
    b = np.exp(-theta)
    v = -np.log(1 + p * (b - 1) / (a * (1 - p) + p)) / theta
    return np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)


def cvm_statistic(u, v, copula_cdf, params):
    Cn = empirical_copula_C(u, v)
    Ct = copula_cdf(u, v, *params)
    return float(np.mean((Cn - Ct) ** 2))


def cvm_gof_bootstrap(u, v, cdf_fn, sim_fn, fit_fn, params, n_bootstrap=300, random_state=42):
    rng = np.random.default_rng(random_state)
    obs = cvm_statistic(u, v, cdf_fn, params)
    sims = []
    for _ in range(n_bootstrap):
        us, vs = sim_fn(len(u), *params, random_state=rng.integers(1_000_000_000))
        param_b = fit_fn(us, vs)
        sims.append(cvm_statistic(us, vs, cdf_fn, (param_b,)))
    sims = np.asarray(sims)
    pval = (1 + np.sum(sims >= obs)) / (n_bootstrap + 1)
    return obs, float(pval)


# ============================================================
# 4. DATA BUILDING
# ============================================================
def _harmonize_tumor_stage(series):
    s = series.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=series.index, dtype=object)

    stage_iv = s.str.contains(r'\bIV\b', regex=True, na=False)
    stage_iii = s.str.contains(r'\bIII\b', regex=True, na=False)
    stage_ii = s.str.contains(r'\bII\b', regex=True, na=False)
    stage_i = s.str.contains(r'\bI\b', regex=True, na=False)

    num = pd.to_numeric(series, errors='coerce')
    stage_iv = stage_iv | (num >= 4)
    stage_iii = stage_iii | ((num >= 3) & (num < 4))
    stage_ii = stage_ii | ((num >= 2) & (num < 3))
    stage_i = stage_i | ((num >= 1) & (num < 2))

    out.loc[stage_iv] = 'Stage IV'
    out.loc[~stage_iv & stage_iii] = 'Stage III'
    out.loc[~stage_iv & ~stage_iii & stage_ii] = 'Stage II'
    out.loc[~stage_iv & ~stage_iii & ~stage_ii & stage_i] = 'Stage I'
    return out


def load_metabric_harmonized(path):
    df = pd.read_csv(path, low_memory=False)
    if STRICT_INPUT_CHECKS:
        assert df.shape == (1904, 693), (
            f"METABRIC shape {df.shape} != (1904, 693); input truncated or wrong.")
    df.columns = [clean_names(c) for c in df.columns]
    if 'TUMOR_STAGE' in df.columns:
        df['TUMOR_STAGE'] = _harmonize_tumor_stage(df['TUMOR_STAGE'])
    if 'AGE_AT_DIAGNOSIS' in df.columns:
        df['AGE_AT_DIAGNOSIS'] = pd.to_numeric(df['AGE_AT_DIAGNOSIS'], errors='coerce')
    return df


def load_tcga_harmonized(path):
    df = pd.read_csv(path, low_memory=False)
    if STRICT_INPUT_CHECKS:
        assert df.shape == (981, 488), (
            f"TCGA shape {df.shape} != (981, 488); input truncated or wrong.")
    df.columns = [clean_names(c) for c in df.columns]
    if 'TUMOR_STAGE' in df.columns:
        df['TUMOR_STAGE'] = _harmonize_tumor_stage(df['TUMOR_STAGE'])
    if 'AGE_AT_DIAGNOSIS' in df.columns:
        df['AGE_AT_DIAGNOSIS'] = pd.to_numeric(df['AGE_AT_DIAGNOSIS'], errors='coerce')
    return df


def build_metabric_5y_overall(df, cutoff_months=60.0):
    required = {'OVERALL_SURVIVAL', 'OVERALL_SURVIVAL_MONTHS'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'METABRIC is missing required columns: {sorted(missing)}')

    time = pd.to_numeric(df['OVERALL_SURVIVAL_MONTHS'], errors='coerce')
    raw = pd.to_numeric(df['OVERALL_SURVIVAL'], errors='coerce')
    dead = raw.eq(METABRIC_DEAD_VALUE).astype(float)

    y = pd.Series(np.nan, index=df.index, dtype=float)
    y.loc[(dead == 1) & (time <= cutoff_months)] = 1.0
    y.loc[((dead == 1) & (time > cutoff_months)) |
          ((dead == 0) & (time >= cutoff_months))] = 0.0
    print(
        f'METABRIC 5-year overall mortality: {(y == 1).sum()} events, '
        f'{(y == 0).sum()} non-events, {y.isna().sum()} excluded'
    )
    return y


def build_tcga_5y_overall(df, cutoff_months=60.0):
    required = {'OVERALL_SURVIVAL', 'OVERALL_SURVIVAL_MONTHS'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'TCGA is missing required columns: {sorted(missing)}')

    time = pd.to_numeric(df['OVERALL_SURVIVAL_MONTHS'], errors='coerce')
    raw = pd.to_numeric(df['OVERALL_SURVIVAL'], errors='coerce')
    dead = raw.eq(TCGA_DEAD_VALUE).astype(float)

    y = pd.Series(np.nan, index=df.index, dtype=float)
    y.loc[(dead == 1) & (time <= cutoff_months)] = 1.0
    y.loc[((dead == 1) & (time > cutoff_months)) |
          ((dead == 0) & (time >= cutoff_months))] = 0.0
    print(
        f'TCGA 5-year overall mortality: {(y == 1).sum()} events, '
        f'{(y == 0).sum()} non-events, {y.isna().sum()} excluded'
    )
    return y


def get_common_genes_and_clinical(metabric, tcga):
    clinical_candidates = ['AGE_AT_DIAGNOSIS', 'TUMOR_STAGE']
    shared_clinical = [
        c for c in clinical_candidates
        if c in metabric.columns and c in tcga.columns
    ]

    met_gene_cols = {
        c for c in metabric.columns
        if c not in KNOWN_NON_GENE_COLUMNS
           and not c.endswith('_MUT')
           and pd.api.types.is_numeric_dtype(metabric[c])
           and pd.to_numeric(metabric[c], errors='coerce').nunique(dropna=True) > 1
    }
    tcga_gene_cols = {
        c for c in tcga.columns
        if c not in KNOWN_NON_GENE_COLUMNS
           and not c.endswith('_MUT')
           and pd.api.types.is_numeric_dtype(tcga[c])
           and pd.to_numeric(tcga[c], errors='coerce').nunique(dropna=True) > 1
    }

    common_genes = sorted(met_gene_cols.intersection(tcga_gene_cols))
    return shared_clinical, common_genes


# ============================================================
# 5. PLOTTING
# ============================================================
def plot_roc(y_true, scores, labels, savepath):
    if not HAS_PLOT:
        return
    plt.figure(figsize=(6, 6))
    for s, lab in zip(scores, labels):
        fpr, tpr, _ = roc_curve(y_true, s)
        plt.plot(fpr, tpr, label=f'{lab} (AUC={auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Independent TCGA external evaluation ROC curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


# ============================================================
# 5b. CALIBRATION  (Editor point 1: Hosmer-Lemeshow etc.)
# ============================================================
def hosmer_lemeshow(y, p, g=10):
    y = np.asarray(y, float);
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p, kind='mergesort')
    grp = np.empty(n, int)
    grp[order] = np.floor(np.arange(n) * g / n).astype(int)
    chi2 = 0.0;
    used = 0
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
    return {'chi2': float(chi2), 'dof': int(dof),
            'p_value': float(stats.chi2.sf(chi2, dof)), 'n_bins': used}


def _fit_unpenalized_logistic(X, y):
    """Support both current and older scikit-learn penalty syntax."""
    last_error = None
    for penalty in (None, 'none'):
        try:
            model = LogisticRegression(
                penalty=penalty, solver='lbfgs', max_iter=5000
            )
            model.fit(X, y)
            return model
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError('Unable to fit unpenalized logistic regression.') from last_error


def safe_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _calibration_predictor(score, score_type):
    score = np.asarray(score, dtype=float)
    if score_type == 'probability':
        score = safe_logit(score)
    elif score_type != 'continuous':
        raise ValueError("score_type must be 'probability' or 'continuous'")
    return score.reshape(-1, 1)


def fit_recalibration_model(y, score, score_type):
    return _fit_unpenalized_logistic(
        _calibration_predictor(score, score_type), np.asarray(y, dtype=int)
    )


def apply_recalibration_model(model, score, score_type):
    X = _calibration_predictor(score, score_type)
    return model.predict_proba(X)[:, 1]


def cross_fitted_recalibration(y, score, score_type, n_splits=5, random_state=42):
    """Second-stage cross-fitted recalibration of out-of-fold scores plus a final transport map."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score) or np.isnan(score).any():
        raise ValueError('Invalid score vector supplied for recalibration.')

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    calibrated_oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        model = fit_recalibration_model(
            y[train_idx], score[train_idx], score_type
        )
        calibrated_oof[test_idx] = apply_recalibration_model(
            model, score[test_idx], score_type
        )
    if np.isnan(calibrated_oof).any():
        raise RuntimeError('Cross-fitted recalibration produced missing predictions.')

    final_model = fit_recalibration_model(y, score, score_type)
    return calibrated_oof, final_model


def calibration_slope_intercept(y, p, eps=1e-6):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    logit_p = safe_logit(p, eps=eps).reshape(-1, 1)
    slope_model = _fit_unpenalized_logistic(logit_p, y)
    slope = float(slope_model.coef_[0, 0])

    intercept = 0.0
    for _ in range(100):
        mu = 1.0 / (1.0 + np.exp(-(intercept + logit_p.ravel())))
        gradient = np.sum(y - mu)
        hessian = -np.sum(mu * (1.0 - mu))
        if abs(hessian) < 1e-12:
            break
        step = gradient / hessian
        intercept -= step
        if abs(step) < 1e-10:
            break
    return {'slope': slope, 'intercept': float(intercept)}


def integrated_calibration_index(y, p):
    from sklearn.isotonic import IsotonicRegression
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    observed = IsotonicRegression(out_of_bounds='clip').fit_transform(p, y)
    return float(np.mean(np.abs(observed - p)))


def calibration_table(scores_dict, y, tag):
    y = np.asarray(y, float)
    rows = []
    for name, p in scores_dict.items():
        p = np.asarray(p, float)
        hl = hosmer_lemeshow(y, p)
        si = calibration_slope_intercept(y, p)
        rows.append({'cohort': tag, 'score': name,
                     'brier': float(np.mean((p - y) ** 2)),
                     'hl_chi2': hl['chi2'], 'hl_dof': hl['dof'],
                     'hl_p_value': hl['p_value'],
                     'calibration_slope': si['slope'],
                     'calibration_intercept': si['intercept'],
                     'ici': integrated_calibration_index(y, p)})
        print(f"  [{tag}:{name}] Brier={rows[-1]['brier']:.4f}  "
              f"HL p={hl['p_value']:.3f}  slope={si['slope']:.3f}  "
              f"intercept={si['intercept']:.3f}  ICI={rows[-1]['ici']:.4f}")
    return rows


def format_p(p):
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def plot_calibration(scores_dict, y, savepath, title):
    if not HAS_PLOT:
        return
    from sklearn.calibration import calibration_curve
    y = np.asarray(y, float)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect calibration')
    for name, p in scores_dict.items():
        p = np.asarray(p, float)
        nb = min(10, max(3, int(len(np.unique(p)) // 2)))
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=nb, strategy='quantile')
        hl = hosmer_lemeshow(y, p)
        plt.plot(mean_pred, frac_pos, marker='o', label=f"{name} (HL {format_p(hl['p_value'])})")
    plt.xlabel('Mean predicted risk');
    plt.ylabel('Observed event fraction')
    plt.title(title);
    plt.legend(loc='upper left')
    plt.tight_layout();
    plt.savefig(savepath, dpi=300);
    plt.close()


# ============================================================
# 6. MAIN
# ============================================================
def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)

    met = load_metabric_harmonized(METABRIC_PATH)
    tcga = load_tcga_harmonized(TCGA_PATH)

    validate_binary_status(
        met, 'METABRIC', 'OVERALL_SURVIVAL', METABRIC_DEAD_VALUE
    )
    metabric_status_agreement = validate_metabric_status_consistency(met)
    validate_binary_status(
        tcga, 'TCGA', 'OVERALL_SURVIVAL', TCGA_DEAD_VALUE
    )

    y_met = build_metabric_5y_overall(met)
    y_tcga = build_tcga_5y_overall(tcga)
    if STRICT_INPUT_CHECKS:
        _e = int((y_met == 1).sum());
        _n = int((y_met == 0).sum());
        _x = int(y_met.isna().sum())
        assert (_e, _n, _x) == (412, 1432, 60), (
            f"METABRIC 5y overall endpoint {(_e, _n, _x)} != (412, 1432, 60).")
        _e = int((y_tcga == 1).sum());
        _n = int((y_tcga == 0).sum());
        _x = int(y_tcga.isna().sum())
        assert (_e, _n, _x) == (85, 232, 664), (
            f"TCGA 5y overall endpoint {(_e, _n, _x)} != (85, 232, 664).")

    met = met.loc[~y_met.isna()].reset_index(drop=True)
    y_met = y_met.loc[~y_met.isna()].astype(int).reset_index(drop=True)

    tcga = tcga.loc[~y_tcga.isna()].reset_index(drop=True)
    y_tcga = y_tcga.loc[~y_tcga.isna()].astype(int).reset_index(drop=True)

    shared_clinical, common_genes = get_common_genes_and_clinical(met, tcga)
    if STRICT_INPUT_CHECKS:
        assert len(common_genes) == 476, (
            f"Shared gene count {len(common_genes)} != 476; input may be wrong.")

    if len(shared_clinical) == 0:
        raise ValueError('No shared clinical predictors found.')
    if len(common_genes) == 0:
        raise ValueError('No shared gene-expression predictors found.')

    clinical_train = met[shared_clinical].copy()
    clinical_test = tcga[shared_clinical].copy()
    gene_train = met[common_genes].copy()
    gene_test = tcga[common_genes].copy()

    pd.DataFrame({'shared_gene': common_genes}).to_csv(
        OUTDIR / 'shared_gene_manifest.csv', index=False
    )
    expression_audit, expression_shift_summary = expression_shift_audit(
        gene_train, gene_test, common_genes, OUTDIR
    )
    with open(OUTDIR / 'harmonization_manifest.json', 'w') as f:
        json.dump({
            'analysis_label': 'independent external evaluation of a harmonized specification',
            'shared_clinical_predictors': shared_clinical,
            'n_shared_genes': len(common_genes),
            'metabric_endpoint': '5-year overall mortality',
            'tcga_endpoint': '5-year overall mortality',
            'primary_metabric_model_directly_validated': False,
            'reason_not_direct_validation': [
                'different endpoint from the primary cancer-specific analysis',
                'reduced shared clinical predictor set',
                'reduced shared gene set',
                'microarray versus RNA-seq platform mismatch'
            ]
        }, f, indent=2)

    for c in shared_clinical:
        if c == 'AGE_AT_DIAGNOSIS':
            clinical_train[c] = pd.to_numeric(clinical_train[c], errors='coerce')
            clinical_test[c] = pd.to_numeric(clinical_test[c], errors='coerce')
        else:
            clinical_train[c] = clinical_train[c].astype(object)
            clinical_test[c] = clinical_test[c].astype(object)

    clin_num = [c for c in shared_clinical if pd.api.types.is_numeric_dtype(clinical_train[c])]
    clin_cat = [c for c in shared_clinical if c not in clin_num]
    gene_num = common_genes
    gene_cat = []

    # Clinical model
    clin_name, clin_pipe, clin_oof, clin_cv_df = fit_and_oof(
        clinical_train, y_met, clin_num, clin_cat, random_state=RANDOM_STATE
    )
    clin_test_prob = clin_pipe.predict_proba(clinical_test)[:, 1]

    # Gene model
    gene_name, gene_pipe, gene_oof, gene_cv_df = fit_and_oof(
        gene_train, y_met, gene_num, gene_cat, random_state=RANDOM_STATE
    )
    gene_test_prob = gene_pipe.predict_proba(gene_test)[:, 1]

    # ---- Verify actual XGBoost device + record environment (Phoenix point 7) ----
    _xgb_device_used = None
    for _nm, _pipe in [(clin_name, clin_pipe), (gene_name, gene_pipe)]:
        if _nm == 'xgboost':
            _xgb_device_used = xgb_actual_device(_pipe)
            print(f"  [device check] XGBoost booster trained on: {_xgb_device_used}")
    with open(OUTDIR / 'run_environment.json', 'w') as _f:
        json.dump({'xgboost_config_device': ('cuda' if _USE_GPU else 'cpu'),
                   'xgboost_actual_booster_device': _xgb_device_used,
                   'package_versions': _pkg_versions(), 'seed': RANDOM_STATE,
                   'strict_input_checks': STRICT_INPUT_CHECKS}, _f, indent=2)

    # Internal METABRIC CV AUCs with bootstrap CIs
    clin_cv_auc, clin_cv_lo, clin_cv_hi = bootstrap_auc_ci(y_met, clin_oof, seed=RANDOM_STATE)
    gene_cv_auc, gene_cv_lo, gene_cv_hi = bootstrap_auc_ci(y_met, gene_oof, seed=RANDOM_STATE)

    # Copula fit on OOF training scores only
    u_train = np.clip(stats.rankdata(clin_oof, method='average') / (len(clin_oof) + 1.0), EPS, 1 - EPS)
    v_train = np.clip(stats.rankdata(gene_oof, method='average') / (len(gene_oof) + 1.0), EPS, 1 - EPS)

    copulas = {
        'Gaussian': {'fit': fit_gaussian, 'cdf': copula_gaussian_cdf, 'sim': simulate_gaussian, 'tail': tail_gaussian},
        'Clayton': {'fit': fit_clayton, 'cdf': copula_clayton_cdf, 'sim': simulate_clayton, 'tail': tail_clayton},
        'Gumbel': {'fit': fit_gumbel, 'cdf': copula_gumbel_cdf, 'sim': simulate_gumbel, 'tail': tail_gumbel},
        'Frank': {'fit': fit_frank, 'cdf': copula_frank_cdf, 'sim': simulate_frank, 'tail': tail_frank},
    }

    copula_rows = []
    best_name = None
    best_cvm = np.inf
    best_gof_p = np.nan
    best_param = None
    best_cdf = None

    for name, spec in copulas.items():
        param = spec['fit'](u_train, v_train)
        cvm, pval = cvm_gof_bootstrap(u_train, v_train, spec['cdf'], spec['sim'], spec['fit'], (param,),
                                      n_bootstrap=300, random_state=RANDOM_STATE)
        lamL, lamU = spec['tail'](param)
        copula_rows.append({
            'copula': name,
            'param': param,
            'cvm_stat': cvm,
            'gof_p_value': pval,
            'lambda_L': lamL,
            'lambda_U': lamU,
        })
        if cvm < best_cvm:
            best_cvm = cvm
            best_gof_p = pval
            best_name = name
            best_param = param
            best_cdf = spec['cdf']

    # Internal copula-fused score (on METABRIC OOF)
    fused_train = best_cdf(u_train, v_train, best_param)
    fused_cv_auc, fused_cv_lo, fused_cv_hi = bootstrap_auc_ci(y_met, fused_train, seed=RANDOM_STATE)
    # External scores
    u_test = ecdf_from_train(clin_oof, clin_test_prob)
    v_test = ecdf_from_train(gene_oof, gene_test_prob)
    fused_test = best_cdf(u_test, v_test, best_param)

    # Metrics
    clin_auc, clin_lo, clin_hi = bootstrap_auc_ci(y_tcga, clin_test_prob, seed=RANDOM_STATE)
    gene_auc, gene_lo, gene_hi = bootstrap_auc_ci(y_tcga, gene_test_prob, seed=RANDOM_STATE)
    fused_auc, fused_lo, fused_hi = bootstrap_auc_ci(y_tcga, fused_test, seed=RANDOM_STATE)

    # ---- Simple fusion baselines in TCGA (is the copula needed externally?) ----
    # Rank-average and probability-average require no training; the logistic
    # stacker is trained on the METABRIC out-of-fold scores and applied unchanged
    # to TCGA, exactly like the base models and the copula.
    rank_avg_test = (u_test + v_test) / 2.0
    prob_avg_test = (clin_test_prob + gene_test_prob) / 2.0
    stacker = LogisticRegression(max_iter=1000).fit(
        np.column_stack([clin_oof, gene_oof]), y_met)
    stack_test = stacker.predict_proba(
        np.column_stack([clin_test_prob, gene_test_prob]))[:, 1]

    ext_baselines = {
        'clinical': clin_test_prob,
        'gene-expression': gene_test_prob,
        'copula-fused': fused_test,
        'rank-average': rank_avg_test,
        'probability-average': prob_avg_test,
        'logistic-stacking': stack_test,
    }
    print('\n' + '=' * 60)
    print('FUSION BASELINES IN TCGA (copula vs simple fusion)')
    print('=' * 60)
    baseline_rows = []
    for name, s in ext_baselines.items():
        a, lo, hi = bootstrap_auc_ci(y_tcga, s, seed=RANDOM_STATE)
        baseline_rows.append({'score': name, 'external_auc': a,
                              'ci_lo': lo, 'ci_hi': hi})
        print(f"  {name:22s} AUC = {a:.4f} [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(baseline_rows).to_csv(OUTDIR / 'fusion_baselines_tcga.csv', index=False)

    summary = {
        'metabric_n_after_endpoint_filter': int(len(met)),
        'tcga_n_after_endpoint_filter': int(len(tcga)),
        'metabric_events_5y_overall': int(y_met.sum()),
        'tcga_events_5y_overall': int(y_tcga.sum()),
        'shared_clinical_predictors': shared_clinical,
        'n_shared_clinical_predictors': int(len(shared_clinical)),
        'n_shared_genes': int(len(common_genes)),
        'best_clinical_model': clin_name,
        'best_gene_model': gene_name,
        'selected_copula': best_name,
        'selected_copula_param': float(best_param),
        'copula_selection_criterion': 'minimum Cramer-von Mises statistic',
        'selected_copula_cvm': float(best_cvm),
        'selected_copula_gof_p_value': float(best_gof_p),
        'metabric_status_coding_agreement': (
            None if np.isnan(metabric_status_agreement)
            else float(metabric_status_agreement)
        ),
        'external_analysis_label': 'independent external evaluation of harmonized specification',
        'expression_shift_summary': expression_shift_summary,
        'external_auc_clinical': float(clin_auc),
        'external_auc_gene': float(gene_auc),
        'external_auc_fused': float(fused_auc),
        'internal_auc_clinical': float(clin_cv_auc),
        'internal_ci_clinical': [float(clin_cv_lo), float(clin_cv_hi)],
        'internal_auc_gene': float(gene_cv_auc),
        'internal_ci_gene': [float(gene_cv_lo), float(gene_cv_hi)],
        'internal_auc_fused': float(fused_cv_auc),
        'internal_ci_fused': [float(fused_cv_lo), float(fused_cv_hi)],
        'external_ci_clinical': [float(clin_lo), float(clin_hi)],
        'external_ci_gene': [float(gene_lo), float(gene_hi)],
        'external_ci_fused': [float(fused_lo), float(fused_hi)]
    }

    pd.DataFrame(copula_rows).to_csv(OUTDIR / 'copula_fit_summary.csv', index=False)
    clin_cv_df.to_csv(OUTDIR / 'clinical_cv_models.csv', index=False)
    gene_cv_df.to_csv(OUTDIR / 'gene_cv_models.csv', index=False)

    pd.DataFrame({
        'metric': ['clinical', 'gene_expression', 'copula_fused'],
        'internal_auc': [clin_cv_auc, gene_cv_auc, fused_cv_auc],
        'internal_ci_lo': [clin_cv_lo, gene_cv_lo, fused_cv_lo],
        'internal_ci_hi': [clin_cv_hi, gene_cv_hi, fused_cv_hi],
        'external_auc': [clin_auc, gene_auc, fused_auc],
        'external_ci_lo': [clin_lo, gene_lo, fused_lo],
        'external_ci_hi': [clin_hi, gene_hi, fused_hi],
    }).to_csv(OUTDIR / 'validation_auc_comparison.csv', index=False)

    pd.DataFrame({
        'patient_id': tcga['PATIENT_ID'] if 'PATIENT_ID' in tcga.columns else np.arange(len(tcga)),
        'y_5y_overall': y_tcga,
        'clinical_score': clin_test_prob,
        'gene_score': gene_test_prob,
        'copula_fused_score': fused_test,
        'u_test': u_test,
        'v_test': v_test,
    }).to_csv(OUTDIR / 'tcga_external_scores.csv', index=False)

    if HAS_PLOT:
        plot_roc(
            y_tcga,
            [clin_test_prob, gene_test_prob, fused_test],
            ['Clinical', 'Gene-expression', 'Copula-fused'],
            OUTDIR / 'external_evaluation_roc.png'
        )

    # Fair calibration comparison. Every score receives the same treatment:
    # cross-fitted recalibration internally, then a final METABRIC map transported
    # unchanged to TCGA.
    print('\n' + '=' * 60)
    print('CALIBRATION ASSESSMENT: 5-YEAR OVERALL MORTALITY')
    print('=' * 60)

    clinical_cal_oof, clinical_cal_model = cross_fitted_recalibration(
        y_met, clin_oof, 'probability', random_state=RANDOM_STATE
    )
    gene_cal_oof, gene_cal_model = cross_fitted_recalibration(
        y_met, gene_oof, 'probability', random_state=RANDOM_STATE
    )
    fused_cal_oof, fused_cal_model = cross_fitted_recalibration(
        y_met, fused_train, 'continuous', random_state=RANDOM_STATE
    )

    clinical_cal_tcga = apply_recalibration_model(
        clinical_cal_model, clin_test_prob, 'probability'
    )
    gene_cal_tcga = apply_recalibration_model(
        gene_cal_model, gene_test_prob, 'probability'
    )
    fused_cal_tcga = apply_recalibration_model(
        fused_cal_model, fused_test, 'continuous'
    )

    internal_scores = {
        'clinical recalibrated': clinical_cal_oof,
        'gene-expression recalibrated': gene_cal_oof,
        'copula-fused recalibrated': fused_cal_oof,
    }
    external_scores = {
        'clinical recalibrated': clinical_cal_tcga,
        'gene-expression recalibrated': gene_cal_tcga,
        'copula-fused recalibrated': fused_cal_tcga,
    }

    calibration_rows = []
    calibration_rows += calibration_table(
        internal_scores, y_met, 'METABRIC internal cross-fitted'
    )
    calibration_rows += calibration_table(
        external_scores, y_tcga, 'TCGA external evaluation'
    )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(OUTDIR / 'calibration_summary.csv', index=False)

    plot_calibration(
        internal_scores, y_met,
        OUTDIR / 'calibration_metabric_internal_cross_fitted.png',
        'Cross-fitted calibration in METABRIC (5-year overall mortality)'
    )
    plot_calibration(
        external_scores, y_tcga,
        OUTDIR / 'calibration_tcga_external_evaluation.png',
        'External calibration in TCGA (5-year overall mortality)'
    )

    pd.DataFrame({
        'patient_id': (
            tcga['PATIENT_ID'] if 'PATIENT_ID' in tcga.columns
            else np.arange(len(tcga))
        ),
        'y_5y_overall': y_tcga,
        'clinical_raw_probability': clin_test_prob,
        'gene_raw_probability': gene_test_prob,
        'copula_raw_score': fused_test,
        'clinical_calibrated_probability': clinical_cal_tcga,
        'gene_calibrated_probability': gene_cal_tcga,
        'copula_calibrated_probability': fused_cal_tcga,
    }).to_csv(OUTDIR / 'tcga_external_calibrated_scores.csv', index=False)

    summary['calibration_file'] = 'calibration_summary.csv'
    summary['calibration_interpretation_warning'] = (
        'Calibration supports probability reliability. It does not establish clinical utility.'
    )
    with open(OUTDIR / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('Done.')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
