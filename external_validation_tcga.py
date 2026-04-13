#For TCGA

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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

RANDOM_STATE = 42
EPS = 1e-10

# ============================================================
# 1. INPUT PATHS
# ============================================================
METABRIC_PATH = 'METABRIC_RNA_Mutation.csv'
TCGA_PATH = 'tcga_final_dataset.csv'
OUTDIR = Path('harmonized_external_validation_outputs')

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
        models.append(('xgboost', XGBClassifier(
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
        oof = cross_val_predict(pipe, X, y, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
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
    df.columns = [clean_names(c) for c in df.columns]
    if 'TUMOR_STAGE' in df.columns:
        df['TUMOR_STAGE'] = _harmonize_tumor_stage(df['TUMOR_STAGE'])
    if 'AGE_AT_DIAGNOSIS' in df.columns:
        df['AGE_AT_DIAGNOSIS'] = pd.to_numeric(df['AGE_AT_DIAGNOSIS'], errors='coerce')
    return df


def load_tcga_harmonized(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [clean_names(c) for c in df.columns]
    if 'TUMOR_STAGE' in df.columns:
        df['TUMOR_STAGE'] = _harmonize_tumor_stage(df['TUMOR_STAGE'])
    if 'AGE_AT_DIAGNOSIS' in df.columns:
        df['AGE_AT_DIAGNOSIS'] = pd.to_numeric(df['AGE_AT_DIAGNOSIS'], errors='coerce')
    return df


def build_metabric_5y_overall(df, cutoff_months=60.0):
    if 'OVERALL_SURVIVAL' not in df.columns or 'OVERALL_SURVIVAL_MONTHS' not in df.columns:
        raise ValueError('METABRIC missing OVERALL_SURVIVAL or OVERALL_SURVIVAL_MONTHS')

    time = pd.to_numeric(df['OVERALL_SURVIVAL_MONTHS'], errors='coerce')
    raw = pd.to_numeric(df['OVERALL_SURVIVAL'], errors='coerce')
    # METABRIC Kaggle convention used in your pipeline: 0=dead, 1=alive
    dead = (raw == 0).astype(float)

    y = pd.Series(np.nan, index=df.index, dtype=float)
    y.loc[(dead == 1) & (time <= cutoff_months)] = 1.0
    y.loc[((dead == 1) & (time > cutoff_months)) | ((dead == 0) & (time >= cutoff_months))] = 0.0
    return y


def build_tcga_5y_overall(df, cutoff_months=60.0):
    if 'OVERALL_SURVIVAL' not in df.columns or 'OVERALL_SURVIVAL_MONTHS' not in df.columns:
        raise ValueError('TCGA missing OVERALL_SURVIVAL or OVERALL_SURVIVAL_MONTHS')

    time = pd.to_numeric(df['OVERALL_SURVIVAL_MONTHS'], errors='coerce')
    raw = pd.to_numeric(df['OVERALL_SURVIVAL'], errors='coerce')
  
    dead = (raw == 1).astype(float)

    y = pd.Series(np.nan, index=df.index, dtype=float)
    y.loc[(dead == 1) & (time <= cutoff_months)] = 1.0
    y.loc[((dead == 1) & (time > cutoff_months)) | ((dead == 0) & (time >= cutoff_months))] = 0.0
    return y


def get_common_genes_and_clinical(metabric, tcga):
    clinical_candidates = ['AGE_AT_DIAGNOSIS', 'TUMOR_STAGE']
    shared_clinical = [c for c in clinical_candidates if c in metabric.columns and c in tcga.columns]

    met_exclude = {
        'PATIENT_ID', 'OVERALL_SURVIVAL_MONTHS', 'OVERALL_SURVIVAL', 'DEATH_FROM_CANCER'
    }
    tcga_exclude = {
        'PATIENT_ID', 'OVERALL_SURVIVAL_MONTHS', 'OVERALL_SURVIVAL', 'DEATH_FROM_CANCER',
        'LYMPH_NODES', 'METASTASIS_STAGE', 'SUBTYPE', 'RACE', 'RADIATION_THERAPY'
    }

    met_gene_cols = [
        c for c in metabric.columns
        if c not in met_exclude
        and c not in shared_clinical
        and not c.endswith('_MUT')
        and pd.api.types.is_numeric_dtype(metabric[c])
    ]
    tcga_gene_cols = [
        c for c in tcga.columns
        if c not in tcga_exclude
        and c not in shared_clinical
        and pd.api.types.is_numeric_dtype(tcga[c])
    ]

    common_genes = sorted(set(met_gene_cols).intersection(tcga_gene_cols))
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
    plt.title('External validation ROC curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()

# ============================================================
# 6. MAIN
# ============================================================
def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)

    met = load_metabric_harmonized(METABRIC_PATH)
    tcga = load_tcga_harmonized(TCGA_PATH)

    y_met = build_metabric_5y_overall(met)
    y_tcga = build_tcga_5y_overall(tcga)

    met = met.loc[~y_met.isna()].reset_index(drop=True)
    y_met = y_met.loc[~y_met.isna()].astype(int).reset_index(drop=True)

    tcga = tcga.loc[~y_tcga.isna()].reset_index(drop=True)
    y_tcga = y_tcga.loc[~y_tcga.isna()].astype(int).reset_index(drop=True)

    shared_clinical, common_genes = get_common_genes_and_clinical(met, tcga)

    if len(shared_clinical) == 0:
        raise ValueError('No shared clinical predictors found.')
    if len(common_genes) == 0:
        raise ValueError('No shared gene-expression predictors found.')

    clinical_train = met[shared_clinical].copy()
    clinical_test = tcga[shared_clinical].copy()
    gene_train = met[common_genes].copy()
    gene_test = tcga[common_genes].copy()

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
    best_p = -1
    best_param = None
    best_cdf = None

    for name, spec in copulas.items():
        param = spec['fit'](u_train, v_train)
        cvm, pval = cvm_gof_bootstrap(u_train, v_train, spec['cdf'], spec['sim'], spec['fit'], (param,), n_bootstrap=300, random_state=RANDOM_STATE)
        lamL, lamU = spec['tail'](param)
        copula_rows.append({
            'copula': name,
            'param': param,
            'cvm_stat': cvm,
            'gof_p_value': pval,
            'lambda_L': lamL,
            'lambda_U': lamU,
        })
        if pval > best_p:
            best_p = pval
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
        'best_copula': best_name,
        'best_copula_param': float(best_param),
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

    with open(OUTDIR / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if HAS_PLOT:
        plot_roc(y_tcga, [clin_test_prob, gene_test_prob, fused_test], ['Clinical', 'Gene-expression', 'Copula-fused'], OUTDIR / 'external_validation_roc.png')

    print('Done.')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
