# ============================================================
# RETINA AI: Student Dropout Risk — ULTIMATE MAX PERFORMANCE
# Pipeline: Tabular + Time-Series + NLP + CatBoost +
#           Optuna Tuning + Pseudo Labeling + 2-Level Stacking
#           + Optimal Threshold Tuning
# ============================================================

import pandas as pd
import numpy as np
import warnings
import re
import os
import time
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# ── ML Libraries ─────────────────────────────────────────────
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import skew, kurtosis

# ── CatBoost ──────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    print("  ✔ CatBoost available")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("  ⚠ CatBoost not installed — run: pip install catboost")
    print("    Continuing without CatBoost...")

# ── Optuna ────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    print("  ✔ Optuna available")
except ImportError:
    OPTUNA_AVAILABLE = False
    print("  ⚠ Optuna not installed — run: pip install optuna")
    print("    Continuing with default params...")

# ── NLP Libraries ─────────────────────────────────────────────
import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords',    quiet=True)
nltk.download('punkt',        quiet=True)
nltk.download('wordnet',      quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# ── Paths ─────────────────────────────────────────────────────
BASE            = r"D:\JAVA\projects\project ABES"
TRAIN_PATH      = os.path.join(BASE, "train.csv")
TEST_PATH       = os.path.join(BASE, "test.csv")
ATTENDANCE_PATH = os.path.join(BASE, "Attendance_series.csv")
COUNSELLOR_PATH = os.path.join(BASE, "Counsellor_notes.csv")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")
OUTPUT_PATH     = os.path.join(BASE, "submission_v3.csv")

print("\n" + "=" * 60)
print("  RETINA AI — ULTIMATE MAX PERFORMANCE")
print("=" * 60)
start_time = time.time()


# ╔══════════════════════════════════════════════════════════╗
# ║              STEP 1: LOAD DATA                           ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[1/9] Loading datasets...")

train      = pd.read_csv(TRAIN_PATH)
test       = pd.read_csv(TEST_PATH)
attendance = pd.read_csv(ATTENDANCE_PATH)
counsellor = pd.read_csv(COUNSELLOR_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(f"  ✔ Train      : {train.shape}")
print(f"  ✔ Test       : {test.shape}")
print(f"  ✔ Attendance : {attendance.shape}")
print(f"  ✔ Counsellor : {counsellor.shape}")

TARGET    = 'dropout_risk'
target    = train[TARGET].copy()
train_ids = train['student_id'].copy()
test_ids  = test['student_id'].copy()

class_counts  = np.bincount(target.values)
total         = len(target)
class_weights = {i: total / (len(class_counts) * class_counts[i])
                 for i in range(len(class_counts))}
print(f"  ✔ Class weights: {class_weights}")


# ╔══════════════════════════════════════════════════════════╗
# ║         STEP 2: ATTENDANCE TIME-SERIES FEATURES          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[2/9] Engineering attendance features...")

def compute_trend(series):
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    try:
        return np.polyfit(x, series, 1)[0]
    except:
        return 0.0

attendance = attendance.sort_values(['student_id', 'semester', 'week'])

att_global = attendance.groupby('student_id')['attendance_pct'].agg(
    att_mean   = 'mean',
    att_std    = 'std',
    att_min    = 'min',
    att_max    = 'max',
    att_median = 'median',
    att_q10    = lambda x: x.quantile(0.10),
    att_q25    = lambda x: x.quantile(0.25),
    att_q75    = lambda x: x.quantile(0.75),
    att_q90    = lambda x: x.quantile(0.90),
    att_skew   = lambda x: skew(x.dropna())     if len(x.dropna()) > 2 else 0,
    att_kurt   = lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 2 else 0,
    att_count  = 'count',
).reset_index()

att_thresh = attendance.groupby('student_id')['attendance_pct'].agg(
    att_pct_below_75 = lambda x: (x < 0.75).mean(),
    att_pct_below_60 = lambda x: (x < 0.60).mean(),
    att_pct_below_50 = lambda x: (x < 0.50).mean(),
    att_pct_below_40 = lambda x: (x < 0.40).mean(),
    att_pct_perfect  = lambda x: (x >= 1.0).mean(),
    att_pct_above_90 = lambda x: (x >= 0.90).mean(),
).reset_index()

att_trend = attendance.groupby('student_id')['attendance_pct'].apply(
    lambda x: compute_trend(x.values)
).reset_index()
att_trend.columns = ['student_id', 'att_trend']

def max_streak(series, threshold=0.75):
    below = (series < threshold).astype(int).values
    max_s = cur_s = 0
    for v in below:
        if v:
            cur_s += 1
            max_s  = max(max_s, cur_s)
        else:
            cur_s  = 0
    return max_s

att_streak = attendance.groupby('student_id')['attendance_pct'].apply(
    lambda x: max_streak(x)
).reset_index()
att_streak.columns = ['student_id', 'att_max_streak']

def recent_vs_early(group):
    vals   = group['attendance_pct'].values
    early  = vals[:4].mean()  if len(vals) >= 4 else vals.mean()
    recent = vals[-4:].mean() if len(vals) >= 4 else vals.mean()
    return pd.Series({'att_recent': recent, 'att_early': early,
                      'att_recent_vs_early': recent - early})

att_rve = attendance.groupby('student_id').apply(recent_vs_early).reset_index()

att_sem = attendance.groupby(['student_id','semester'])['attendance_pct'].mean().reset_index()
att_sem.columns = ['student_id','semester','sem_att']

first_sem = att_sem.groupby('student_id').apply(
    lambda x: x.nsmallest(1,'semester')['sem_att'].values[0]).reset_index()
first_sem.columns = ['student_id','att_first_sem']

last_sem = att_sem.groupby('student_id').apply(
    lambda x: x.nlargest(1,'semester')['sem_att'].values[0]).reset_index()
last_sem.columns = ['student_id','att_last_sem']

sem_change = first_sem.merge(last_sem, on='student_id')
sem_change['att_sem_decline']    = sem_change['att_first_sem'] - sem_change['att_last_sem']
sem_change['att_is_declining']   = (sem_change['att_sem_decline'] > 0).astype(int)
sem_change['att_decline_severe'] = (sem_change['att_sem_decline'] > 0.15).astype(int)

att_sem_count = att_sem.groupby('student_id')['semester'].nunique().reset_index()
att_sem_count.columns = ['student_id','att_num_semesters']

att_subj = attendance.groupby(['student_id','subject'])['attendance_pct'].mean().unstack(fill_value=np.nan)
att_subj.columns = [f'att_subj_{c}' for c in att_subj.columns]
att_subj = att_subj.reset_index()
att_subj['att_worst_subject']  = att_subj.drop('student_id',axis=1).min(axis=1)
att_subj['att_best_subject']   = att_subj.drop('student_id',axis=1).max(axis=1)
att_subj['att_subject_spread'] = att_subj['att_best_subject'] - att_subj['att_worst_subject']

att_vol = attendance.groupby(['student_id','semester'])['attendance_pct'].std().reset_index()
att_vol.columns = ['student_id','semester','sem_vol']
att_vol_agg = att_vol.groupby('student_id')['sem_vol'].agg(
    att_vol_mean='mean', att_vol_max='max', att_vol_std='std').reset_index()

def rolling_drop(group):
    vals = group['attendance_pct'].values
    if len(vals) < 4:
        return pd.Series({'att_roll_drop': 0})
    rolled = pd.Series(vals).rolling(2).mean().dropna().values
    return pd.Series({'att_roll_drop': float(np.min(np.diff(rolled)))})

att_roll = attendance.groupby('student_id').apply(rolling_drop).reset_index()

# NEW: Mid-semester dip
def mid_dip(group):
    vals = group['attendance_pct'].values
    if len(vals) < 6:
        return pd.Series({'att_mid_dip': 0})
    mid = vals[len(vals)//4 : 3*len(vals)//4]
    return pd.Series({'att_mid_dip': vals.mean() - mid.mean()})

att_mid = attendance.groupby('student_id').apply(mid_dip).reset_index()

att_features = (att_global
    .merge(att_thresh,    on='student_id', how='left')
    .merge(att_trend,     on='student_id', how='left')
    .merge(att_streak,    on='student_id', how='left')
    .merge(att_rve,       on='student_id', how='left')
    .merge(sem_change[['student_id','att_first_sem','att_last_sem',
                        'att_sem_decline','att_is_declining','att_decline_severe']],
           on='student_id', how='left')
    .merge(att_sem_count, on='student_id', how='left')
    .merge(att_subj,      on='student_id', how='left')
    .merge(att_vol_agg,   on='student_id', how='left')
    .merge(att_roll,      on='student_id', how='left')
    .merge(att_mid,       on='student_id', how='left')
)
print(f"  ✔ Attendance features: {att_features.shape}")


# ╔══════════════════════════════════════════════════════════╗
# ║            STEP 3: NLP FEATURES                          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[3/9] Engineering NLP features...")

text_col = [c for c in counsellor.columns if c != 'student_id'][0]
counsellor[text_col] = counsellor[text_col].fillna('').astype(str)

counsellor_agg = counsellor.groupby('student_id')[text_col].apply(
    lambda x: ' '.join(x)).reset_index()
counsellor_agg.columns = ['student_id','notes']

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    s = sia.polarity_scores(str(text))
    return pd.Series({
        'sent_compound':    s['compound'],
        'sent_pos':         s['pos'],
        'sent_neg':         s['neg'],
        'sent_neu':         s['neu'],
        'sent_neg_pos_ratio': s['neg'] / (s['pos'] + 1e-6),
    })

sent_features = counsellor_agg['notes'].apply(get_sentiment)
counsellor_agg = pd.concat([counsellor_agg, sent_features], axis=1)

RISK_WORDS = [
    'fail','absent','concern','poor','weak','struggle','miss','dropout',
    'risk','problem','issue','low','depressed','anxious','stress',
    'financial','family','withdrawn','disengaged','unmotivated','irregular',
    'warning','critical','severe','intervention','repeated','chronic',
    'hostile','conflict','isolation','quit','leave','give up','behind',
    'failing','missed','skipped','skipping','not attending','performance'
]
POS_WORDS = [
    'improve','good','excellent','attend','progress','motivated','engaged',
    'active','consistent','regular','better','positive','dedicated',
    'hardworking','improvement','recovered','responsive','showing','effort'
]

def text_stats(text):
    text  = str(text)
    words = text.split()
    sents = re.split(r'[.!?]', text)
    tl    = text.lower()
    rc    = sum(1 for w in RISK_WORDS if w in tl)
    pc    = sum(1 for w in POS_WORDS  if w in tl)
    return pd.Series({
        'nlp_word_count':        len(words),
        'nlp_char_count':        len(text),
        'nlp_sent_count':        len([s for s in sents if s.strip()]),
        'nlp_risk_word_count':   rc,
        'nlp_pos_word_count':    pc,
        'nlp_risk_pos_ratio':    rc / (pc + 1e-6),
        'nlp_net_sentiment':     pc - rc,
        'nlp_avg_word_len':      np.mean([len(w) for w in words]) if words else 0,
        'nlp_unique_words':      len(set(words)),
        'nlp_lexical_diversity': len(set(words)) / (len(words) + 1e-6),
        'nlp_exclamation':       text.count('!'),
        'nlp_question':          text.count('?'),
        'nlp_uppercase_ratio':   sum(1 for c in text if c.isupper()) / (len(text) + 1e-6),
    })

stats_features = counsellor_agg['notes'].apply(text_stats)
counsellor_agg = pd.concat([counsellor_agg, stats_features], axis=1)

# TF-IDF + SVD (LSA)
stop_words   = list(stopwords.words('english'))
tfidf = TfidfVectorizer(
    max_features=500, stop_words=stop_words,
    ngram_range=(1, 3), min_df=2, sublinear_tf=True
)
tfidf_matrix  = tfidf.fit_transform(counsellor_agg['notes'])
svd           = TruncatedSVD(n_components=60, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)
tfidf_df      = pd.DataFrame(tfidf_reduced, columns=[f'tfidf_svd_{i}' for i in range(60)])
tfidf_df['student_id'] = counsellor_agg['student_id'].values

nlp_base_cols = ['student_id','sent_compound','sent_pos','sent_neg','sent_neu',
                 'sent_neg_pos_ratio','nlp_word_count','nlp_char_count',
                 'nlp_sent_count','nlp_risk_word_count','nlp_pos_word_count',
                 'nlp_risk_pos_ratio','nlp_net_sentiment','nlp_avg_word_len',
                 'nlp_unique_words','nlp_lexical_diversity',
                 'nlp_exclamation','nlp_question','nlp_uppercase_ratio']
nlp_features = counsellor_agg[nlp_base_cols].merge(tfidf_df, on='student_id', how='left')
print(f"  ✔ NLP features: {nlp_features.shape}")


# ╔══════════════════════════════════════════════════════════╗
# ║          STEP 4: TABULAR FEATURE ENGINEERING             ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[4/9] Engineering tabular features...")

def engineer_tabular(df):
    df       = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'student_id' in cat_cols: cat_cols.remove('student_id')
    if TARGET       in num_cols: num_cols.remove(TARGET)

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('Unknown')
        df[col] = le.fit_transform(df[col])

    grade_cols = [c for c in num_cols if any(k in c.lower() for k in
                  ['grade','gpa','score','mark','cgpa','sgpa','result'])]
    if grade_cols:
        df['grade_mean']          = df[grade_cols].mean(axis=1)
        df['grade_std']           = df[grade_cols].std(axis=1).fillna(0)
        df['grade_min']           = df[grade_cols].min(axis=1)
        df['grade_max']           = df[grade_cols].max(axis=1)
        df['grade_range']         = df['grade_max'] - df['grade_min']
        df['grade_median']        = df[grade_cols].median(axis=1)
        df['grade_trend']         = df[grade_cols].apply(
            lambda row: compute_trend(row.dropna().values), axis=1)
        df['grade_improving']     = (df['grade_trend'] > 0).astype(int)
        df['grade_failing_count'] = (df[grade_cols] < 40).sum(axis=1)
        df['grade_cv']            = df['grade_std'] / (df['grade_mean'] + 1e-6)

    att_tab = [c for c in num_cols if 'attend' in c.lower()]
    if att_tab:
        df['tab_att_mean'] = df[att_tab].mean(axis=1)
        df['tab_att_min']  = df[att_tab].min(axis=1)
        df['tab_att_std']  = df[att_tab].std(axis=1).fillna(0)

    if grade_cols and att_tab:
        df['grade_x_att']   = df['grade_mean'] * df['tab_att_mean']
        df['grade_div_att'] = df['grade_mean'] / (df['tab_att_mean'] + 1e-6)
        df['risk_score']    = (1 - df['tab_att_mean']) + (1 - df['grade_mean'] / 100)

    return df

train_eng = engineer_tabular(train.drop(columns=[TARGET]))
test_eng  = engineer_tabular(test)
print(f"  ✔ Tabular features: {train_eng.shape}")


# ╔══════════════════════════════════════════════════════════╗
# ║         STEP 5: MERGE ALL FEATURES                       ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[5/9] Merging all feature sets...")

def merge_all(df):
    df = df.merge(att_features, on='student_id', how='left')
    df = df.merge(nlp_features, on='student_id', how='left')
    return df

train_full = merge_all(train_eng)
test_full  = merge_all(test_eng)

train_full     = train_full.drop(columns=['student_id'])
test_ids_final = test_full['student_id'].copy()
test_full      = test_full.drop(columns=['student_id'])

train_full, test_full = train_full.align(test_full, join='left', axis=1, fill_value=0)

imputer    = SimpleImputer(strategy='median')
train_full = pd.DataFrame(imputer.fit_transform(train_full), columns=train_full.columns)
test_full  = pd.DataFrame(imputer.transform(test_full),      columns=test_full.columns)

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(train_full)
T_scaled = scaler.transform(test_full)

print(f"  ✔ Final train : {train_full.shape}")
print(f"  ✔ Final test  : {test_full.shape}")

X = train_full.values
y = target.values


# ╔══════════════════════════════════════════════════════════╗
# ║         STEP 6: OPTUNA HYPERPARAMETER TUNING             ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[6/9] Optuna Hyperparameter Tuning...")

best_lgbm_params = None
best_xgb_params  = None

if OPTUNA_AVAILABLE:
    tune_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Tune LightGBM ─────────────────────────────────────
    def lgbm_objective(trial):
        params = {
            'objective':         'multiclass',
            'num_class':         3,
            'metric':            'multi_logloss',
            'verbosity':         -1,
            'n_jobs':            -1,
            'random_state':      42,
            'class_weight':      'balanced',
            'n_estimators':      trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth':         trial.suggest_int('max_depth', 4, 10),
            'num_leaves':        trial.suggest_int('num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
        }
        scores = []
        for tr_idx, val_idx in tune_skf.split(X, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(-1)])
            p = m.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], p, average='macro'))
        return np.mean(scores)

    print("  Tuning LightGBM (40 trials)...")
    lgbm_study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
    lgbm_study.optimize(lgbm_objective, n_trials=40, show_progress_bar=False)
    best_lgbm_params = lgbm_study.best_params
    print(f"  ✔ Best LGBM F1: {lgbm_study.best_value:.4f}")
    print(f"    Params: {best_lgbm_params}")

    # ── Tune XGBoost ──────────────────────────────────────
    def xgb_objective(trial):
        params = {
            'objective':        'multi:softprob',
            'num_class':        3,
            'eval_metric':      'mlogloss',
            'use_label_encoder': False,
            'verbosity':        0,
            'n_jobs':           -1,
            'random_state':     42,
            'n_estimators':     trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 9),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        }
        scores = []
        for tr_idx, val_idx in tune_skf.split(X, y):
            m = xgb.XGBClassifier(**params)
            m.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[val_idx], y[val_idx])],
                  verbose=False)
            p = m.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], p, average='macro'))
        return np.mean(scores)

    print("  Tuning XGBoost (30 trials)...")
    xgb_study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=False)
    best_xgb_params = xgb_study.best_params
    print(f"  ✔ Best XGB  F1: {xgb_study.best_value:.4f}")
    print(f"    Params: {best_xgb_params}")
else:
    print("  ⚠ Skipping Optuna — using default params")


# ╔══════════════════════════════════════════════════════════╗
# ║         STEP 7: BUILD & TRAIN ALL MODELS                 ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[7/9] Training full ensemble (10-fold CV)...")

N_FOLDS = 10
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# ── LightGBM 1 (Optuna-tuned or default) ─────────────────
if best_lgbm_params:
    lgbm1 = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, metric='multi_logloss',
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        **best_lgbm_params
    )
else:
    lgbm1 = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, metric='multi_logloss',
        n_estimators=2000, learning_rate=0.02, max_depth=8,
        num_leaves=127, min_child_samples=15,
        subsample=0.75, subsample_freq=1, colsample_bytree=0.75,
        reg_alpha=0.2, reg_lambda=1.5,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
    )

# ── LightGBM 2 (different config for diversity) ───────────
lgbm2 = lgb.LGBMClassifier(
    objective='multiclass', num_class=3, metric='multi_logloss',
    n_estimators=1500, learning_rate=0.03, max_depth=6,
    num_leaves=63, min_child_samples=30,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=2.0,
    class_weight='balanced', random_state=123, n_jobs=-1, verbose=-1
)

# ── XGBoost 1 (Optuna-tuned or default) ──────────────────
if best_xgb_params:
    xgb1 = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, eval_metric='mlogloss',
        use_label_encoder=False, random_state=42, n_jobs=-1, verbosity=0,
        **best_xgb_params
    )
else:
    xgb1 = xgb.XGBClassifier(
        objective='multi:softprob', num_class=3, eval_metric='mlogloss',
        n_estimators=1500, learning_rate=0.03, max_depth=7,
        subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.2, reg_lambda=2.0, min_child_weight=5,
        use_label_encoder=False, random_state=42, n_jobs=-1, verbosity=0
    )

# ── XGBoost 2 (shallow, high regularization) ─────────────
xgb2 = xgb.XGBClassifier(
    objective='multi:softprob', num_class=3, eval_metric='mlogloss',
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=3.0, min_child_weight=10,
    use_label_encoder=False, random_state=99, n_jobs=-1, verbosity=0
)

# ── Extra Trees ───────────────────────────────────────────
et1 = ExtraTreesClassifier(
    n_estimators=1000, max_depth=20, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1
)

# ── Random Forest ─────────────────────────────────────────
rf1 = RandomForestClassifier(
    n_estimators=800, max_depth=18, min_samples_leaf=5,
    class_weight='balanced_subsample', random_state=42, n_jobs=-1
)

# ── CatBoost ──────────────────────────────────────────────
BASE_MODELS = [
    ('lgbm1', lgbm1),
    ('lgbm2', lgbm2),
    ('xgb1',  xgb1),
    ('xgb2',  xgb2),
    ('et1',   et1),
    ('rf1',   rf1),
]

if CATBOOST_AVAILABLE:
    cat1 = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=7,
        loss_function='MultiClass', eval_metric='TotalF1',
        class_weights=list(class_weights.values()),
        random_seed=42, verbose=0, thread_count=-1,
        early_stopping_rounds=50,
        l2_leaf_reg=3.0, bagging_temperature=0.5
    )
    BASE_MODELS.append(('cat1', cat1))
    print("  ✔ CatBoost added to ensemble")

N_MODELS = len(BASE_MODELS)
print(f"  Total base models: {N_MODELS}")

# ── OOF Arrays ───────────────────────────────────────────
oof_preds  = {name: np.zeros((len(X), 3)) for name,_ in BASE_MODELS}
test_preds = {name: np.zeros((len(test_full), 3)) for name,_ in BASE_MODELS}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr,  X_val  = X[tr_idx],        X[val_idx]
    y_tr,  y_val  = y[tr_idx],        y[val_idx]

    fold_results = []

    for name, model in BASE_MODELS:
        if 'lgbm' in name:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(-1)])
        elif 'xgb' in name:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        elif 'cat' in name:
            model.fit(X_tr, y_tr,
                      eval_set=(X_val, y_val),
                      verbose=False)
        else:
            model.fit(X_tr, y_tr)

        oof_preds[name][val_idx]  = model.predict_proba(X_val)
        test_preds[name]         += model.predict_proba(test_full.values) / N_FOLDS

        fold_results.append(
            f1_score(y_val,
                     np.argmax(oof_preds[name][val_idx], axis=1),
                     average='macro'))

    # Equal-weight blend for reporting
    blend_val = np.mean([oof_preds[n][val_idx] for n,_ in BASE_MODELS], axis=0)
    fold_f1   = f1_score(y_val, np.argmax(blend_val, axis=1), average='macro')
    model_str = " | ".join([f"{n}: {s:.4f}" for (n,_), s in zip(BASE_MODELS, fold_results)])
    print(f"  Fold {fold+1:2d} | Blend: {fold_f1:.4f} | {model_str}")

print("\n  Base model OOF Macro F1:")
for name,_ in BASE_MODELS:
    f1 = f1_score(y, np.argmax(oof_preds[name], axis=1), average='macro')
    print(f"    {name:10s}: {f1:.4f}")


# ╔══════════════════════════════════════════════════════════╗
# ║         LEVEL 2: META-LEARNER STACKING                   ║
# ╚══════════════════════════════════════════════════════════╝
print("\n  Training Level-2 Meta-Learner...")

meta_train   = np.hstack([oof_preds[n]  for n,_ in BASE_MODELS])
meta_test    = np.hstack([test_preds[n] for n,_ in BASE_MODELS])

meta_scaler  = RobustScaler()
meta_train_s = meta_scaler.fit_transform(meta_train)
meta_test_s  = meta_scaler.transform(meta_test)

meta_lgbm = lgb.LGBMClassifier(
    objective='multiclass', num_class=3, metric='multi_logloss',
    n_estimators=500, learning_rate=0.02, max_depth=4,
    num_leaves=15, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=2.0,
    class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
)
meta_lr = LogisticRegression(
    C=1.0, class_weight='balanced',
    max_iter=1000, random_state=42
)

meta_skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta_lgbm  = np.zeros((len(X), 3))
oof_meta_lr    = np.zeros((len(X), 3))
test_meta_lgbm = np.zeros((len(test_full), 3))
test_meta_lr   = np.zeros((len(test_full), 3))

for fold, (tr_idx, val_idx) in enumerate(meta_skf.split(meta_train_s, y)):
    meta_lgbm.fit(meta_train_s[tr_idx], y[tr_idx])
    oof_meta_lgbm[val_idx]  = meta_lgbm.predict_proba(meta_train_s[val_idx])
    test_meta_lgbm         += meta_lgbm.predict_proba(meta_test_s) / 5

    meta_lr.fit(meta_train_s[tr_idx], y[tr_idx])
    oof_meta_lr[val_idx]  = meta_lr.predict_proba(meta_train_s[val_idx])
    test_meta_lr         += meta_lr.predict_proba(meta_test_s) / 5

print(f"  Meta-LGBM F1 : {f1_score(y, np.argmax(oof_meta_lgbm, axis=1), average='macro'):.4f}")
print(f"  Meta-LR   F1 : {f1_score(y, np.argmax(oof_meta_lr,   axis=1), average='macro'):.4f}")

direct_blend_oof  = np.mean([oof_preds[n]  for n,_ in BASE_MODELS], axis=0)
direct_blend_test = np.mean([test_preds[n] for n,_ in BASE_MODELS], axis=0)

final_oof_proba  = 0.4*direct_blend_oof  + 0.35*oof_meta_lgbm  + 0.25*oof_meta_lr
final_test_proba = 0.4*direct_blend_test + 0.35*test_meta_lgbm + 0.25*test_meta_lr

print(f"\n  Stacked OOF F1: {f1_score(y, np.argmax(final_oof_proba, axis=1), average='macro'):.4f}")


# ╔══════════════════════════════════════════════════════════╗
# ║         STEP 8: PSEUDO LABELING                          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[8/9] Pseudo Labeling...")

# Use high-confidence test predictions as extra training data
CONFIDENCE_THRESHOLD = 0.85  # only very confident predictions

max_proba     = final_test_proba.max(axis=1)
pseudo_mask   = max_proba >= CONFIDENCE_THRESHOLD
pseudo_X      = test_full.values[pseudo_mask]
pseudo_y      = np.argmax(final_test_proba[pseudo_mask], axis=1)

print(f"  High-confidence pseudo labels: {pseudo_mask.sum()} / {len(test_full)}")
print(f"  Distribution: {np.bincount(pseudo_y)}")

if pseudo_mask.sum() > 100:
    # Augment training data
    X_aug = np.vstack([X, pseudo_X])
    y_aug = np.concatenate([y, pseudo_y])

    # Retrain best model (lgbm1) on augmented data
    pseudo_lgbm = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, metric='multi_logloss',
        n_estimators=2000, learning_rate=0.02, max_depth=8,
        num_leaves=127, min_child_samples=15,
        subsample=0.75, subsample_freq=1, colsample_bytree=0.75,
        reg_alpha=0.2, reg_lambda=1.5,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
    )

    # OOF for pseudo model (only on original training data)
    oof_pseudo = np.zeros((len(X), 3))
    test_pseudo = np.zeros((len(test_full), 3))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        # Train on augmented (original + pseudo), validate on original only
        pseudo_in_fold = np.arange(len(X), len(X_aug))
        aug_tr_idx     = np.concatenate([tr_idx, pseudo_in_fold])

        pseudo_lgbm.fit(X_aug[aug_tr_idx], y_aug[aug_tr_idx],
                        eval_set=[(X[val_idx], y[val_idx])],
                        callbacks=[lgb.early_stopping(100, verbose=False),
                                   lgb.log_evaluation(-1)])
        oof_pseudo[val_idx]  = pseudo_lgbm.predict_proba(X[val_idx])
        test_pseudo         += pseudo_lgbm.predict_proba(test_full.values) / N_FOLDS

    pseudo_f1 = f1_score(y, np.argmax(oof_pseudo, axis=1), average='macro')
    print(f"  ✔ Pseudo-label model OOF F1: {pseudo_f1:.4f}")

    # Blend pseudo model into final predictions
    final_oof_proba  = 0.7 * final_oof_proba  + 0.3 * oof_pseudo
    final_test_proba = 0.7 * final_test_proba + 0.3 * test_pseudo
    print(f"  ✔ After pseudo blend OOF F1: {f1_score(y, np.argmax(final_oof_proba, axis=1), average='macro'):.4f}")
else:
    print("  ⚠ Not enough confident predictions for pseudo labeling")


# ╔══════════════════════════════════════════════════════════╗
# ║         OPTIMAL THRESHOLD TUNING                         ║
# ╚══════════════════════════════════════════════════════════╝
print("\n  Tuning decision thresholds...")

best_f1         = 0
best_thresholds = [1/3, 1/3, 1/3]

# Search over threshold boosts for Medium and High Risk
for t1 in np.arange(0.90, 1.15, 0.05):   # Medium Risk multiplier
    for t2 in np.arange(0.90, 1.15, 0.05):  # High Risk multiplier
        adj = final_oof_proba.copy()
        adj[:, 1] *= t1
        adj[:, 2] *= t2
        adj = adj / adj.sum(axis=1, keepdims=True)
        f1 = f1_score(y, np.argmax(adj, axis=1), average='macro')
        if f1 > best_f1:
            best_f1         = f1
            best_thresholds = [t1, t2]

print(f"  ✔ Best threshold multipliers — Medium: {best_thresholds[0]:.2f}, High: {best_thresholds[1]:.2f}")
print(f"  ✔ Threshold-tuned OOF F1: {best_f1:.4f}")

# Apply best thresholds to test
final_test_proba_adj = final_test_proba.copy()
final_test_proba_adj[:, 1] *= best_thresholds[0]
final_test_proba_adj[:, 2] *= best_thresholds[1]
final_test_proba_adj = final_test_proba_adj / final_test_proba_adj.sum(axis=1, keepdims=True)

print(f"\n  Final Classification Report (OOF):")
print(classification_report(y, np.argmax(final_oof_proba, axis=1),
      target_names=['Low Risk','Medium Risk','High Risk']))


# ╔══════════════════════════════════════════════════════════╗
# ║         STEP 9: SAVE SUBMISSION                          ║
# ╚══════════════════════════════════════════════════════════╝
print("\n[9/9] Saving submission...")

final_preds = np.argmax(final_test_proba_adj, axis=1)
submission  = pd.DataFrame({
    'student_id':   test_ids_final.values,
    'dropout_risk': final_preds
})
submission.to_csv(OUTPUT_PATH, index=False)

elapsed = (time.time() - start_time) / 60
print(f"  ✔ Saved to : {OUTPUT_PATH}")
print(f"  ✔ Total    : {len(submission)}")
print(f"\n  Distribution:")
print(submission['dropout_risk'].value_counts().sort_index().rename(
    {0:'0-Low', 1:'1-Medium', 2:'2-High'}))

print("\n" + "=" * 60)
print(f"  DONE!  Final OOF Macro F1 : {best_f1:.4f}")
print(f"  Time elapsed              : {elapsed:.1f} min")
print(f"  Submit: {OUTPUT_PATH}")
print("=" * 60)