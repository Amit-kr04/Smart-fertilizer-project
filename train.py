# train_user_models.py
# This script trains all models and saves them to the 'models_user' folder.

import os, joblib, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ====== CONFIG ======
NUTRIENT_PATH = "nutrient_release_data.csv"
FERT_PATH = "fertilizer_recommendation_data.csv"
MODEL_DIR = "models_user"
os.makedirs(MODEL_DIR, exist_ok=True)

# If FAST_MODE True => use smaller sample sizes and fewer estimators (faster)
FAST_MODE = False
SAMPLE_NR = 5000 if FAST_MODE else None
SAMPLE_FR = 10000 if FAST_MODE else None
N_EST_CLASS = 50 if FAST_MODE else 200
N_EST_REG = 50 if FAST_MODE else 200
# ====================

print("Loading datasets...")
try:
    df_nr = pd.read_csv(NUTRIENT_PATH)
    df_fr = pd.read_csv(FERT_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure 'nutrient_release_data.csv' and 'fertilizer_recommendation_data.csv' are in the same directory.")
    exit()

print("NR shape:", df_nr.shape, "FR shape:", df_fr.shape)

# Helper to find column by likely names
def find_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

# ---------- Nutrient release training ----------
nr_cols = list(df_nr.columns)
nr_feat_candidates = ['temperature','temp','Temparature','soil_moisture','Moisture','rainfall','Rainfall','coating','coating_thickness','time_days','soil_pH','pH']
nr_features = [c for c in nr_cols if c.lower() in [x.lower() for x in nr_feat_candidates]]
if not nr_features:
    nr_features = df_nr.select_dtypes(include=[np.number]).columns.tolist()

target_rate = find_col(nr_cols, ['nutrient_release_rate','release_rate','release_pct','release'])
target_cat = find_col(nr_cols, ['release_category','release_type','category'])

print("NR features:", nr_features)
print("NR targets:", target_rate, target_cat)

if SAMPLE_NR is not None:
    df_nr_s = df_nr.sample(n=min(SAMPLE_NR, len(df_nr)), random_state=42).reset_index(drop=True)
else:
    df_nr_s = df_nr.copy()

# Regression (rate)
if target_rate and len(nr_features)>0:
    X = df_nr_s[nr_features].dropna()
    y = df_nr_s.loc[X.index, target_rate]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression(); dt = DecisionTreeRegressor(max_depth=8, random_state=42)
    lr.fit(X_train, y_train); dt.fit(X_train, y_train)
    
    # FIX: Use np.sqrt(mean_squared_error(...)) for older scikit-learn versions
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr.predict(X_test)))
    rmse_dt = np.sqrt(mean_squared_error(y_test, dt.predict(X_test)))
    
    best_nr = lr if rmse_lr < rmse_dt else dt
    joblib.dump(best_nr, os.path.join(MODEL_DIR, "nutrient_release_model.joblib"))
    print(f"Nutrient release regression RMSE: LR={rmse_lr:.3f}, DT={rmse_dt:.3f}. Selected: {'LR' if best_nr==lr else 'DT'}")

# Classification (category)
if target_cat and len(nr_features)>0:
    Xc = df_nr_s[nr_features].dropna()
    yc = df_nr_s.loc[Xc.index, target_cat].astype(str)
    le_nr_cat = LabelEncoder(); yc_enc = le_nr_cat.fit_transform(yc)
    joblib.dump(le_nr_cat, os.path.join(MODEL_DIR, "le_nr_category.joblib"))

    # FIX: Check class counts before stratifying
    class_counts_nr = np.bincount(yc_enc)
    if np.min(class_counts_nr) < 2:
        print(f"Warning: Not stratifying NR category. Smallest class has {np.min(class_counts_nr)} sample(s).")
        stratify_nr = None
    else:
        stratify_nr = yc_enc

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc_enc, test_size=0.2, random_state=42, stratify=stratify_nr)
    
    clf_nr = RandomForestClassifier(n_estimators=N_EST_CLASS, random_state=42, n_jobs=-1)
    clf_nr.fit(Xc_train, yc_train)
    acc_nr = accuracy_score(yc_test, clf_nr.predict(Xc_test))
    joblib.dump(clf_nr, os.path.join(MODEL_DIR, "nutrient_release_category_clf.joblib"))
    print(f"Nutrient release category classifier accuracy: {acc_nr:.3f}")

# ---------- Fertilizer recommendation training ----------
fr_cols = list(df_fr.columns)
fert_type = find_col(fr_cols, ['fertilizer_type','fertilizer','Fertilizer','fertilizer_name','Fertilizer Name'])
fert_amount = find_col(fr_cols, ['fertilizer_amount','amount','application_amount','fertilizer_application','fertilizer_amount_kg_ha'])
fert_timing = find_col(fr_cols, ['fertilizer_timing','timing','application_timing','days_after_sowing','fertilizer_timing'])
print("Detected FR targets:", fert_type, fert_amount, fert_timing)

crop_col = find_col(fr_cols, ['crop_type','Crop Type','crop'])
soilph_col = find_col(fr_cols, ['soil_pH','soil pH','pH'])
n_col = find_col(fr_cols, ['N','Nitrogen'])
p_col = find_col(fr_cols, ['P','Phosphorous','Phosphorus'])
k_col = find_col(fr_cols, ['K','Potassium'])
nr_col_fr = find_col(fr_cols, ['nutrient_release_rate','release_rate','release_pct'])

feature_list = [c for c in [crop_col, soilph_col, n_col, p_col, k_col, nr_col_fr] if c is not None]
print("FR features used:", feature_list)

if SAMPLE_FR is not None:
    df_fr_s = df_fr.sample(n=min(SAMPLE_FR, len(df_fr)), random_state=42).reset_index(drop=True)
else:
    df_fr_s = df_fr.copy()

if fert_type and feature_list:
    fr = df_fr_s[feature_list + [fert_type]].dropna().copy()
    le_crop = LabelEncoder(); fr['crop_enc'] = le_crop.fit_transform(fr[crop_col]); joblib.dump(le_crop, os.path.join(MODEL_DIR,"le_crop.joblib"))
    le_fert = LabelEncoder(); y_f_enc = le_fert.fit_transform(fr[fert_type]); joblib.dump(le_fert, os.path.join(MODEL_DIR,"le_fert.joblib"))
    Xf = fr[[c for c in ['crop_enc', soilph_col, n_col, p_col, k_col, nr_col_fr] if c in fr.columns]].copy()
    
    # FIX: Check class counts before stratifying
    class_counts_f = np.bincount(y_f_enc)
    if np.min(class_counts_f) < 2:
        print(f"Warning: Not stratifying fertilizer type. Smallest class has {np.min(class_counts_f)} sample(s).")
        stratify_f = None
    else:
        stratify_f = y_f_enc

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(Xf, y_f_enc, test_size=0.2, random_state=42, stratify=stratify_f)
    
    clf_f = RandomForestClassifier(n_estimators=N_EST_CLASS, random_state=42, n_jobs=-1)
    clf_f.fit(X_train_f, y_train_f)
    acc_f = accuracy_score(y_test_f, clf_f.predict(X_test_f))
    joblib.dump(clf_f, os.path.join(MODEL_DIR,"fert_type_clf.joblib"))
    print(f"Fertilizer type classifier accuracy: {acc_f:.3f}")

if fert_amount and feature_list:
    fra = df_fr_s[feature_list + [fert_amount]].dropna().copy()
    fra['crop_enc'] = LabelEncoder().fit_transform(fra[crop_col])
    X_amt = fra[[c for c in ['crop_enc', soilph_col, n_col, p_col, k_col, nr_col_fr] if c in fra.columns]]
    y_amt = fra[fert_amount]
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_amt, y_amt, test_size=0.2, random_state=42)
    reg_a = RandomForestRegressor(n_estimators=N_EST_REG, random_state=42, n_jobs=-1)
    reg_a.fit(X_train_a, y_train_a)
    
    rmse_a = np.sqrt(mean_squared_error(y_test_a, reg_a.predict(X_test_a)))
    
    joblib.dump(reg_a, os.path.join(MODEL_DIR,"fert_amount_reg.joblib"))
    print(f"Fertilizer amount regressor RMSE: {rmse_a:.3f}")

if fert_timing and feature_list:
    frt = df_fr_s[feature_list + [fert_timing]].dropna().copy()
    frt['crop_enc'] = LabelEncoder().fit_transform(frt[crop_col])
    X_time = frt[[c for c in ['crop_enc', soilph_col, n_col, p_col, k_col, nr_col_fr] if c in frt.columns]]
    y_time = frt[fert_timing]
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_time, y_time, test_size=0.2, random_state=42)
    reg_t = RandomForestRegressor(n_estimators=N_EST_REG, random_state=42, n_jobs=-1)
    reg_t.fit(X_train_t, y_train_t)
    
    rmse_t = np.sqrt(mean_squared_error(y_test_t, reg_t.predict(X_test_t)))
    
    joblib.dump(reg_t, os.path.join(MODEL_DIR,"fert_timing_reg.joblib"))
    print(f"Fertilizer timing regressor RMSE: {rmse_t:.3f}")

print("Training complete. Models saved in:", MODEL_DIR)
