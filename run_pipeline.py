import os
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def eval_metric(y_true, y_pred):
    return max(0, 100 * r2_score(y_true, y_pred))

def preprocess_datasets(train_df, test_df):
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # 1. Spatiotemporal Features
    for df in [train_copy, test_copy]:
        df['hour'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]) if pd.notnull(x) and ':' in str(x) else -1)
        df['minute'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[1]) if pd.notnull(x) and ':' in str(x) else -1)
        df.drop(['timestamp'], axis=1, inplace=True, errors='ignore')
    
    # 2. Temperature Imputation
    temp_median = train_copy['Temperature'].median()
    train_copy['Temperature'] = train_copy['Temperature'].fillna(temp_median)
    test_copy['Temperature'] = test_copy['Temperature'].fillna(temp_median)
    
    # 3. Categorical Alignment (Unified CategoricalDtype for Train & Test)
    cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    category_map = {}
    
    for col in cat_cols:
        train_copy[col] = train_copy[col].fillna('Unknown').astype(str)
        test_copy[col] = test_copy[col].fillna('Unknown').astype(str)
        
        all_categories = sorted(list(set(train_copy[col]).union(set(test_copy[col]))))
        cat_type = pd.CategoricalDtype(categories=all_categories)
        
        train_copy[col] = train_copy[col].astype(cat_type)
        test_copy[col] = test_copy[col].astype(cat_type)
        category_map[col] = all_categories
        
    for df in [train_copy, test_copy]:
        if 'NumberofLanes' in df.columns:
            df['NumberofLanes'] = df['NumberofLanes'].fillna(-1)
            
    return train_copy, test_copy, temp_median, category_map

def main():
    print("="*60)
    print(" [TRAFFIC DEMAND PREDICTION ML PIPELINE EXECUTOR]")
    print("="*60)
    
    start_time = time.time()
    
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Dataset files missing. Looked for {train_path} and {test_path}")
        return
        
    print("[1/5] Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f" -> Train dataset shape: {train.shape}")
    print(f" -> Test dataset shape:  {test.shape}")
    
    print("\n[2/5] Preprocessing and aligning categorical feature spaces...")
    train_clean, test_clean, temp_median, category_map = preprocess_datasets(train, test)
    
    X = train_clean.drop(['Index', 'demand'], axis=1)
    y = train_clean['demand']
    X_test = test_clean.drop(['Index'], axis=1)
    
    cat_features = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    
    lgb_params = {'learning_rate': 0.05, 'num_leaves': 60, 'max_depth': 8, 'n_estimators': 500, 'random_state': 42, 'n_jobs': -1}
    cat_params = {'learning_rate': 0.05, 'depth': 6, 'iterations': 500, 'random_seed': 42, 'verbose': False, 'cat_features': cat_features, 'thread_count': -1}
    xgb_params = {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 500, 'random_state': 42, 'enable_categorical': True, 'tree_method': 'hist', 'n_jobs': -1}
    
    print("\n[3/5] Performing 5-Fold Out-of-Fold (OOF) Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_lgb = np.zeros(len(X))
    oof_preds_cat = np.zeros(len(X))
    oof_preds_xgb = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f" -> Processing Fold {fold}/5...")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        m_lgb = LGBMRegressor(**lgb_params)
        m_lgb.fit(X_tr, y_tr, categorical_feature=cat_features)
        oof_preds_lgb[val_idx] = m_lgb.predict(X_va)
        
        m_cat = CatBoostRegressor(**cat_params)
        m_cat.fit(X_tr, y_tr)
        oof_preds_cat[val_idx] = m_cat.predict(X_va)
        
        m_xgb = XGBRegressor(**xgb_params)
        m_xgb.fit(X_tr, y_tr)
        oof_preds_xgb[val_idx] = m_xgb.predict(X_va)
        
    oof_preds_blend = (oof_preds_lgb + oof_preds_cat + oof_preds_xgb) / 3.0
    
    oof_r2_custom = eval_metric(y, oof_preds_blend)
    oof_r2_raw = r2_score(y, oof_preds_blend)
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds_blend))
    oof_mae = mean_absolute_error(y, oof_preds_blend)
    
    print("\n" + "="*60)
    print(" [5-FOLD OUT-OF-FOLD (OOF) CV EVALUATION METRICS]")
    print("="*60)
    print(f" -> OOF Custom R2 Score (%): {oof_r2_custom:.4f}%")
    print(f" -> OOF Standard R2 Score:  {oof_r2_raw:.4f}")
    print(f" -> OOF RMSE:                {oof_rmse:.4f}")
    print(f" -> OOF MAE:                 {oof_mae:.4f}")
    print("="*60)
    
    print("\n[4/5] Training final ensemble on full dataset...")
    print(" -> Training LightGBM...")
    full_lgb = LGBMRegressor(**lgb_params)
    full_lgb.fit(X, y, categorical_feature=cat_features)
    preds_lgb = full_lgb.predict(X_test)
    
    print(" -> Training CatBoost...")
    full_cat = CatBoostRegressor(**cat_params)
    full_cat.fit(X, y)
    preds_cat = full_cat.predict(X_test)
    
    print(" -> Training XGBoost...")
    full_xgb = XGBRegressor(**xgb_params)
    full_xgb.fit(X, y)
    preds_xgb = full_xgb.predict(X_test)
    
    preds_blend = (preds_lgb + preds_cat + preds_xgb) / 3.0
    
    print("\n[5/5] Saving test predictions & model artifacts...")
    sub = pd.DataFrame({
        'Index': test['Index'],
        'demand': preds_blend
    })
    
    sub.to_csv('test_predictions.csv', index=False)
    print(" -> Successfully saved 'test_predictions.csv'")
    
    os.makedirs('submission', exist_ok=True)
    sub.to_csv('submission/submission.csv', index=False)
    print(" -> Successfully saved 'submission/submission.csv'")
    
    os.makedirs('models', exist_ok=True)
    artifacts = {
        'model_lgb': full_lgb,
        'model_cat': full_cat,
        'model_xgb': full_xgb,
        'temp_median': temp_median,
        'cat_features': cat_features,
        'category_map': category_map,
        'feature_names': list(X.columns),
        'metrics': {
            'oof_r2': oof_r2_custom,
            'oof_rmse': oof_rmse,
            'oof_mae': oof_mae
        }
    }
    
    with open('models/model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print(" -> Successfully saved trained models to 'models/model_artifacts.pkl'")
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] ML Pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()
