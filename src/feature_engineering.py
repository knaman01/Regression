import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineeringConfig:
    """Configuration for feature engineering"""
    ENABLE = True  # Master switch for all feature engineering
    
    # Individual feature flags
    AGE_FEATURES = True      # Age-related features
    BP_CATEGORIES = True     # Blood pressure categories
    CHOL_FEATURES = True     # Cholesterol-related features
    INTERACTION_TERMS = True # Feature interactions

def engineer_features(df, config=FeatureEngineeringConfig):
    """Apply basic feature engineering to the heart disease dataset"""
    
    # If feature engineering is disabled, return original data
    if not config.ENABLE:
        print("\nFeature engineering disabled. Using original features...")
        return df.copy()
    
    print("\nApplying feature engineering...")
    df_engineered = df.copy()
    
    try:
        # 1. Age-related features
        if config.AGE_FEATURES:
            df_engineered['age_squared'] = df_engineered['age'] ** 2
            df_engineered['is_elderly'] = (df_engineered['age'] >= 60).astype(int)
        
        # 2. Blood pressure categories
        if config.BP_CATEGORIES:
            df_engineered['bp_category'] = pd.cut(
                df_engineered['trestbps'],
                bins=[0, 120, 140, 180, 300],
                labels=['normal', 'prehypertension', 'hypertension', 'severe']
            ).astype(str)
            df_engineered = pd.get_dummies(df_engineered, columns=['bp_category'], prefix='bp')
            df_engineered = df_engineered.drop('trestbps', axis=1)
        
        # 3. Cholesterol features
        if config.CHOL_FEATURES:
            df_engineered['chol_category'] = pd.cut(
                df_engineered['chol'],
                bins=[0, 200, 240, 1000],
                labels=['normal', 'borderline', 'high']
            ).astype(str)
            df_engineered = pd.get_dummies(df_engineered, columns=['chol_category'], prefix='chol')
            df_engineered = df_engineered.drop('chol', axis=1)
        
        # 4. Interaction features
        if config.INTERACTION_TERMS:
            if 'chol' in df_engineered.columns:
                df_engineered['age_chol'] = df_engineered['age'] * df_engineered['chol']
            df_engineered['age_thalach'] = df_engineered['age'] * df_engineered['thalach']
        
        print(f"Feature engineering completed. Features: {len(df.columns)} → {len(df_engineered.columns)}")
        
    except Exception as e:
        print(f"Error during feature engineering: {str(e)}")
        print("Falling back to original features.")
        return df.copy()
    
    return df_engineered

def prepare_features(df, test_size=0.3, config=FeatureEngineeringConfig):
    """Prepare features for modeling"""
    
    # Apply feature engineering
    df_processed = engineer_features(df, config)
    
    # Separate features and target
    X = df_processed.drop('target', axis=1)
    y = df_processed['target']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns 