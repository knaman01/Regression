import pandas as pd
from sklearn.model_selection import train_test_split
from .feature_engineering import prepare_features

def load_data():
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # Load from local CSV file instead of URL
    df = pd.read_csv('heart.csv', names=columns, na_values=['?'])
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert target to binary (0 or 1)
    df['target'] = df['target'].map(lambda x: 1 if x > 0 else 0)
    return df

def prepare_data(df, test_size=0.3):
    # Get engineered features
    X_scaled, y, feature_names = prepare_features(df)
    
    # Split data
    return train_test_split(X_scaled, y, test_size=test_size, random_state=42), feature_names 