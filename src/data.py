import pandas as pd
from sklearn.model_selection import train_test_split
from .feature_engineering import prepare_features

def load_data(sex_filter=None, age_filter=None):
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    # Load from local CSV file instead of URL
    df = pd.read_csv('heart.csv', names=columns, na_values=['?'])
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Convert target to binary (0 or 1)
    df['target'] = df['target'].map(lambda x: 1 if x > 0 else 0)
    
    # Filter by sex if specified
    if sex_filter is not None:
        df = df[df['sex'] == sex_filter]
    
    # Filter by age bracket if specified
    if age_filter is not None:
        age_brackets = {
            'under_30': (0, 30),
            '30s': (30, 40),
            '40s': (40, 50),
            '50s': (50, 60),
            '60s': (60, 70),
            '70s': (70, 80),
            '80_and_above': (80, 120)
        }
        if age_filter in age_brackets:
            min_age, max_age = age_brackets[age_filter]
            df = df[(df['age'] >= min_age) & (df['age'] < max_age)]
    
    return df

def prepare_data(df, test_size=0.3):
    # Get engineered features
    X_scaled, y, feature_names = prepare_features(df)
    
    # Split data
    return train_test_split(X_scaled, y, test_size=test_size, random_state=42), feature_names 