import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load heart disease data"""
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv('heart.csv', names=columns, na_values=['?'])
    df = df.dropna()
    df['target'] = df['target'].map(lambda x: 1 if x > 0 else 0)
    return df

def prepare_data(df, test_size=0.3):
    """Prepare and split data"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=test_size, random_state=42), X.columns 