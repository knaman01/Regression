import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

def load_data(dataset='cleveland'):
    """Load heart disease data
    
    Args:
        dataset: 'cleveland' for local heart.csv, 'openml' for OpenML dataset
    """
    try:
        if dataset == 'cleveland':
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df = pd.read_csv('heart.csv', names=columns, na_values=['?'])
            print("\nLoaded Cleveland dataset")
            
        elif dataset == 'openml':
            print("\nLoading OpenML heart dataset...")
            data = fetch_openml(name='heart', version=1, as_frame=False)
            df = pd.DataFrame(data.data.toarray(), columns=data.feature_names)
            df['target'] = data.target
            print("Loaded OpenML dataset")
        
        # Clean data
        df = df.dropna()
        df['target'] = df['target'].map(lambda x: 1 if float(x) > 0 else 0)
        print(f"Total samples: {len(df)}")
        return df
        
    except Exception as e:
        print(f"\nError loading {dataset} dataset: {str(e)}")
        return None

def prepare_data(df, test_size=0.3):
    """Prepare and split data"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=test_size, random_state=42), X.columns 