import pandas as pd
import matplotlib.pyplot as plt
from dython.nominal import associations
import seaborn as sns
from src.data import load_data

def perform_eda():
    # Load your dataset
    df = load_data()

    # Separate continuous and categorical features
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Calculate associations for categorical features
    associations_matrix = associations(df[categorical_features + ['target']], 
                                       nominal_columns='all', 
                                       plot=True, 
                                       figsize=(10, 8))

    plt.tight_layout()
    plt.show()

    # Correlation for continuous features
    corr = df[continuous_features + ['target']].corr()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title('Correlation Matrix for Continuous Features')
    plt.show()

if __name__ == "__main__":
    perform_eda() 