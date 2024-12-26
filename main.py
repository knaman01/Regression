from src.data import load_data, prepare_data
from src.model import compare_models
from src.visualization import generate_report
import os

def main():
    # Load data
    df = load_data()
    print("\nDataset Information:")
    print(f"Total patients: {len(df)}")
    print(f"Features: {len(df.columns)-1}")
    
    # Prepare data
    (X_train, X_test, y_train, y_test), feature_names = prepare_data(df)
    
    # Compare different models
    model_results = compare_models(X_train, X_test, y_train, y_test)
    
    # Generate single report with all models
    report_file = generate_report(
        models_results=model_results,
        feature_names=feature_names,
        y_test=y_test
    )
    print(f"\nComparison report generated: {report_file}")

if __name__ == "__main__":
    main() 