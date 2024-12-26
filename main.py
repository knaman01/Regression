from src.data import load_data, prepare_data
from src.model import LogisticRegression, compare_models
from src.visualization import generate_report
from sklearn.metrics import accuracy_score, classification_report
import os

os.environ['ENABLE_FEATURE_ENGINEERING'] = 'True'  # or 'False'

def main():
    # Load data
    df = load_data()
    print("\nDataset Information:")
    print(f"Total patients: {len(df)}")
    print(f"Features: {len(df.columns)-1}")
    
    # Prepare data
    (X_train, X_test, y_train, y_test), feature_names = prepare_data(df)
    
    # Train model
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Print results
    print("\nModel Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
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