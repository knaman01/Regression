from src.data import load_data, prepare_data
from src.model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(X_train, X_test, y_train, y_test, dataset_name=""):
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\nResults on {dataset_name} dataset:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return model

def main():
    # Test on both datasets
    for dataset in ['cleveland', 'openml']:
        print(f"\n=== {dataset.title()} Dataset ===")
        df = load_data(dataset)
        if df is not None:
            try:
                (X_train, X_test, y_train, y_test), feature_names = prepare_data(df)
                evaluate_model(X_train, X_test, y_train, y_test, dataset)
            except Exception as e:
                print(f"Error processing {dataset} dataset: {str(e)}")

if __name__ == "__main__":
    main() 