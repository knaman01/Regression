from src.data import load_data, prepare_data
from src.model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    # Print results
    print("\nTest Set Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main() 