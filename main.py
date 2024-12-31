from src.data import load_data, prepare_data
from src.model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from src.feature_eng import FeatureEngineeringConfig, prepare_features
from sklearn.model_selection import train_test_split

def evaluate_model(X_train, X_test, y_train, y_test, dataset_name=""):
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Get classification report as a dict
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\nResults on {dataset_name} dataset:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClass 1 (Heart Disease) Metrics:")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall:    {report['1']['recall']:.4f}")
    print(f"F1-Score:  {report['1']['f1-score']:.4f}")
    return model

def main():
    # Test on both datasets
    for dataset in ['cleveland', 'openml']:
        print(f"\n=== {dataset.title()} Dataset ===")
        df = load_data(dataset)
        if df is not None:
            try:
                # Configure feature engineering
                config = FeatureEngineeringConfig()
                config.ENABLE = False
                config.AGE_FEATURES = True
                config.INTERACTION_TERMS = True

                # print(f"Original features: {len(df.columns)}")
                X_scaled, y, feature_names = prepare_features(df, config=config)
                # print(f"After feature engineering: {len(feature_names)}")
                # print("\nNew features:", feature_names)

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42
                )

                evaluate_model(X_train, X_test, y_train, y_test, dataset)
            except Exception as e:
                print(f"Error processing {dataset} dataset: {str(e)}")

if __name__ == "__main__":
    main() 