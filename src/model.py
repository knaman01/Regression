import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data import load_data
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from dython.nominal import associations

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "n_iterations": self.n_iterations}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted] 
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_model)
        return np.vstack([1 - probabilities, probabilities]).T 

def get_models():
    models = {
        'Logistic Regression': LogisticRegression(learning_rate=0.01, n_iterations=1000),
        # 'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        # 'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        # 'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        # 'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    return models

def compare_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    
    print("\n=== Model Performance Metrics by Class ===")
    print("=" * 50)
    
    # Load best thresholds
    try:
        with open('output/best_thresholds.json', 'r') as f:
            best_thresholds = json.load(f)
    except FileNotFoundError:
        best_thresholds = {model_name: 0.5 for model_name in get_models().keys()}
    

    cv_results = evaluate_models_with_cv(X_train, y_train) 
    print(cv_results)
        

    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 30)
        
        # Train model
        model.fit(X_train, y_train)
        
        
        # Make predictions using custom threshold
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        threshold = best_thresholds.get(name, 0.5)  # Default to 0.5 if not found
        print (threshold)

        y_pred = (y_prob >= threshold).astype(int) if y_prob is not None else model.predict(X_test)
        
        # Get classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Print metrics for each class
        print("Class 0 (No Heart Disease):")
        print(f"Precision:    {report['0']['precision']:.4f} ({int(report['0']['precision'] * 100)}%)")
        print(f"Recall:       {report['0']['recall']:.4f} ({int(report['0']['recall'] * 100)}%)")
        print(f"F1-Score:     {report['0']['f1-score']:.4f}")
        
        print("\nClass 1 (Heart Disease):")
        print(f"Precision:    {report['1']['precision']:.4f} ({int(report['1']['precision'] * 100)}%)")
        print(f"Recall:       {report['1']['recall']:.4f} ({int(report['1']['recall'] * 100)}%)")
        print(f"F1-Score:     {report['1']['f1-score']:.4f}")
        
        # Calculate and print confusion matrix numbers
        true_negatives = sum((y_test == 0) & (y_pred == 0))
        false_positives = sum((y_test == 0) & (y_pred == 1))
        false_negatives = sum((y_test == 1) & (y_pred == 0))
        true_positives = sum((y_test == 1) & (y_pred == 1))
        
        print("\nDetailed Numbers:")
        print("No Heart Disease (Class 0):")
        print(f"Total Cases:           {sum(y_test == 0)}")
        print(f"Correctly Identified:  {true_negatives}")
        print(f"Incorrectly Flagged:  {false_positives}")
        
        print("\nHeart Disease (Class 1):")
        print(f"Total Cases:           {sum(y_test == 1)}")
        print(f"Correctly Identified:  {true_positives}")
        print(f"Missed Cases:          {false_negatives}")
        
        # Store results
        results[name] = {
            'predictions': y_pred,
            'probabilities': y_prob,
            'metrics': report,
            'raw_numbers': {
                'true_positives': true_positives,
                'true_negatives': true_negatives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
        }
    
    return results 

def evaluate_models_with_cv(X, y):
    models = get_models()
    cv_results = {}

    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nEvaluating {name} with cross-validation:")
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        cv_results[name] = scores
        print(f"F1 Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return cv_results

