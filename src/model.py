import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
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
        # Return 2D array with shape (n_samples, 2)
        return np.vstack([1 - probabilities, probabilities]).T 

def get_models():
    models = {
        'Logistic Regression': LogisticRegression(learning_rate=0.01, n_iterations=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    return models

def compare_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = {}
    
    print("\n=== Model Performance Metrics by Class ===")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 30)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
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