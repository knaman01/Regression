import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from .html_components import create_html_content

def generate_report(models_results, feature_names, y_test):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%B_%d_%Y_%I%M%p").lower()
    filename = os.path.join(output_dir, f"model_comparison_{timestamp}.html")
    
    # Initialize comparison data dictionary
    comparison_data = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'AUC-ROC': []
    }
    
    # Populate comparison data for each model
    for model_name, results in models_results.items():
        y_pred = results['predictions']
        y_prob = results['probabilities']
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate ROC AUC if probabilities are available
        roc_auc = None
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
        
        # Add data to comparison
        comparison_data['Model'].append(model_name)
        comparison_data['Accuracy'].append(report['accuracy'])
        comparison_data['Precision'].append(report['weighted avg']['precision'])
        comparison_data['Recall'].append(report['weighted avg']['recall'])
        comparison_data['F1 Score'].append(report['weighted avg']['f1-score'])
        comparison_data['AUC-ROC'].append(roc_auc)
    
    generate_html_report(comparison_data, models_results, y_test, filename)
    return filename

def generate_html_report(comparison_data, models_results, y_test, filename):
    html_content = create_html_content(comparison_data, models_results, y_test)
    with open(filename, "w") as f:
        f.write(html_content)
