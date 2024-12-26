from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC
from yellowbrick.model_selection import LearningCurve
import matplotlib.pyplot as plt
import os

def visualize_model(model, X_train, X_test, y_train, y_test, model_name):
    """Generate Yellowbrick visualizations for model analysis"""
    output_dir = os.path.join('output', 'yellowbrick', model_name.lower().replace(' ', '_'))
    os.makedirs(output_dir, exist_ok=True)
    
    visualizers = {
        'classification_report': ClassificationReport(model, support=True),
        'confusion_matrix': ConfusionMatrix(model),
        'roc_curve': ROCAUC(model),
        'learning_curve': LearningCurve(model, scoring='f1')
    }
    
    for name, viz in visualizers.items():
        if name != 'learning_curve':
            viz.fit(X_train, y_train)
            viz.score(X_test, y_test)
        else:
            viz.fit(X_train, y_train)
        viz.show(outpath=os.path.join(output_dir, f'{name}.png'))
        plt.close() 