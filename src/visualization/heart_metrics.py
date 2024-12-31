import plotly.graph_objects as go
from datetime import datetime
import os

def generate_heart_disease_report(models_results, y_test):
    """Generate HTML report focusing on heart disease detection metrics"""
    
    # Create comparison table
    comparison_table = go.Figure(data=[go.Table(
        header=dict(values=['Model', 'Precision', 'Recall', 'F1 Score', 'Missed Cases', 'False Alarms'],
                   fill_color='navy',
                   align='left',
                   font=dict(color='white', size=12)),
        cells=dict(values=[
            list(models_results.keys()),
            [f"{results['accuracy']:.2%}" for results in models_results.values()],
            [f"{results['recall']:.2%}" for results in models_results.values()],
            [f"{results['f1_score']:.2%}" for results in models_results.values()],
            [results['missed_cases'] for results in models_results.values()],
            [results['false_alarms'] for results in models_results.values()]
        ],
        align='left'))
    ])
    
    comparison_table.update_layout(
        title="Heart Disease Detection - Model Comparison",
        title_x=0.5,
        width=1000,
        height=400
    )
    
    # Create directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%B_%d_%Y_%I%M%p").lower()
    filename = f'output/heart_disease_comparison_{timestamp}.html'
    
    # Write HTML file
    with open(filename, 'w') as f:
        f.write("<h1>Heart Disease Detection Analysis</h1>")
        f.write(comparison_table.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<br><br>")
        f.write("<p><b>Metrics Explanation:</b></p>")
        f.write("<ul>")
        f.write("<li><b>Precision:</b> When we predict heart disease, how often are we correct?</li>")
        f.write("<li><b>Recall:</b> What percentage of actual heart disease cases do we catch?</li>")
        f.write("<li><b>Missed Cases:</b> Number of heart disease cases we failed to identify</li>")
        f.write("<li><b>False Alarms:</b> Number of healthy patients incorrectly flagged for heart disease</li>")
        f.write("</ul>")
    
    print(f"\nHeart Disease Detection report generated: {filename}")
    return filename 