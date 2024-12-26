import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from datetime import datetime

def plot_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.weights)
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance in Heart Disease Prediction'
    )
    return fig

def generate_report(models_results, feature_names, y_test):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison table data
    comparison_data = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'AUC-ROC': []
    }
    
    for name, results in models_results.items():
        y_pred = results['predictions']
        y_prob = results['probabilities']
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        comparison_data['Model'].append(name)
        comparison_data['Accuracy'].append(results['accuracy'])
        comparison_data['Precision'].append(class_report['1']['precision'])
        comparison_data['Recall'].append(class_report['1']['recall'])
        comparison_data['F1 Score'].append(class_report['1']['f1-score'])
        
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            comparison_data['AUC-ROC'].append(roc_auc)
        else:
            comparison_data['AUC-ROC'].append(None)
    
    # Generate HTML report
    filename = f"model_comparison_{timestamp}.html"
    generate_html_report(comparison_data, models_results, y_test, filename)
    
    return filename

def create_html_content(comparison_data, models_results, y_test):
    # Create comparison table HTML
    comparison_table = create_comparison_table(comparison_data)
    
    html_content = f"""
    <html>
    <head>
        <title>Heart Disease Prediction - Model Comparison</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 2rem;
                text-align: center;
                margin-bottom: 2rem;
                border-radius: 0 0 10px 10px;
            }}
            .comparison-table {{
                margin: 2rem 0;
                overflow-x: auto;
            }}
            table.comparison {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            table.comparison th,
            table.comparison td {{
                padding: 12px 15px;
                text-align: center;
            }}
            table.comparison th {{
                background-color: #2c3e50;
                color: white;
                font-weight: 500;
            }}
            table.comparison tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .model-section {{
                background: white;
                margin-bottom: 2rem;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Heart Disease Prediction - Model Comparison</h1>
            <div class="timestamp">Generated on: {datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")}</div>
        </div>
        
        <div class="container">
            {comparison_table}
            {create_model_sections(models_results, y_test)}
        </div>
    </body>
    </html>
    """
    return html_content

def create_comparison_table(comparison_data):
    table_html = """
        <div class="section comparison-table">
            <h2>Model Comparison Summary</h2>
            <table class="comparison">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>AUC-ROC</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for i in range(len(comparison_data['Model'])):
        table_html += f"""
            <tr>
                <td>{comparison_data['Model'][i]}</td>
                <td>{comparison_data['Accuracy'][i]:.2%}</td>
                <td>{comparison_data['Precision'][i]:.2%}</td>
                <td>{comparison_data['Recall'][i]:.2%}</td>
                <td>{comparison_data['F1 Score'][i]:.2%}</td>
                <td>{'N/A' if comparison_data['AUC-ROC'][i] is None else f"{comparison_data['AUC-ROC'][i]:.3f}"}</td>
            </tr>
        """
    
    table_html += """
                </tbody>
            </table>
        </div>
    """
    return table_html

def create_model_sections(models_results, y_test):
    sections_html = ""
    for name, results in models_results.items():
        y_pred = results['predictions']
        y_prob = results['probabilities']
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No', 'Predicted Yes'],
            y=['Actual No', 'Actual Yes'],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='RdBu'
        ))
        
        sections_html += f"""
            <div class="model-section">
                <h2>{name}</h2>
                <div class="plot-container">
                    {fig_cm.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
            </div>
        """
        
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = create_roc_curve(fpr, tpr, roc_auc)
            sections_html += f"""
                <div class="plot-container">
                    {fig_roc.to_html(full_html=False, include_plotlyjs='cdn')}
                </div>
            """
    
    return sections_html

def create_roc_curve(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        mode='lines',
        line=dict(color='#2ecc71', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='#95a5a6')
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    return fig

def generate_html_report(comparison_data, models_results, y_test, filename):
    html_content = create_html_content(comparison_data, models_results, y_test)
    with open(filename, "w") as f:
        f.write(html_content) 