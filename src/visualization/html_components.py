from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from .plot_components import create_confusion_matrix_plot, create_roc_curve, create_precision_recall_curve

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
                        <th>Best Threshold</th>
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
                <td>{'N/A' if comparison_data['Best Threshold'][i] is None else f"{comparison_data['Best Threshold'][i]:.3f}"}</td>
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
        
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = create_confusion_matrix_plot(cm)
        
        # Create ROC and PR curve buttons HTML separately
        roc_button = ''
        pr_button = ''
        if y_prob is not None:
            roc_button = f'<button class="tab-button" onclick="openTab(event, \'{name}-roc\', \'{name}\')">ROC Curve</button>'
            pr_button = f'<button class="tab-button" onclick="openTab(event, \'{name}-pr\', \'{name}\')">Precision-Recall Curve</button>'
        
        sections_html += f"""
            <div class="model-section" id="{name}">
                <h2>{name}</h2>
                <div class="tabs">
                    <button class="tab-button active" onclick="openTab(event, '{name}-confusion', '{name}')">Confusion Matrix</button>
                    {roc_button}
                    {pr_button}
                </div>
                
                <div id="{name}-confusion" class="tab-content active">
                    <div class="plot-container">
                        {fig_cm.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
        """
        
        if y_prob is not None:
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc = create_roc_curve(fpr, tpr, roc_thresholds, roc_auc)
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            fig_pr = create_precision_recall_curve(precision, recall, pr_thresholds, avg_precision)
            
            sections_html += f"""
                <div id="{name}-roc" class="tab-content">
                    <div class="plot-container">
                        {fig_roc.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
                <div id="{name}-pr" class="tab-content">
                    <div class="plot-container">
                        {fig_pr.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
            """
        
        sections_html += "</div>"  # Close model-section
    
    return sections_html

def create_html_content(comparison_data, models_results, y_test):
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
            .tabs {{
                margin-bottom: 20px;
                border-bottom: 2px solid #e9ecef;
            }}
            .tab-button {{
                background-color: transparent;
                border: none;
                padding: 10px 20px;
                cursor: pointer;
                font-size: 16px;
                margin-right: 5px;
                transition: all 0.3s;
            }}
            .tab-button:hover {{
                background-color: #e9ecef;
            }}
            .tab-button.active {{
                border-bottom: 2px solid #2c3e50;
                color: #2c3e50;
                font-weight: 500;
            }}
            .tab-content {{
                display: none;
                padding: 20px 0;
            }}
            .tab-content.active {{
                display: block;
                padding-bottom: 0;
            }}
            .plot-container {{
                margin-bottom: 0;
                height: 400px;
            }}
        </style>
        <script>
            function openTab(evt, tabName, sectionId) {{
                // Get the specific model section
                var modelSection = document.getElementById(sectionId);
                
                // Hide all tab content within this section
                var tabcontent = modelSection.getElementsByClassName("tab-content");
                for (var i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                
                // Remove active class from all tab buttons within this section
                var tablinks = modelSection.getElementsByClassName("tab-button");
                for (var i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                
                // Show the selected tab content and mark the button as active
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }}
        </script>
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
