import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve
import numpy as np

def create_confusion_matrix_plot(cm):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='RdBu'
    ))
    
    # Update layout to make the plot smaller
    fig.update_layout(
        width=500,  # Reduced from default
        height=400,  # Reduced from default
        margin=dict(l=50, r=50, t=30, b=30),  # Reduced margins
        font=dict(size=12)  # Slightly smaller font
    )
    return fig

def create_roc_curve(fpr, tpr, thresholds, roc_auc):
    # Create hover text with thresholds
    hover_text = [f'Threshold: {threshold:.3f}<br>FPR: {fpr_val:.3f}<br>TPR: {tpr_val:.3f}'
                 for fpr_val, tpr_val, threshold in zip(fpr[1:], tpr[1:], thresholds)]
    # Add the point for threshold 1.0 (first point)
    hover_text.insert(0, f'Threshold: 1.000<br>FPR: {fpr[0]:.3f}<br>TPR: {tpr[0]:.3f}')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC (AUC = {roc_auc:.3f})',
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Classifier'
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 1]),
        showlegend=True
    )
    return fig

def find_best_threshold(y_true, y_prob, min_precision=0.8):
    """Find the best threshold where recall is highest and precision is greater than min_precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    best_threshold = None
    best_recall = 0

    for precision, recall, threshold in zip(precisions, recalls, thresholds):
        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_threshold = threshold

    return best_threshold

def create_precision_recall_curve(precision, recall, thresholds, avg_precision):
    # Create hover text with thresholds
    # Note: precision_recall_curve returns one more point than thresholds
    hover_text = [f'Threshold: {threshold:.3f}<br>Precision: {prec:.3f}<br>Recall: {rec:.3f}'
                 for prec, rec, threshold in zip(precision[:-1], recall[:-1], thresholds)]
    # Add the last point (threshold = 0.0)
    hover_text.append(f'Threshold: 0.000<br>Precision: {precision[-1]:.3f}<br>Recall: {recall[-1]:.3f}')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, 
        y=precision,
        mode='lines',
        name=f'P-R (AP = {avg_precision:.3f})',
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 1]),
        showlegend=True
    )
    return fig

