import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

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
    
    # Update layout to match confusion matrix size
    fig.update_layout(
        width=500,
        height=400,
        margin=dict(l=50, r=50, t=30, b=30),
        font=dict(size=12),
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    return fig
