import pandas as pd
import plotly.express as px

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
