from ydata_profiling import ProfileReport
from src.data import load_data

def perform_eda():
    """Automated EDA using pandas-profiling"""
    
    dataset_name = 'cleveland'
    df = load_data()
    
        
    if df is not None:
        # Generate profile report
        profile = ProfileReport(df, 
            title=f"{dataset_name.title()} Heart Disease Dataset Analysis",
            minimal=True,  # Faster generation
            explorative=True  # Include more detailed analysis
        )
        
        # Save report
        profile.to_file(f"output/eda_{dataset_name}.html")

if __name__ == "__main__":
    perform_eda() 