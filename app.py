from src.data_loader import load_data
from src.eda import basic_eda
from src.insight_generator import generate_insights
from src.visualization import plot_correlation
print("Loading data...")
data=load_data('data/sample_data.csv')
eda=basic_eda(data)

print("Dataset Shape:", eda["shape"])
print("\nColumns:", eda["columns"])
print("\nMissing Values:\n", eda["missing_values"])
print("\nStatistical Summary:\n", eda["describe"])


plot_correlation(data)  
    
insights=generate_insights(data)
print("\nAI-Insights:")
for insight in insights:
    print("- ",insight)