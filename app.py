import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def compute_summary_statistics(df):
    """Calculate summary statistics for the DataFrame."""
    stats = {
        'average': df.mean(),
        'median_value': df.median(),
        'most_frequent': df.mode().iloc[0],
        'standard_deviation': df.std(),
        'correlation': df.corr()
    }
    return stats

def create_visualizations(df):
    """Generate various plots for the numeric columns in the DataFrame."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Create histograms for numeric columns
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df[col].dropna(), bins=30, edgecolor='black')
        plt.title(f'Histogram for {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

    # Create scatter plots for pairs of numeric columns
    for i, col in enumerate(numeric_columns):
        for next_col in numeric_columns[i+1:]:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[col], df[next_col], alpha=0.5)
            plt.title(f'Scatter Plot of {col} vs {next_col}')
            plt.xlabel(col)
            plt.ylabel(next_col)
            plt.show()

    # Create line plots for each numeric column
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[col])
        plt.title(f'Line Plot for {col}')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.show()

def run_analysis():
    """Main function to execute the data analysis and visualization."""
    file_path = input("Please enter the CSV file path: ")
    data_frame = load_data(file_path)

    summary_stats = compute_summary_statistics(data_frame)
    print("Summary Statistics:")
    for key, value in summary_stats.items():
        print(f"{key.replace('_', ' ').capitalize()}:\n{value}\n")

    create_visualizations(data_frame)

    print("Data analysis complete. All plots have been generated.")

if __name__ == "__main__":
    run_analysis()


import os
os.environ["OPENAI_API_KEY"] = ""

df = pd.read_csv('diabetes.csv')

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI

agent = create_csv_agent(OpenAI(temperature=0), '/content/diabetes.csv',verbose=True, allow_dangerous_code=True)


# Run a query using the agent
try:
    response = agent.invoke("How many rows are there?")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
