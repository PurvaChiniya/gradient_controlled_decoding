import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from itertools import product
import numpy as np

# Load the data
file_path = 'security_output_refined.csv'
df = pd.read_csv(file_path)

# Define a function to calculate metrics
def calculate_metrics(df, t1, t2):
    # Apply thresholds
    df['final_label'] = ((df['sure'] >= t1) & (df['sorry'] >= t2)).astype(int)
    
    # Calculate metrics
    f1 = f1_score(df['ground_truth'], df['final_label'])
    precision = precision_score(df['ground_truth'], df['final_label'])
    recall = recall_score(df['ground_truth'], df['final_label'])
    tn, fp, fn, tp = confusion_matrix(df['ground_truth'], df['final_label']).ravel()
    
    # Attack Success Rate (ASR)
    asr = 100 - (tp / sum(df['ground_truth']==1)) * 100

    return f1, precision, recall, fp, asr

# Define a range of thresholds to test
t1_values = np.linspace(0, 1, 100)
t2_values = np.linspace(0, 1, 100)

best_f1 = 0
best_t1 = 0
best_t2 = 0
best_metrics = {}

# Iterate over all combinations of t1 and t2
for t1, t2 in product(t1_values, t2_values):
    f1, precision, recall, fp, asr = calculate_metrics(df, t1, t2)
    if f1 > best_f1:
        best_f1 = f1
        best_t1 = t1
        best_t2 = t2
        best_metrics = {
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall,
            'False Positives': fp,
            'Attack Success Rate': asr
        }

# Output the best thresholds and corresponding metrics
print(f"Best t1: {best_t1}")
print(f"Best t2: {best_t2}")
print(f"Best metrics: {best_metrics}")
