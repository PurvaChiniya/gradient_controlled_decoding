import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Load and prepare the data
df = pd.read_csv("advbench.csv")
df["ground_truth"] = 1  # Assume this is the correct way to set your labels

def compute_f1_scores(data, label_column, score_column):
    labels = data[label_column]
    scores = data[score_column]
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Handling NaN values
    
    max_f1_index = np.argmax(f1_scores)
    max_f1_threshold = thresholds[max_f1_index]
    max_f1_score = f1_scores[max_f1_index]
    
    return precision, recall, thresholds, max_f1_score, max_f1_threshold, max_f1_index

def plot_pr_curve(precision, recall, thresholds, max_f1_index, score_type):
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
    plt.scatter(thresholds[max_f1_index], precision[max_f1_index], c='red', label='Max F1 Point')
    plt.title(f'Precision-Recall curve for "{score_type}" response')
    plt.xlabel('Threshold')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"advbench_{score_type}.png")
    plt.show()

# Calculate and plot for 'sure' scores
precision_sure, recall_sure, thresholds_sure, max_f1_score_sure, max_f1_threshold_sure, max_f1_index_sure = compute_f1_scores(df, 'ground_truth', 'sure')
print(f"Maximum F1 Score for 'Sure': {max_f1_score_sure}")
print(f"Threshold at Maximum F1 Score for 'Sure': {max_f1_threshold_sure}")
plot_pr_curve(precision_sure, recall_sure, thresholds_sure, max_f1_index_sure, 'Sure')

# Calculate and plot for 'sorry' scores if needed
precision_sorry, recall_sorry, thresholds_sorry, max_f1_score_sorry, max_f1_threshold_sorry, max_f1_index_sorry = compute_f1_scores(df, 'ground_truth', 'sorry')
print(f"Maximum F1 Score for 'Sorry': {max_f1_score_sorry}")
print(f"Threshold at Maximum F1 Score for 'Sorry': {max_f1_threshold_sorry}")
plot_pr_curve(precision_sorry, recall_sorry, thresholds_sorry, max_f1_index_sorry, 'Sorry')
