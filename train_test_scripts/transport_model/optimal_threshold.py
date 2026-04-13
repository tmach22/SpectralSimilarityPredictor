import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# 1. Load the raw predictions from your test script
# (Update this path if your test script saved it somewhere else)
results_csv = "/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor/results/predictions/binary_classifier/transport_model/stage3_ot_test_predictions.csv"
print(f"Loading predictions from {results_csv}...")
df = pd.read_csv(results_csv)

y_true = df['true_label'].values
y_probs = df['prob_similarity'].values

# Calculate Base AUC (Threshold Independent)
auc = roc_auc_score(y_true, y_probs)
print(f"Base ROC-AUC: {auc:.4f}\n")

# 2. Scan Thresholds from 0.01 to 0.99
thresholds = np.linspace(0.01, 0.99, 100)
f1_scores = []
precisions = []
recalls = []

for t in thresholds:
    # Apply threshold
    y_pred = (y_probs >= t).astype(float)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    
    f1_scores.append(f1)
    precisions.append(p)
    recalls.append(r)

# 3. Find the Optimal Threshold
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_p = precisions[best_idx]
best_r = recalls[best_idx]

print("="*50)
print(f"DEFAULT METRICS (Threshold = 0.500)")
print("="*50)
default_pred = (y_probs >= 0.5).astype(float)
print(f"Accuracy:  {accuracy_score(y_true, default_pred):.4f}")
print(f"Precision: {precision_score(y_true, default_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_true, default_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_true, default_pred, zero_division=0):.4f}")

print("\n" + "="*50)
print(f"OPTIMIZED METRICS (Threshold = {best_threshold:.3f})")
print("="*50)
best_pred = (y_probs >= best_threshold).astype(float)
print(f"Accuracy:  {accuracy_score(y_true, best_pred):.4f}")
print(f"Precision: {best_p:.4f}")
print(f"Recall:    {best_r:.4f}")
print(f"F1 Score:  {best_f1:.4f}")
print("="*50)

# Optional: Plot the curves to visualize the tradeoff
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', color='blue', alpha=0.7)
plt.plot(thresholds, recalls, label='Recall', color='red', alpha=0.7)
plt.plot(thresholds, f1_scores, label='F1 Score', color='green', linewidth=2)
plt.axvline(best_threshold, color='black', linestyle='--', label=f'Optimal Threshold ({best_threshold:.2f})')
plt.title('Precision-Recall Tradeoff vs. Decision Threshold')
plt.xlabel('Probability Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig('threshold_tuning.png')
print("\nSaved tuning visualization to threshold_tuning.png")