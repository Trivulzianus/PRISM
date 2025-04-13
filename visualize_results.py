import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
from PRISM import PRISM
import logging

# Configure warning level to reduce console noise
logging.getLogger("PRISM").setLevel(logging.ERROR)

# Load the trained model
print("Loading the trained PRISM model...")
prism = PRISM()
prism.load_model('test_results/prism_demo')

# Load the original dataset
print("Loading original dataset...")
df = prism.load_dataset('llm_inputs_dataset.csv', text_col='input', label_col='label', malicious_label='malicious')

# Add batch prediction methods to reduce warning messages
def batch_predict(prism, texts, batch_size=10):
    """Run predictions in batch mode to reduce warnings"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            result = prism.predict(text)
            results.append(result)
        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} predictions", end="\r")
    print()  # Add newline after progress indicator
    return results

def batch_predict_proba(prism, texts, batch_size=10):
    """Run probability predictions in batch mode to reduce warnings"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            result = prism.predict_proba(text)
            results.append(result['malicious_probability'])
        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} probability calculations", end="\r")
    print()  # Add newline after progress indicator
    return results

# Get predictions on all data
print("Making predictions on the entire dataset...")
all_texts = df["clean_text"].values
results = batch_predict(prism, all_texts)
probabilities = batch_predict_proba(prism, all_texts)

# Build predictions list
predictions = []
for idx, text in enumerate(all_texts):
    predictions.append({
        'text': text[:50] + '...' if len(text) > 50 else text,
        'actual': 'Malicious' if df["binary_label"].values[idx] == 1 else 'Benign',
        'predicted': results[idx],
        'probability': probabilities[idx]
    })

# Create DataFrame with predictions
results_df = pd.DataFrame(predictions)

# Save all predictions for reference
results_df.to_csv('test_results/predictions_demo.csv', index=False)
print("Saved all predictions to 'test_results/predictions_demo.csv'")

# Calculate metrics
print("Calculating performance metrics...")
y_true = [1 if actual == 'Malicious' else 0 for actual in results_df['actual']]
y_pred = [1 if pred == 'Malicious' else 0 for pred in results_df['predicted']]

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Print metrics
print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print confusion matrix
print("\nConfusion Matrix:")
print(f"True Negative: {tn}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"True Positive: {tp}")

# Create a visual confusion matrix
plt.figure(figsize=(10, 8))
labels = ['Benign', 'Malicious']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('PRISM Confusion Matrix', fontsize=16)
plt.savefig('test_results/confusion_matrix.png')
print("Confusion matrix visualization saved to 'test_results/confusion_matrix.png'")

# Create histogram of prediction probabilities by class
plt.figure(figsize=(12, 6))
benign_probs = results_df[results_df['actual'] == 'Benign']['probability']
malicious_probs = results_df[results_df['actual'] == 'Malicious']['probability']

plt.hist(benign_probs, alpha=0.5, label='Benign', bins=20, range=(0,1), color='green')
plt.hist(malicious_probs, alpha=0.5, label='Malicious', bins=20, range=(0,1), color='red')
plt.title('Distribution of Malicious Probability Scores by Class', fontsize=16)
plt.xlabel('Malicious Probability Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('test_results/probability_distribution.png')
print("Probability distribution visualization saved to 'test_results/probability_distribution.png'")

# Calculate metrics by probability threshold
thresholds = np.arange(0.1, 1.0, 0.1)
results = []

for threshold in thresholds:
    y_pred_threshold = [1 if prob >= threshold else 0 for prob in results_df['probability']]
    cm = confusion_matrix(y_true, y_pred_threshold)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Plot metrics by threshold
results_df_threshold = pd.DataFrame(results)
plt.figure(figsize=(12, 6))
plt.plot(results_df_threshold['threshold'], results_df_threshold['accuracy'], marker='o', label='Accuracy')
plt.plot(results_df_threshold['threshold'], results_df_threshold['precision'], marker='s', label='Precision')
plt.plot(results_df_threshold['threshold'], results_df_threshold['recall'], marker='^', label='Recall')
plt.plot(results_df_threshold['threshold'], results_df_threshold['f1'], marker='*', label='F1 Score')
plt.title('Performance Metrics by Probability Threshold', fontsize=16)
plt.xlabel('Probability Threshold', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('test_results/threshold_performance.png')
print("Threshold performance visualization saved to 'test_results/threshold_performance.png'")

# Save misclassifications for error analysis
misclassified = results_df[results_df['actual'] != results_df['predicted']]
misclassified.to_csv('test_results/misclassified_examples.csv', index=False)
print(f"Saved {len(misclassified)} misclassified examples to 'test_results/misclassified_examples.csv'")

print("\nVisualization and analysis complete!")