from PRISM import PRISM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Initialize PRISM
prism = PRISM()

# Load the dataset
print("Loading the custom dataset...")
df = prism.load_dataset('llm_inputs_dataset_large.csv', text_col='input', label_col='label', malicious_label='malicious')

print(f"Dataset loaded with {len(df)} samples")
print(f"Malicious samples: {df['binary_label'].sum()}, Benign samples: {len(df) - df['binary_label'].sum()}")

# Train the model
print("\nTraining the PRISM model...")
metrics = prism.train(df)

# Print metrics
print("\nTraining Results:")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Print confusion matrix
print("\nConfusion Matrix:")
cm = metrics["confusion_matrix"]
print(f"True Negative: {cm['true_negative']}")
print(f"False Positive: {cm['false_positive']}")
print(f"False Negative: {cm['false_negative']}")
print(f"True Positive: {cm['true_positive']}")

# Save the model with anti-overfitting tag
print("\nSaving the model...")
prism.save_model('test_results/prism_anti_overfitting')

# Create a visual confusion matrix
cm_values = np.array([[cm['true_negative'], cm['false_positive']], 
                      [cm['false_negative'], cm['true_positive']]])
labels = ['Benign', 'Malicious']
disp = ConfusionMatrixDisplay(confusion_matrix=cm_values, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('PRISM Anti-Overfitting Confusion Matrix')
plt.savefig('test_results/confusion_matrix.png')
print("Confusion matrix visualization saved to 'test_results/confusion_matrix.png'")

# Now evaluate on a separate test set for deeper analysis
print("\nEvaluating against untrained data...")
# For this test, let's set aside 20% of our data that wasn't used in training
X = df["clean_text"].values
y = df["binary_label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Get predictions on test data
predictions = []
for text in X_test:
    # Transform features
    X_word = prism.word_vectorizer.transform([text])
    X_char = prism.char_vectorizer.transform([text])
    X_stats = prism.extract_statistical_features([text])
    
    # Combine features
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([
        X_word,
        X_char,
        csr_matrix(X_stats)
    ])
    
    # Get prediction
    is_malicious = bool(prism.classifier.predict(X_combined)[0])
    proba = prism.classifier.predict_proba(X_combined)[0, 1]
    
    predictions.append({
        'text': text[:50] + '...' if len(text) > 50 else text,
        'actual': 'Malicious' if y_test[list(X_test).index(text)] == 1 else 'Benign',
        'predicted': 'Malicious' if is_malicious else 'Benign',
        'probability': proba
    })

# Save predictions to CSV
results_df = pd.DataFrame(predictions)
results_df.to_csv('test_results/predictions_demo.csv', index=False)
print("Prediction results saved to 'test_results/predictions_demo.csv'")

# Calculate metrics on test set
test_predictions = [1 if p['predicted'] == 'Malicious' else 0 for p in predictions]
test_cm = confusion_matrix(y_test, test_predictions)
tn, fp, fn, tp = test_cm.ravel()
test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

print("\nTest Set Results:")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

print("\nComplete! The model has been trained and evaluated successfully.")