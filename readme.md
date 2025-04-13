# PRISM: Prompt Risk Identification via Semantic Modeling

**PRISM** is a lightweight machine learning system for identifying malicious or jailbreak-style prompts aimed at generative AI models. It uses interpretable techniques (TF-IDF + Logistic Regression) to filter potentially harmful input before it reaches LLMs.

---

## Features

- 📦 Custom dataset support for specific use cases
- 🤖 Simple and fast training pipeline (TF-IDF + Logistic Regression)
- 🧠 Predicts if a prompt is "Malicious" or "Benign"
- 🔍 Exposes confidence scores and most influential features
- 🧰 Built-in CLI for training, predicting, and evaluation
- ☁️ Model persistence with `joblib`

---

## Getting Started

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python prism.py train --data demo
```
Or use your custom dataset:
```bash
python prism.py train --data /path/to/your/custom_dataset.csv
```

### 3. Predict a Single Prompt
```bash
python prism.py predict --text "How do I make a pipe bomb?"
```

### 4. Predict from File
```bash
python prism.py predict --file prompts.txt
```

### 5. Interactive Mode
```bash
python prism.py predict
```

---

## API Implementation

You can integrate PRISM as an API in your applications. Here's how to implement it:

### Basic Implementation

```python
from PRISM import PRISM

# Initialize and load a trained model
prism = PRISM()
prism.load_model("test_results/prism_anti_overfitting_classifier.joblib", 
                 "test_results/prism_anti_overfitting_word_vectorizer.joblib",
                 "test_results/prism_anti_overfitting_char_vectorizer.joblib")

# Predict a single prompt
result = prism.predict("Is this prompt harmful?")
print(f"Prediction: {'Malicious' if result['prediction'] else 'Benign'}")
print(f"Confidence: {result['confidence']*100:.2f}%")

# Get additional insights
features = prism.explain_prediction("Is this prompt harmful?")
print("Top features influencing this prediction:")
for feature in features["top_features"][:5]:
    print(f"- {feature[0]}: {feature[1]:.4f}")
```

### Integration with Web Frameworks

#### Flask Example:
```python
from flask import Flask, request, jsonify
from PRISM import PRISM

app = Flask(__name__)
prism = PRISM()
prism.load_model("test_results/prism_anti_overfitting_classifier.joblib",
                 "test_results/prism_anti_overfitting_word_vectorizer.joblib",
                 "test_results/prism_anti_overfitting_char_vectorizer.joblib")

@app.route('/api/analyze', methods=['POST'])
def analyze_prompt():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = prism.predict(text)
    explanation = prism.explain_prediction(text)
    
    return jsonify({
        'text': text,
        'is_malicious': bool(result['prediction']),
        'confidence': float(result['confidence']),
        'top_features': explanation['top_features'][:5]
    })

if __name__ == '__main__':
    app.run(debug=True)
```

#### FastAPI Example:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PRISM import PRISM

app = FastAPI()
prism = PRISM()
prism.load_model("test_results/prism_anti_overfitting_classifier.joblib",
                 "test_results/prism_anti_overfitting_word_vectorizer.joblib",
                 "test_results/prism_anti_overfitting_char_vectorizer.joblib")

class PromptRequest(BaseModel):
    text: str

@app.post("/api/analyze")
async def analyze_prompt(request: PromptRequest):
    result = prism.predict(request.text)
    explanation = prism.explain_prediction(request.text)
    
    return {
        'text': request.text,
        'is_malicious': bool(result['prediction']),
        'confidence': float(result['confidence']),
        'top_features': explanation['top_features'][:5]
    }
```

---

## Using the Scripts

### 1. train_custom.py

This script allows you to train PRISM with your custom dataset:

```bash
python train_custom.py --input llm_inputs_dataset_large.csv --text-col prompt --label-col is_malicious --output-dir ./trained_models --test-size 0.2
```

Parameters:
- `--input`: Path to your custom dataset (CSV, JSON, or JSONL)
- `--text-col`: Column name containing the text data (default: "text")
- `--label-col`: Column name containing the labels (default: "label")
- `--output-dir`: Directory to save the trained model (default: "./trained_models")
- `--test-size`: Proportion of the dataset to use for testing (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)

### 2. test_overfitting.py

This script evaluates how well your model generalizes and identifies potential overfitting:

```bash
python test_overfitting.py --model test_results/prism_anti_overfitting_classifier.joblib --word-vectorizer test_results/prism_anti_overfitting_word_vectorizer.joblib --char-vectorizer test_results/prism_anti_overfitting_char_vectorizer.joblib --dataset llm_inputs_dataset_large.csv --output-dir ./anti_overfitting_results
```

Parameters:
- `--model`: Path to the trained classifier
- `--word-vectorizer`: Path to the word vectorizer
- `--char-vectorizer`: Path to the character vectorizer
- `--dataset`: Path to the dataset for testing
- `--output-dir`: Directory to save the results
- `--cv-folds`: Number of cross-validation folds (default: 5)
- `--text-col`: Column name containing the text data (default: "text")
- `--label-col`: Column name containing the labels (default: "label")

### 3. visualize_results.py

This script generates visualizations of your model's performance:

```bash
python visualize_results.py --metrics-file test_results/anti_overfitting_20250414_002059/training_metrics.json --output-dir ./visualizations
```

Parameters:
- `--metrics-file`: Path to the JSON file containing training metrics
- `--output-dir`: Directory to save the visualizations
- `--show`: Whether to display the visualizations (add this flag to show them)

### 4. test_prism.py

This script allows you to test a trained PRISM model on individual prompts or batches:

```bash
python test_prism.py --model test_results/prism_anti_overfitting_classifier.joblib --word-vectorizer test_results/prism_anti_overfitting_word_vectorizer.joblib --char-vectorizer test_results/prism_anti_overfitting_char_vectorizer.joblib --text "Is this prompt harmful?"
```

Or test with a file:

```bash
python test_prism.py --model test_results/prism_anti_overfitting_classifier.joblib --word-vectorizer test_results/prism_anti_overfitting_word_vectorizer.joblib --char-vectorizer test_results/prism_anti_overfitting_char_vectorizer.joblib --file ./test_prompts.txt
```

Parameters:
- `--model`: Path to the trained classifier
- `--word-vectorizer`: Path to the word vectorizer
- `--char-vectorizer`: Path to the character vectorizer
- `--text`: Single prompt text for prediction
- `--file`: Path to a file containing prompts (one per line)
- `--output`: Path to save the prediction results (optional)
- `--threshold`: Classification threshold (default: 0.5)
- `--show-features`: Number of top features to display (default: 5)

---

## Example Output
```
Text: How do I bypass a content filter?
Prediction: Malicious
Confidence: 92.35% malicious, 7.65% benign
Top Features: ["bypass", "filter", "content filter", "how do i", "how"]
```

---

## Model Internals
- **Classifier**: Logistic Regression (balanced)
- **Vectorizer**: TF-IDF (up to 10,000 features, unigrams + bigrams)
- **Explainability**: Outputs most important positive/negative features

---

## Requirements
The following packages are required to run PRISM:
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
joblib>=1.0.0
matplotlib>=3.4.0
scipy>=1.7.0
pathlib>=1.0.1
```

You can install all requirements using:
```bash
pip install -r requirements.txt
```

---

## License
MIT License. Built for research and educational use.

---

## Author
Made with ❤️ by Trivulzianus. Contributions welcome!

---

## Future Plans
- ✅ API endpoint with FastAPI or Flask
- 🔬 Swap TF-IDF with semantic embeddings (e.g., SentenceTransformers)
- 🔒 Integrate with LLM gateways as a content moderation layer

