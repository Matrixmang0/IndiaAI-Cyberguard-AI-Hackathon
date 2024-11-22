# Cybersecurity Incident Classification using T5 Transformer

## Overview
This project implements a fine-tuned T5 transformer model for classifying cybersecurity incidents into categories and sub-categories based on detailed incident descriptions. The model processes textual descriptions of cybersecurity incidents and provides hierarchical classification, supporting both single and multi-label predictions.

## Key Features
- Hierarchical classification (categories and sub-categories)
- Multi-label support for ambiguous cases
- Structured prompt-based approach
- GPU-accelerated training and inference
- Comprehensive data preprocessing pipeline
- Detailed logging and monitoring

## Model Performance
- Category Classification Accuracy: 73.99%
- Sub-category Classification Accuracy: 49.54%
- Combined Classification Accuracy: 49.14%

## Project Structure
```
.
├── README.md
├── requirements.txt
├── categories.pkl                          # Serialized categories mapping
├── subcategories.pkl                       # Serialized subcategories mapping
├── Data_Analysis_and_Preprocessing.ipynb   # Data preprocessing notebook
├── Inferencing_and_Results.ipynb          # Model evaluation notebook
├── finetuning_fT5.py                      # Model training script
├── logs/
│   └── training_log_fT5.txt               # Training logs
├── dataset/
│   ├── train.csv                          # Original training data
│   └── test.csv                           # Original test data
├── cleaned_dataset/
│   ├── cleaned_train_dataset.csv          # Preprocessed training data
│   └── cleaned_test_dataset.csv           # Preprocessed test data
└── fine_tuned_t5small/                    # Fine-tuned model files
    ├── config.json
    ├── model.safetensors
    ├── spiece.model
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── generation_config.json
    └── added_tokens.json
```

## Requirements
- Python 3.8+
- PyTorch 2.0.1+
- Transformers 4.31.0+
- pandas 2.0.3+
- numpy 1.25.2+
- scikit-learn 1.3.0+

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd cybersecurity-incident-classification
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
Run the data preprocessing notebook to clean and prepare the dataset:
```bash
jupyter execute Data_Analysis_and_Preprocessing.ipynb
```

This notebook handles:
- Missing value imputation
- Text cleaning
- Data validation
- Category/subcategory mapping

### Model Training
To train the model:
```bash
python3 finetuning_fT5.py
```

Training parameters can be modified in the script:
- Learning rate: 2e-3
- Batch size: 16
- Training epochs: 3
- Max sequence length: 1500

### Model Evaluation
Run the evaluation notebook to assess model performance:
```bash
jupyter notebook Inferencing_and_Results.ipynb
```

## Model Details

### Architecture
- Base Model: Flan-T5-small
- Input Processing: Structured prompts with crime description and available categories
- Output Format: "category: <category> subcategory: <subcategory>"

### Training Configuration
- Learning Rate: 2e-3
- Batch Size: 16
- Training Epochs: 3
- Weight Decay: 0.01
- Mixed Precision Training: Enabled
- Gradient Accumulation: Enabled

## Inference

### Sample Usage
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./fine_tuned_t5small')
tokenizer = T5Tokenizer.from_pretrained('./fine_tuned_t5small')

# Prepare input
prompt = convert_to_prompt(crime_description)
inputs = tokenizer(prompt, return_tensors="pt", max_length=1500, padding=True, truncation=True)

# Generate prediction
outputs = model.generate(**inputs, max_length=1500)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Batch Processing
The model supports batch processing for efficient inference:
```python
batch_size = 128
for i in range(0, len(inputs), batch_size):
    batch = inputs[i:i+batch_size]
    predictions = generate_text(batch)
```

## Performance Optimization
- GPU acceleration for training and inference
- Batch processing for efficient prediction
- Optimized prompt engineering
- Efficient text preprocessing pipeline

## Limitations
- Model performance may vary for previously unseen crime descriptions
- Multi-label classification has lower precision compared to single-label cases
- Resource requirements may be significant for large batch sizes

## Future Improvements
1. Experiment with larger T5 variants
2. Implement model quantization
3. Add active learning for edge cases
4. Enhance multi-label classification performance
5. Implement streaming inference

## Acknowledgments
- Google for the Flan-T5 base model
- Hugging Face for the Transformers library
