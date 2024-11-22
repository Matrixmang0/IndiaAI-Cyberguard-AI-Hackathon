"""
Crime Classification Model: Flan-T5 Fine-tuning Implementation
===========================================================

This script implements fine-tuning of the Flan-T5 model for classifying cybersecurity crimes 
into categories and sub-categories. The model is trained to understand crime descriptions 
and output both the main category and sub-category of the crime.
"""

# Import required libraries
import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset, DatasetDict
import numpy as np
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load preprocessed datasets
train_dataset = pd.read_csv('./cleaned_dataset/cleaned_train_dataset.csv')
test_dataset = pd.read_csv('./cleaned_dataset/cleaned_test_dataset.csv')

# Load predefined categories and subcategories from pickle files
with open("categories.pkl", "rb") as file:
    categories = pickle.load(file)

with open("subcategories.pkl", "rb") as file:
    subcategories = pickle.load(file)

def convert_to_prompt(row):
    """
    Converts a crime description into a structured prompt for the model.
    
    Args:
        row: A string containing the crime description
        
    Returns:
        str: Formatted prompt containing the crime description, categories, and instructions
    """
    prompt_template = '''
    You are provided with information provided by the victim about a cybersecurity crime, along with exhaustive lists of categories and sub-categories. Your task is to classify the crime into one of the categories and one of the sub-categories based on the provided information.

    Details:
    - **Crime Description**:  
    {crime}

    - **Categories**:  
    {categories}

    - **Sub-categories**:  
    {subcategories}

    **Instructions**:  
    1. Based on the provided crime description, identify the most appropriate category and sub-category from the lists provided above.  
    2. If the information is insufficient or ambiguous to assign it into a specific category or subcategory, assign up to a maximum of three sub-categories and three categories that are most relevant to the context. Separate multiple categories or sub-categories using `' or '`.  
    3. Do not provide explanations or additional text; strictly follow the output format.

    **Output Format**:  
    category: <respective_category> subcategory: <respective_subcategory>
    '''

    prompt = prompt_template.format(crime=row, categories=categories, subcategories=subcategories)
    return prompt

def preprocess_data(data):
    """
    Preprocesses the dataset by converting crime descriptions into prompts and formatting targets.
    
    Args:
        data (pd.DataFrame): DataFrame containing crime descriptions and their classifications
        
    Returns:
        pd.DataFrame: Processed DataFrame with input_text and target_text columns
    """
    df = pd.DataFrame()
    # Convert crime descriptions to structured prompts
    df['input_text'] = data['crimeaditionalinfo'].apply(convert_to_prompt)
    # Format target text with category and subcategory
    df['target_text'] = data.apply(lambda row: f"category: {row['category']} subcategory: {row['sub_category']}", axis=1)
    return df

# Preprocess train and test datasets
prepro_train_dataset = preprocess_data(train_dataset)
prepro_test_dataset = preprocess_data(test_dataset)

# Split training data into train and validation sets
from sklearn.model_selection import train_test_split
prepro_train_dataset, prepro_val_dataset = train_test_split(prepro_train_dataset, test_size=0.05, shuffle=True)

# Convert pandas DataFrames to HuggingFace Dataset format
train_dataset = Dataset.from_pandas(prepro_train_dataset)
val_dataset = Dataset.from_pandas(prepro_val_dataset)
test_dataset = Dataset.from_pandas(prepro_test_dataset)

# Create dataset dictionary for trainer
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')

def tokenizer_function(examples):
    """
    Tokenizes the input and target texts for the model.
    
    Args:
        examples: Dictionary containing input_text and target_text
        
    Returns:
        dict: Tokenized inputs and labels with padding
    """
    inputs = examples['input_text']
    targets = examples['target_text']
    # Tokenize inputs with padding and truncation
    model_inputs = tokenizer(inputs, max_length=1500, padding='max_length', truncation=True)
    # Tokenize targets with padding and truncation
    labels = tokenizer(targets, max_length=1500, padding='max_length', truncation=True)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize all datasets
tokenized_datasets = dataset_dict.map(tokenizer_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./fT5small_training_chkpts',          # Directory for saving model checkpoints
    eval_strategy='epoch',                           # Evaluation strategy
    eval_accumulation_steps=32,                       # Accumulation steps for evaluation
    save_strategy='steps',                           # Checkpoint saving strategy
    save_total_limit=1,                             # Maximum number of checkpoints to keep
    learning_rate=2e-3,                             # Learning rate
    per_device_train_batch_size=16,                  # Training batch size per device
    per_device_eval_batch_size=16,                   # Evaluation batch size per device
    num_train_epochs=3,                             # Number of training epochs
    weight_decay=0.01,                              # Weight decay for regularization
    fp16=True,                                      # Enable mixed precision training
    logging_first_step=True,                        # Log first step
    logging_dir='./tb_logs',                        # TensorBoard log directory
    logging_steps=100,                              # Log every N steps
    report_to='tensorboard'                         # Report metrics to TensorBoard
)


class LoggingCallback(TrainerCallback):
    """
    Custom callback for logging training progress and metrics.
    
    Attributes:
        log_dir (str): Directory for saving logs
        log_file (str): Name of the log file
    """
    def __init__(self, log_dir='./logs', log_file='training_log_fT5.txt'):
        super().__init__()
        self.log_dir = log_dir
        self.log_file = log_file
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.log_file)
        
        # Initialize log file with dataset information
        with open(self.log_path, 'w') as f:
            f.write("Number of Training Datapoints : "+str(len(tokenized_datasets['train']))+"\n")
            f.write("Number of Validation Datapoints : "+str(len(tokenized_datasets['validation']))+"\n\n")
            f.write("Training Logs\n\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Callback function called during training to log metrics.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control
            logs: Dictionary containing metrics to log
        """
        if logs is not None:
            with open(self.log_path, 'a') as f:
                f.write(f"Step: {state.global_step}\n")
                for key, value in logs.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    callbacks=[LoggingCallback()]
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained('./fine_tuned_t5small')
tokenizer.save_pretrained('./fine_tuned_t5small')