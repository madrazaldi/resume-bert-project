import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress pandas future warnings from Hugging Face datasets
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- FULL Configuration Dictionary ---
# Includes the original configs (1, 2, 4, 5) and our improved ones.
CONFIGS = {
    "config_1_baseline": {
        "name": "config_1_baseline (BERT-base)",
        "model_name": "bert-base-uncased",
        "learning_rate": 3e-5,
        "batch_size": 32,
        "epochs": 3,
        "max_seq_length": 256,
        "warmup_steps": 500,
        "dropout_prob": 0.1,
        "weight_decay": 0.01,
        "output_dir": './results/config_1_baseline'
    },
    "config_2_long_seq": {
        "name": "config_2_long_seq (BERT-base)",
        "model_name": "bert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 4,
        "max_seq_length": 512,
        "warmup_steps": 1000,
        "dropout_prob": 0.1,
        "weight_decay": 0.01,
        "output_dir": './results/config_2_long_seq'
    },
    "config_3_enhanced": {
        "name": "config_3_enhanced (DistilBERT)",
        "model_name": "distilbert-base-uncased",
        "learning_rate": 5e-5,
        "batch_size": 32,
        "epochs": 4,
        "max_seq_length": 256,
        "warmup_steps": 100,
        "dropout_prob": 0.1,
        "weight_decay": 0.01,
        "output_dir": './results/config_3_enhanced'
    },
    "config_4_high_reg": {
        "name": "config_4_high_reg (BERT-base)",
        "model_name": "bert-base-uncased",
        "learning_rate": 2.5e-5,
        "batch_size": 32,
        "epochs": 5,
        "max_seq_length": 256,
        "warmup_steps": 750,
        "dropout_prob": 0.3,
        "weight_decay": 0.05,
        "output_dir": './results/config_4_high_reg'
    },
    "config_5_large": {
        "name": "config_5_large (BERT-large)",
        "model_name": "bert-large-uncased",
        "learning_rate": 2e-5,
        "batch_size": 8, # Reduced from 16 to fit in 12GB VRAM
        "epochs": 2,
        "max_seq_length": 256,
        "warmup_steps": 500,
        "dropout_prob": 0.1,
        "weight_decay": 0.01,
        "output_dir": './results/config_5_large'
    },
    "config_6_fixed": {
        "name": "config_6_fixed (BERT-base)",
        "model_name": "bert-base-uncased",
        "learning_rate": 4e-5,
        "batch_size": 32,
        "epochs": 3,
        "max_seq_length": 128,
        "warmup_steps": 100,
        "dropout_prob": 0.2,
        "weight_decay": 0.02,
        "output_dir": './results/config_6_fixed'
    }
}

# --- PyTorch Dataset Class ---
class ResumeDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# --- Main Pipeline Function ---
def run_pipeline(config, train_df, test_df, label_encoder):
    print("\n" + "="*80)
    print(f"🚀 Starting Pipeline for: {config['name']}")
    print("="*80)

    # 1. TOKENIZER AND DATASETS
    print(f"\n[1/4] Loading tokenizer and preparing datasets for {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    train_encodings = tokenizer(train_df['cleaned_resume'].tolist(), truncation=True, padding=True, max_length=config['max_seq_length'])
    test_encodings = tokenizer(test_df['cleaned_resume'].tolist(), truncation=True, padding=True, max_length=config['max_seq_length'])
    
    train_dataset = ResumeDataset(train_encodings, train_df['label'].tolist())
    test_dataset = ResumeDataset(test_encodings, test_df['label'].tolist())

    # 2. MODEL LOADING
    num_labels = len(label_encoder.classes_)
    print(f"\n[2/4] Loading model '{config['model_name']}' for {num_labels} classes...")
    model_config_kwargs = {'num_labels': num_labels}
    if 'distilbert' in config['model_name']:
        model_config_kwargs['dropout'] = config['dropout_prob']
        model_config_kwargs['attention_dropout'] = config['dropout_prob']
    else:
        model_config_kwargs['attention_probs_dropout_prob'] = config['dropout_prob']
        model_config_kwargs['hidden_dropout_prob'] = config['dropout_prob']
        
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], **model_config_kwargs)
    
    # 3. TRAINING
    print("\n[3/4] Setting up training arguments and initiating training...")
    
    training_args_dict = {
        "output_dir": config['output_dir'],
        "num_train_epochs": config['epochs'],
        "per_device_train_batch_size": config['batch_size'],
        "per_device_eval_batch_size": config['batch_size'],
        "warmup_steps": config['warmup_steps'],
        "weight_decay": config['weight_decay'],
        "logging_dir": './logs',
        "logging_steps": 100,
        "report_to": "none"
    }

    # Enable mixed-precision training for memory-intensive models if a GPU is available
    if "large" in config['model_name'] and torch.cuda.is_available():
        print("\n[INFO] BERT-large detected. Enabling mixed-precision training (fp16) to conserve VRAM.")
        training_args_dict['fp16'] = True
    
    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    trainer.train()
    print("🎉 Training complete!")

    # 4. EVALUATION AND PREDICTION
    print("\n[4/4] Evaluating the final model and running prediction...")
    test_predictions = trainer.predict(test_dataset)
    test_preds_labels = np.argmax(test_predictions.predictions, axis=1)
    
    print("\n--- Test Set Evaluation Results ---")
    report = classification_report(test_df['label'].tolist(), test_preds_labels, target_names=label_encoder.classes_, zero_division=0)
    print(report)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    report_path = os.path.join(config['output_dir'], 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    cm = confusion_matrix(test_df['label'].tolist(), test_preds_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {config["name"]}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = os.path.join(config['output_dir'], 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    sample_text = "Experienced java developer with a history of working in the information technology and services industry. Skilled in Spring Boot, Microservices, and SQL. Strong engineering professional with a Bachelor of Technology."
    inputs = tokenizer(sample_text, return_tensors="pt").to(trainer.model.device)
    with torch.no_grad():
        logits = trainer.model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    predicted_class_name = label_encoder.inverse_transform([predicted_class_id])[0]
    
    print(f"\n--- Sample Prediction ---")
    print(f"Sample Text: '{sample_text}'")
    print(f"Predicted Category: ✨ {predicted_class_name} ✨")
    print("\n" + "="*80)
    print(f"✅ Pipeline finished for: {config['name']}")
    print("="*80)

def main():
    try:
        train_val_df = pd.concat([pd.read_csv('train.csv'), pd.read_csv('validation.csv')], ignore_index=True)
        test_df = pd.read_csv('test.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run the improved eda_and_preprocessing.py first.")
        return

    train_val_df['cleaned_resume'] = train_val_df['cleaned_resume'].fillna('')
    test_df['cleaned_resume'] = test_df['cleaned_resume'].fillna('')

    le = LabelEncoder()
    train_val_df['label'] = le.fit_transform(train_val_df['Category'])
    test_df['label'] = le.transform(test_df['Category'])
    print(f"Labels encoded. Found {len(le.classes_)} unique categories.")

    # Run the pipeline for all specified configurations
    for config_key in CONFIGS:
        run_pipeline(CONFIGS[config_key], train_val_df, test_df, le)
    
    print("\nAll pipelines are complete.")

if __name__ == '__main__':
    main()

