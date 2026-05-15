"""
BERT Sentiment Analysis Model with LoRA Fine-Tuning
For: AI-Powered Autonomous Systems Sentiment Analysis
Author: Ankita Choudhary (22CS002467)

This script trains a BERT model optimized with LoRA (Low-Rank Adaptation) 
for sentiment analysis on autonomous systems social media data.

Installation:
    pip install transformers peft datasets accelerate evaluate torch wandb
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import evaluate
import torch
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "google-bert/bert-base-uncased"  # or "answerdotai/ModernBERT-base" for 2025 variant
NUM_LABELS = 3  # positive, neutral, negative
OUTPUT_DIR = "./results"
MODEL_SAVE_DIR = "./sentiment_model"

# =============================================================================
# 1. INITIALIZE MODEL & TOKENIZER
# =============================================================================

def load_pretrained_model():
    """Load BERT model and tokenizer."""
    print("📥 Loading pre-trained BERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    return model, tokenizer


# =============================================================================
# 2. APPLY LoRA (Low-Rank Adaptation) FOR EFFICIENT FINE-TUNING
# =============================================================================

def apply_lora(model):
    """
    Apply LoRA to model - only ~0.5% parameters become trainable.
    This reduces memory usage and speeds up training dramatically.
    """
    print("⚙️  Applying LoRA configuration...")
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,                           # rank (higher = more capacity, more parameters)
        lora_alpha=32,                  # scaling factor
        lora_dropout=0.1,               # dropout for regularization
        target_modules=["query", "value"],  # BERT attention modules
        bias="none"                     # no bias in LoRA layers
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # Shows ~0.5% trainable
    
    return model


# =============================================================================
# 3. PREPARE DATASET FOR SENTIMENT ANALYSIS
# =============================================================================

def prepare_dataset(dataset_name="imdb"):
    """
    Load and prepare dataset for training.
    Options: "imdb", "amazon_reviews_multi", or custom autonomous systems data
    """
    print(f"📊 Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name)
    return dataset


def tokenize_data(examples, tokenizer):
    """Tokenize text data for BERT."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


def prepare_custom_dataset(csv_path):
    """
    Load custom dataset from CSV for autonomous systems sentiment.
    Expected columns: 'text', 'label' (0=negative, 1=neutral, 2=positive)
    """
    print(f"📂 Loading custom dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    return dataset


# =============================================================================
# 4. TRAINING CONFIGURATION (2025 OPTIMIZED)
# =============================================================================

def get_training_arguments():
    """Configure training parameters with 2025 optimizations."""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,             # Keep low for fine-tuning
        weight_decay=0.01,              # L2 regularization
        warmup_ratio=0.1,               # 10% of training as warmup
        fp16=True,                      # Mixed precision (faster on GPU)
        torch_compile=True,             # 2025: PyTorch 2.0 speed boost
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",               # Set to "wandb" for W&B logging
        logging_steps=100,
        seed=42
    )


# =============================================================================
# 5. TRAIN THE MODEL
# =============================================================================

def train_model(model, tokenizer, train_dataset, eval_dataset, training_args):
    """Fine-tune BERT with LoRA using Hugging Face Trainer."""
    
    print("\n🚀 Starting training...")
    
    # Metrics
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(pred):
        predictions, labels = pred.predictions.argmax(-1), pred.label_ids
        return accuracy.compute(predictions=predictions, references=labels)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Start training
    trainer.train()
    
    return trainer, model


# =============================================================================
# 6. SAVE TRAINED MODEL
# =============================================================================

def save_model(model, tokenizer, save_dir=MODEL_SAVE_DIR):
    """Save the trained model and tokenizer."""
    Path(save_dir).mkdir(exist_ok=True)
    
    print(f"\n💾 Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("✅ Model saved successfully!")


# =============================================================================
# 7. INFERENCE: USE TRAINED MODEL FOR PREDICTIONS
# =============================================================================

def predict_sentiment(text, model_dir=MODEL_SAVE_DIR):
    """
    Predict sentiment for new text using trained model.
    Returns: sentiment class and confidence score
    """
    print(f"🔍 Loading model from {model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Create pipeline
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # Auto-detect GPU
    )
    
    result = pipe(text)
    
    label_map = {
        "LABEL_0": "Negative ❌",
        "LABEL_1": "Neutral 😐",
        "LABEL_2": "Positive ✅"
    }
    
    return {
        "text": text,
        "sentiment": label_map.get(result[0]["label"], result[0]["label"]),
        "confidence": round(result[0]["score"] * 100, 2)
    }


# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 BERT Sentiment Analysis Training Pipeline")
    print("="*70 + "\n")
    
    # Step 1: Load model and tokenizer
    model, tokenizer = load_pretrained_model()
    
    # Step 2: Apply LoRA
    model = apply_lora(model)
    
    # Step 3: Load dataset (using IMDB by default, replace with custom data)
    dataset = prepare_dataset("imdb")
    
    # Step 4: Tokenize dataset
    print("⏳ Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_data(x, tokenizer),
        batched=True
    )
    
    # Step 5: Get training arguments
    training_args = get_training_arguments()
    
    # Step 6: Train model
    trainer, trained_model = train_model(
        model,
        tokenizer,
        tokenized_dataset["train"],
        tokenized_dataset["test"],
        training_args
    )
    
    # Step 7: Save trained model
    save_model(trained_model, tokenizer)
    
    # Step 8: Test inference
    print("\n" + "="*70)
    print("✨ Testing Inference on Sample Texts")
    print("="*70 + "\n")
    
    test_texts = [
        "Autonomous vehicles will make roads much safer for everyone.",
        "I'm concerned about job loss due to automation.",
        "Self-driving cars are interesting but need more testing."
    ]
    
    for text in test_texts:
        result = predict_sentiment(text)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']} ({result['confidence']}% confidence)\n")
    
    print("="*70)
    print("🎉 Training complete! Model ready for deployment.")
    print("="*70)
