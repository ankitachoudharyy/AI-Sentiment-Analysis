# 🤖 AutonoSentix - AI-Powered Autonomous Systems Sentiment Analysis

**A comprehensive AI framework for analyzing public sentiment toward autonomous systems using BERT + LoRA**

## 📋 Project Structure

```
vs code/
├── index.html                 # Main website (modern UI with Tailwind CSS)
├── about.html                 # Project overview
├── impact.html                # Societal impact analysis
├── sentiment.html             # Interactive sentiment demo
├── contact.html               # Contact page
├── blog.html                  # Blog/news section
│
├── train_bert_model.py        # BERT model training script with LoRA
├── app.py                     # Flask API backend
├── requirements.txt           # Python dependencies
├── .env                       # Environment configuration
│
├── sentiment_model/           # Trained model (generated after training)
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files
│
└── results/                   # Training results & checkpoints
```

---

## 🚀 Quick Start

### 1️⃣ Clone/Open the Project
```bash
cd "c:\Users\HP\OneDrive\Desktop\vs code"
```

### 2️⃣ Create Python Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Mac/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the BERT Model (One-time)
```bash
python train_bert_model.py
```

**What happens:**
- Downloads BERT-base-uncased (109M parameters)
- Applies LoRA (only 0.5% parameters trainable!)
- Fine-tunes on sentiment data (IMDb by default)
- Saves trained model to `./sentiment_model/`
- Runs inference test on sample texts

**Duration:** ~15-30 minutes (GPU: 5-10 min) ⚡

### 5️⃣ Start Flask Backend API
```bash
python app.py
```

**Output:**
```
🚀 Starting API server on port 5000...
```

### 6️⃣ Open Website
Open `index.html` in your browser → **Live Demo** button will now use the real trained model!

---

## 📊 Training the Model

### Default: Using IMDB Dataset
```bash
python train_bert_model.py
```

### Custom: Using Your Own Data
Create `autonomous_sentiment.csv`:
```csv
text,label
"Self-driving cars are great!",2
"I'm worried about job loss",0
"Interesting technology",1
```

Then modify `train_bert_model.py`:
```python
# Replace line in main:
dataset = prepare_custom_dataset("autonomous_sentiment.csv")
```

### Training Hyperparameters
- **Model:** BERT-base-uncased (110M → 0.5M trainable params with LoRA)
- **Epochs:** 3
- **Batch Size:** 16 (train), 64 (eval)
- **Learning Rate:** 2e-5
- **Optimizer:** AdamW with warmup
- **Mixed Precision:** FP16 (faster on GPU)
- **Device:** Auto-detects GPU or uses CPU

---

## 🔌 API Endpoints

### 1. Single Text Prediction
**Endpoint:** `POST /api/predict`

**Request:**
```json
{
  "text": "Autonomous vehicles will revolutionize transportation but raise ethical concerns",
  "category": "autonomous-vehicles"
}
```

**Response:**
```json
{
  "text": "Autonomous vehicles will revolutionize...",
  "sentiment": "Positive",
  "emoji": "🟢",
  "confidence": 87.5,
  "color": "#00ff44",
  "concerns": ["safety", "ethics"],
  "polarity_score": 1,
  "model_used": "BERT"
}
```

### 2. Batch Prediction
**Endpoint:** `POST /api/batch-predict`

**Request:**
```json
{
  "texts": ["text1", "text2", "text3"]
}
```

### 3. Model Stats
**Endpoint:** `GET /api/stats`

**Response:**
```json
{
  "model": "./sentiment_model",
  "model_loaded": true,
  "device": "GPU",
  "max_text_length": 512,
  "version": "1.0.0"
}
```

---

## 🖥️ Connecting Frontend to Backend

The `index.html` demo can be connected to the API by modifying the `analyzeSentiment()` function:

```javascript
// Replace the keyword-based analysis with API call:
async function analyzeSentiment() {
    const input = document.getElementById('userInput').value.trim()
    
    // Call backend API
    const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input })
    })
    
    const result = await response.json()
    
    // Update UI with result
    updateDemoUI(result)
}
```

---

## 📈 Model Performance

Expected accuracy on test set:
- **Accuracy:** ~90-95%
- **F1 Score:** ~0.88-0.93
- **Inference Speed:** ~50-100ms per text (CPU), ~10-20ms (GPU)

### Metrics:
- **Precision:** High confidence in predictions
- **Recall:** Captures most sentiments correctly
- **Support for Long Texts:** Truncates to 512 tokens (BERT limit)

---

## 🔧 Advanced Configuration

### Use ModernBERT (Faster 2025 Model)
```python
# In train_bert_model.py, change:
MODEL_NAME = "answerdotai/ModernBERT-base"  # 2x faster than BERT
```

### Enable GPU Acceleration
```bash
# Install CUDA-compatible PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Weights & Biases Logging
```python
# In training_args, set:
report_to="wandb"

# Then login:
wandb login
```

### Quantization for Faster Inference
```python
# Add to app.py for 8-bit inference (faster, less memory):
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

---

## 📝 Project Report Integration

This code implements the concepts from your project report:

| Chapter | Implementation |
|---------|-----------------|
| Ch. 3 - Problem Definition | `train_bert_model.py` analyzes public sentiment gaps |
| Ch. 4 - Literature Review | BERT + LoRA represents SOTA NLP approach |
| Ch. 5 - System Design | Preprocessing, tokenization, feature extraction |
| Ch. 6 - Implementation | Training module with tracking & benchmarks |
| Ch. 7 - Results/UI | `index.html` demo + API integration |
| Ch. 8 - Future Work | Deployment, scalability, multi-language support |

---

## 🛠️ Troubleshooting

### ❌ CUDA Out of Memory
```bash
# Reduce batch size in train_bert_model.py:
per_device_train_batch_size=8  # Instead of 16
```

### ❌ Model Not Found
```bash
# Run training first:
python train_bert_model.py

# Or download pre-trained:
python -c "from transformers import AutoModel; AutoModel.from_pretrained('google-bert/bert-base-uncased')"
```

### ❌ API Connection Error
```bash
# Ensure Flask is running:
python app.py

# Check if port 5000 is available:
netstat -ano | findstr :5000  # Windows
```

### ❌ Slow Inference
- Use GPU: Install `torch` with CUDA support
- Use `torch_compile=True` for 2x speedup
- Use quantization for 4x speedup

---

## 📦 Deployment Options

### Option 1: Vercel (Frontend)
```bash
# Deploy index.html as static site
vercel
```

### Option 2: Heroku (Backend API)
```bash
# Create Procfile:
echo "web: python app.py" > Procfile

heroku create your-app-name
git push heroku main
```

### Option 3: Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

---

## 📚 Resources & References

- **BERT Paper:** [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **LoRA Paper:** [LoRA: Low-Rank Adaptation of Large Models](https://arxiv.org/abs/2106.09685)
- **Hugging Face:** https://huggingface.co/
- **PEFT Library:** https://github.com/huggingface/peft

---

## 👤 Author

**Ankita Choudhary** (22CS002467)  
B.Tech CSE-AIML | Sir Padampat Singhania University

---

## 📄 License

This project is open source and available under the MIT License.

---

## ✨ Next Steps

1. ✅ Train the model: `python train_bert_model.py`
2. ✅ Start API: `python app.py`
3. ✅ Open website: `index.html`
4. ✅ Test Live Demo
5. ✅ Deploy online

**Questions?** Check the troubleshooting section or refer to the project report! 🚀
