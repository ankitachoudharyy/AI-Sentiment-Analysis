"""
Flask API Backend for BERT Sentiment Analysis
Connects the trained model to the web frontend

Usage:
    python app.py
    
Then the frontend can make requests to:
    POST http://localhost:5000/api/predict
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = os.getenv("MODEL_DIR", "cardiffnlp/twitter-roberta-base-sentiment-latest")  # Use pre-trained model
MAX_TEXT_LENGTH = 512
DEVICE = 0 if torch.cuda.is_available() else -1  # Auto-detect GPU

# Label mapping for the pre-trained model
LABEL_MAP = {
    "LABEL_0": {"label": "Negative", "emoji": "🔴", "color": "#ff4444"},
    "LABEL_1": {"label": "Neutral", "emoji": "🟡", "color": "#ffaa00"},
    "LABEL_2": {"label": "Positive", "emoji": "🟢", "color": "#00ff44"}
}

# =============================================================================
# LOAD MODEL & TOKENIZER (On startup)
# =============================================================================

logger.info("🤖 Initializing BERT model...")

logger.info("🤖 Initializing BERT model...")

try:
    # Use pre-trained sentiment analysis pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=MODEL_DIR,
        device=DEVICE
    )
    
    logger.info("✅ Pre-trained model loaded successfully!")
    MODEL_LOADED = True
    
except Exception as e:
    logger.warning(f"⚠️  Could not load model: {e}")
    logger.warning("⚠️  Using fallback keyword-based sentiment...")
    MODEL_LOADED = False
    sentiment_pipeline = None


# =============================================================================
# FALLBACK: KEYWORD-BASED SENTIMENT (if model not available)
# =============================================================================

POSITIVE_WORDS = [
    "safe", "efficient", "future", "great", "amazing", "reliable", 
    "revolution", "trust", "better", "progress", "innovation", "beneficial",
    "improved", "excellent", "promising", "opportunity", "revolutionary"
]

NEGATIVE_WORDS = [
    "dangerous", "risk", "unreliable", "accident", "fear", "job loss", 
    "ethics", "doubt", "scary", "problem", "crash", "expensive", "concern",
    "threat", "vulnerability", "failure", "unemployment", "disaster"
]

def fallback_sentiment_analysis(text):
    """Simple keyword-based sentiment when model unavailable."""
    lower_text = text.lower()
    
    positive_count = sum(1 for word in POSITIVE_WORDS if word in lower_text)
    negative_count = sum(1 for word in NEGATIVE_WORDS if word in lower_text)
    
    total = positive_count + negative_count
    
    if total == 0:
        label_id = 1  # Neutral
        score = 0.5
    elif positive_count > negative_count:
        label_id = 2  # Positive
        score = min(0.95, 0.5 + (positive_count / max(1, total)) * 0.45)
    elif negative_count > positive_count:
        label_id = 0  # Negative
        score = min(0.95, 0.5 + (negative_count / max(1, total)) * 0.45)
    else:
        label_id = 1  # Neutral
        score = 0.5
    
    return {
        "label": label_id,
        "score": score,
        "method": "fallback_keyword"
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "model_loaded": MODEL_LOADED,
        "message": "AutonoSentix BERT Sentiment Analysis API"
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict_sentiment():
    """
    Predict sentiment for input text.
    
    Request format:
    {
        "text": "Your opinion about autonomous systems...",
        "category": "general"  # optional
    }
    
    Response format:
    {
        "text": "...",
        "sentiment": "Positive",
        "confidence": 87.5,
        "emoji": "🟢",
        "color": "#00ff44",
        "concerns": ["safety", "ethics"],
        "model_used": "BERT-base-uncased"
    }
    """
    
    try:
        # Validate request
        if not request.json or "text" not in request.json:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = request.json.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) > 1000:
            return jsonify({"error": "Text too long (max 1000 characters)"}), 400
        
        # Get prediction
        if MODEL_LOADED and sentiment_pipeline:
            result = sentiment_pipeline(text)
            label_key = result[0]["label"]  # e.g., "LABEL_0", "LABEL_1", "LABEL_2"
            confidence = result[0]["score"]
            method = "BERT (Pre-trained)"
        else:
            result = fallback_sentiment_analysis(text)
            label_key = f"LABEL_{result['label']}"  # Convert to LABEL format
            confidence = result["score"]
            method = "Fallback (Keywords)"
        
        # Extract concerns/topics
        concerns = extract_concerns(text)
        
        # Build response
        sentiment_info = LABEL_MAP[label_key]
        
        response = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": sentiment_info["label"],
            "emoji": sentiment_info["emoji"],
            "confidence": round(confidence * 100, 2),
            "color": sentiment_info["color"],
            "concerns": concerns,
            "polarity_score": int(label_key.split("_")[1]) - 1,  # -1 (negative) to +1 (positive)
            "model_used": method
        }
        
        logger.info(f"✅ Prediction: {response['sentiment']} ({response['confidence']}%)")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    """
    Predict sentiment for multiple texts.
    
    Request: {"texts": ["text1", "text2", ...]}
    """
    
    try:
        if not request.json or "texts" not in request.json:
            return jsonify({"error": "Missing 'texts' field"}), 400
        
        texts = request.json.get("texts", [])
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "texts must be non-empty list"}), 400
        
        results = []
        for text in texts[:50]:  # Limit to 50 texts
            if isinstance(text, str):
                # Call single predict
                result = sentiment_pipeline(text) if MODEL_LOADED else fallback_sentiment_analysis(text)
                results.append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "sentiment": LABEL_MAP[int(result[0]["label"].split("_")[1]) if MODEL_LOADED else result["label"]]["label"]
                })
        
        return jsonify({"results": results}), 200
    
    except Exception as e:
        logger.error(f"❌ Batch error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get model statistics and configuration."""
    return jsonify({
        "model": MODEL_DIR,
        "model_loaded": MODEL_LOADED,
        "device": "GPU" if DEVICE == 0 else "CPU",
        "max_text_length": MAX_TEXT_LENGTH,
        "version": "1.0.0",
        "created": "2026-04-15"
    }), 200


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_concerns(text):
    """Extract main topic concerns from text."""
    concerns = []
    
    concern_keywords = {
        "safety": ["safe", "safety", "accident", "crash", "dangerous"],
        "employment": ["job", "employment", "work", "displacement", "labor"],
        "ethics": ["ethic", "moral", "responsible", "bias", "discrimination"],
        "privacy": ["privacy", "data", "surveillance", "tracking"],
        "cost": ["cost", "expensive", "price", "afford"],
        "regulation": ["regulation", "policy", "government", "law"]
    }
    
    lower_text = text.lower()
    
    for concern, keywords in concern_keywords.items():
        if any(keyword in lower_text for keyword in keywords):
            concerns.append(concern)
    
    return concerns[:3]  # Top 3 concerns


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    logger.info(f"🚀 Starting API server on port {port}...")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )
