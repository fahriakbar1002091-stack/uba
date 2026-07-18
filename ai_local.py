from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

_sentiment_pipeline = None

def load_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _sentiment_pipeline

def analyze_text_local(text: str) -> dict:
    """
    Analisis sentimen text pakai DistilBERT.
    Output contoh:
    {
        "sentiment": "POSITIVE",
        "confidence": 0.9987
    }
    """
    try:
        if not text or not text.strip():
            return {
                "sentiment": "UNKNOWN",
                "confidence": 0
            }

        sentiment_model = load_sentiment_pipeline()
        result = sentiment_model(text[:512])[0]

        return {
            "sentiment": result.get("label", "UNKNOWN"),
            "confidence": float(result.get("score", 0))
        }

    except Exception as e:
        logger.exception(f"Gagal analyze_text_local: {e}")
        return {
            "sentiment": "UNKNOWN",
            "confidence": 0,
            "error": str(e)
        }