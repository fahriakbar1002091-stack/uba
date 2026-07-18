"""
ai_engine.py - Social Radar AI Analysis Engine v3.0
Menggunakan DistilBERT untuk:
1. Sentimen PER POSTINGAN (POSITIVE / NEGATIVE / NEUTRAL)
2. Sentimen OVERALL + BREAKDOWN (persentase)
3. Risk score AKURAT berdasarkan konten nyata
4. Personality traits (OCEAN model)
5. Summary OBJEKTIF berdasarkan fakta
6. Red flags & green flags berdasarkan konten
7. Rekomendasi & suitable roles
"""

from __future__ import annotations

import re
import math
from typing import Dict, List, Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL DistilBERT sekali saat module diimport
# ─────────────────────────────────────────────────────────────────────────────
_sentiment_pipeline = None

def _load_model():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    try:
        from transformers import pipeline as hf_pipeline
        print("⏳ [AI] Loading DistilBERT sentiment model...")
        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )
        print("✅ [AI] DistilBERT model loaded!")
    except Exception as e:
        print(f"⚠️ [AI] Gagal load DistilBERT: {e} — menggunakan rule-based fallback")
        _sentiment_pipeline = None

    return _sentiment_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVE KEYWORDS untuk risk scoring
# ─────────────────────────────────────────────────────────────────────────────
_HIGH_RISK_KEYWORDS = [
    # Kekerasan & ancaman
    "kill", "murder", "violence", "attack", "bomb", "weapon", "shoot", "stab",
    "bunuh", "ancam", "serang", "bom", "senjata",
    # Ujaran kebencian
    "hate", "racist", "racism", "nazi", "terrorist", "extremist",
    "rasis", "teroris", "ekstremis", "kafir", "sesat",
    # SARA
    "n****", "anjing", "babi", "monyet",
    # Pornografi
    "porn", "nude", "naked", "sex tape",
    # Narkoba
    "drug", "cocaine", "heroin", "meth", "ganja", "narkoba", "sabu",
    # Penipuan
    "scam", "fraud", "hack", "phishing", "penipuan", "tipu",
]

_MEDIUM_RISK_KEYWORDS = [
    "protest", "riot", "controversial", "fake", "lie", "cheat",
    "fight", "drama", "exposed", "cancel", "banned",
    "demo", "rusuh", "kontroversial", "palsu", "bohong", "curang",
    "berantem", "skandal", "tersangka",
]

_POSITIVE_KEYWORDS = [
    # Inggris
    "love", "happy", "fun", "amazing", "great", "wonderful", "beautiful",
    "success", "win", "joy", "laugh", "smile", "thank", "grateful",
    "awesome", "excited", "blessed", "perfect", "fantastic", "enjoy",
    "proud", "inspired", "good", "nice", "best", "wow", "cool",
    # Indonesia
    "senang", "bahagia", "lucu", "keren", "sukses", "menang", "terima kasih",
    "syukur", "indah", "luar biasa", "bangga", "seru", "asik", "mantap",
    "bagus", "hebat", "salut", "sayang", "cinta", "gembira", "semangat",
    "berhasil", "bersyukur", "menyenangkan", "ramah", "positif", "damai",
    "tenang", "nyaman", "sehat", "selamat", "alhamdulillah", "subhanallah",
]

# Emoji positif — dideteksi langsung dari teks asli (sebelum cleaning)
_POSITIVE_EMOJI_SIGNALS = [
    "😍", "❤️", "🔥", "✨", "🙌", "👏", "💪", "😊", "🥰", "💕",
    "🎉", "🥳", "😁", "🤩", "👍", "💯", "🌟", "⭐", "😄", "😃",
    "🙏", "💚", "💙", "💛", "🧡", "💜", "🤗", "😂", "🥹", "🫶",
]

_NEGATIVE_EMOJI_SIGNALS = [
    "😡", "🤬", "💔", "😢", "😭", "😤", "🤮", "🖕", "😒", "😞",
    "😔", "😟", "🥺", "😠", "👎", "💀", "😩", "😫", "🤦",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: analisis sentimen satu teks
# ─────────────────────────────────────────────────────────────────────────────
def _analyze_single_sentiment(text: str) -> Dict[str, Any]:
    """
    Mengembalikan {"label": "POSITIVE"/"NEGATIVE"/"NEUTRAL", "score": float}
    Pipeline:
    1. Deteksi emoji kuat → langsung tentukan
    2. Deteksi bahasa (ID/EN)
    3. Bahasa Indonesia → rule-based (lebih akurat)
    4. Bahasa Inggris → DistilBERT dengan threshold 0.72
    5. Fallback rule-based
    """
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.5}

    # ── Step 1: hitung sinyal emoji dari teks asli ────────────────────────
    pos_emoji = sum(1 for e in _POSITIVE_EMOJI_SIGNALS if e in text)
    neg_emoji = sum(1 for e in _NEGATIVE_EMOJI_SIGNALS if e in text)
    emoji_signal = pos_emoji - neg_emoji  # positif = lebih banyak emoji positif

    # ── Step 2: bersihkan teks untuk model (tapi simpan teks asli) ────────
    clean_for_model = re.sub(r"http\S+", "", text)
    # Jangan hapus emoji dulu — pakai untuk deteksi bahasa
    # Hapus karakter non-kata tapi biarkan huruf latin dan angka
    clean_latin = re.sub(r"[^\w\s.,!?']", " ", clean_for_model).strip()
    words = clean_latin.split()

    # ── Step 3: deteksi bahasa Indonesia ─────────────────────────────────
    _ID_INDICATORS = [
        "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "tidak",
        "ada", "ke", "dari", "nya", "aku", "gue", "gw", "kamu", "lo",
        "aja", "sih", "banget", "dong", "deh", "yuk", "udah", "udah",
        "sama", "kalau", "gimana", "gimana", "makasih", "nggak", "ngga",
    ]
    text_lower = text.lower()
    id_count = sum(1 for kw in _ID_INDICATORS if f" {kw} " in f" {text_lower} ")
    is_indonesian = id_count >= 1 or len(words) < 4

    # ── Step 4: jika Indonesia ATAU teks pendek → rule-based yang kaya ───
    if is_indonesian or len(words) < 4:
        rb = _rule_based_sentiment(text)
        # Sesuaikan dengan sinyal emoji
        if emoji_signal >= 2 and rb["label"] == "NEUTRAL":
            return {"label": "POSITIVE", "score": min(0.65 + emoji_signal * 0.05, 0.92)}
        if emoji_signal <= -2 and rb["label"] == "NEUTRAL":
            return {"label": "NEGATIVE", "score": min(0.65 + abs(emoji_signal) * 0.05, 0.92)}
        # Kuatkan score jika ada emoji pendukung
        if emoji_signal > 0 and rb["label"] == "POSITIVE":
            return {"label": "POSITIVE", "score": min(rb["score"] + 0.05, 0.97)}
        if emoji_signal < 0 and rb["label"] == "NEGATIVE":
            return {"label": "NEGATIVE", "score": min(rb["score"] + 0.05, 0.97)}
        return rb

    # ── Step 5: bahasa Inggris → DistilBERT dengan threshold lebih ketat ─
    model = _load_model()
    if model:
        try:
            # Pakai teks bersih tanpa emoji untuk model
            clean_for_distilbert = re.sub(r"[^\w\s.,!?']", " ", clean_for_model).strip()
            result = model(clean_for_distilbert[:512])[0]
            label = result["label"].upper()
            score = round(float(result["score"]), 4)

            # Threshold 0.72 — lebih ketat, lebih objektif
            if score < 0.72:
                # Cek emoji untuk tilt NEUTRAL ke salah satu arah
                if emoji_signal >= 2:
                    return {"label": "POSITIVE", "score": 0.65}
                if emoji_signal <= -2:
                    return {"label": "NEGATIVE", "score": 0.65}
                return {"label": "NEUTRAL", "score": round(score, 4)}

            # Kuatkan dengan emoji signal
            if emoji_signal > 0 and label == "POSITIVE":
                score = min(score + 0.03, 0.99)
            elif emoji_signal < 0 and label == "NEGATIVE":
                score = min(score + 0.03, 0.99)
            elif emoji_signal > 0 and label == "NEGATIVE":
                # Emoji positif tapi model bilang negatif → turunkan confidence
                if score < 0.85:
                    return {"label": "NEUTRAL", "score": 0.52}
            elif emoji_signal < 0 and label == "POSITIVE":
                if score < 0.85:
                    return {"label": "NEUTRAL", "score": 0.52}

            return {"label": label, "score": round(score, 4)}
        except Exception as e:
            print(f"   ⚠️ DistilBERT error: {e}")

    # ── Fallback ─────────────────────────────────────────────────────────
    rb = _rule_based_sentiment(text)
    if emoji_signal >= 2 and rb["label"] == "NEUTRAL":
        return {"label": "POSITIVE", "score": 0.65}
    if emoji_signal <= -2 and rb["label"] == "NEUTRAL":
        return {"label": "NEGATIVE", "score": 0.65}
    return rb


def _rule_based_sentiment(text: str) -> Dict[str, Any]:
    """Keyword-based sentiment dengan bobot berbeda untuk keyword kuat vs lemah."""
    lower = text.lower()

    # Keyword negatif dari _HIGH_RISK dan _MEDIUM_RISK sudah ada
    neg_hits_high   = sum(1 for kw in _HIGH_RISK_KEYWORDS   if kw in lower)
    neg_hits_medium = sum(1 for kw in _MEDIUM_RISK_KEYWORDS if kw in lower)
    pos_hits        = sum(1 for kw in _POSITIVE_KEYWORDS    if kw in lower)

    # Keyword kuat punya bobot 3x
    neg_score = (neg_hits_high * 3) + neg_hits_medium
    pos_score = pos_hits

    if pos_score > neg_score:
        conf = min(0.60 + (pos_score * 0.06), 0.95)
        return {"label": "POSITIVE", "score": round(conf, 4)}
    elif neg_score > pos_score:
        conf = min(0.60 + (neg_score * 0.06), 0.95)
        return {"label": "NEGATIVE", "score": round(conf, 4)}
    else:
        return {"label": "NEUTRAL", "score": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: analisis isi konten satu post
# ─────────────────────────────────────────────────────────────────────────────
_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "olahraga":      ["football", "soccer", "sport", "goal", "match", "game", "win", "champion",
                      "bola", "sepak", "olahraga", "gol", "juara", "pertandingan"],
    "motivasi":      ["believe", "dream", "work hard", "never give up", "inspire", "legend",
                      "percaya", "mimpi", "semangat", "inspirasi", "berjuang"],
    "keluarga":      ["family", "son", "daughter", "wife", "mother", "father", "love",
                      "keluarga", "anak", "istri", "ibu", "ayah", "cinta"],
    "promosi":       ["check out", "link in bio", "buy", "discount", "offer", "sponsor",
                      "promo", "beli", "diskon", "penawaran"],
    "kesehatan":     ["workout", "training", "fitness", "gym", "health", "strong",
                      "latihan", "kebugaran", "sehat", "kuat"],
    "perjalanan":    ["travel", "trip", "city", "country", "visit", "tour",
                      "perjalanan", "kota", "negara", "wisata"],
    "hiburan":       ["funny", "fun", "comedy", "laugh", "entertainment",
                      "lucu", "komedi", "hiburan", "seru"],
    "nasionalisme":  ["portugal", "brazil", "indonesia", "country", "flag", "nation",
                      "bangsa", "negara", "bendera", "tanah air"],
}


def _analyze_post_content(text: str, activity: Dict) -> Dict[str, Any]:
    """
    Analisis lengkap isi konten satu postingan:
    - caption_sentiment   : sentimen dari teks caption
    - ai_description      : deskripsi natural language tentang isi konten
    - ai_sentiment        : sentimen dari combined_text (caption + OCR + transcript)
    - sentiment_comparison: perbandingan keduanya
    - topics              : daftar topik yang terdeteksi
    - content_type        : tipe media
    - engagement_level    : level engagement
    """
    if not text:
        text = activity.get("content_description") or ""

    # ── Input teks terbaik ────────────────────────────────────────────────
    combined   = activity.get("combined_text") or text
    caption    = activity.get("text") or activity.get("caption_text") or text
    ocr_text   = activity.get("ocr_text") or ""
    transcript = activity.get("audio_transcript") or ""
    hashtags   = activity.get("hashtags") or []
    mentions   = activity.get("mentions") or []
    media_type = activity.get("content_type") or "video"
    lower      = combined.lower()

    # ── Sentimen dari CAPTION ─────────────────────────────────────────────
    caption_sent = _analyze_single_sentiment(caption) if caption.strip() else {"label": "NEUTRAL", "score": 0.5}

    # ── Sentimen dari COMBINED TEXT (lebih kaya) ──────────────────────────
    ai_sent = _analyze_single_sentiment(combined) if combined.strip() else caption_sent

    # ── Deteksi topik ─────────────────────────────────────────────────────
    topic_scores: Dict[str, int] = {}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > 0:
            topic_scores[topic] = hits

    top_topics = sorted(topic_scores, key=lambda k: topic_scores[k], reverse=True)[:3]
    if not top_topics and hashtags:
        top_topics = hashtags[:3]
    main_topic = top_topics[0] if top_topics else "umum"

    # ── Engagement level ──────────────────────────────────────────────────
    metrics   = activity.get("metrics", {})
    likes     = metrics.get("likes", 0)
    comments  = metrics.get("comments", 0)
    plays     = metrics.get("plays", 0)
    total_eng = likes + (comments * 5) + (plays // 100 if plays else 0)

    if total_eng >= 5_000_000:
        eng_level = "VIRAL"
    elif total_eng >= 1_000_000:
        eng_level = "SANGAT TINGGI"
    elif total_eng >= 100_000:
        eng_level = "TINGGI"
    elif total_eng >= 10_000:
        eng_level = "SEDANG"
    else:
        eng_level = "RENDAH"

    # ── AI Description — deskripsi kontekstual & natural ─────────────────
    # Dibangun dari inferensi hashtag + caption + topik + OCR + transcript.
    # Tidak mengarang visual yang tidak terlihat, tapi juga tidak sekadar
    # mengulang caption — lebih natural dengan konteks yang terdeteksi.

    clean_caption = re.sub(r"[#@]\w+", "", caption).strip()
    clean_caption = re.sub(r"\s+", " ", clean_caption).strip()

    # Deteksi konteks visual dari hashtag
    all_tags_lower = [h.lower() for h in hashtags]
    all_text_lower = (caption + " " + " ".join(hashtags)).lower()

    # Peta konteks → kalimat pembuka deskriptif
    _CONTEXT_MAP = [
        # Alam & outdoor
        (["gunung", "mountain", "hiking", "pendaki", "summit", "puncak",
          "ranukumbolo", "rinjani", "semeru", "sumbing", "merbabu", "prau"],
         "Konten ini menampilkan aktivitas di alam pegunungan"),
        (["pantai", "beach", "laut", "ocean", "sea", "sunset", "sunrise"],
         "Konten ini menampilkan pemandangan alam pantai atau lautan"),
        (["alam", "nature", "hutan", "forest", "air terjun", "waterfall", "sawah"],
         "Konten ini menampilkan keindahan alam terbuka"),
        # Food & kuliner
        (["food", "makanan", "kuliner", "makan", "recipe", "masak", "cooking",
          "restaurant", "cafe", "kopi", "coffee", "minuman"],
         "Konten ini berkaitan dengan kuliner atau makanan"),
        # Hiburan & komedi
        (["comedy", "komedi", "lucu", "funny", "humor", "meme", "parody",
          "sketch", "prank", "viral"],
         "Konten ini berupa hiburan atau komedi"),
        # Olahraga
        (["gym", "workout", "fitness", "sport", "olahraga", "latihan",
          "football", "soccer", "basketball", "bola"],
         "Konten ini berkaitan dengan olahraga atau kebugaran"),
        # Musik & dance
        (["musik", "music", "dance", "nyanyi", "singing", "lagu", "song",
          "cover", "dance challenge"],
         "Konten ini menampilkan musik atau tari"),
        # Travel & lifestyle
        (["travel", "wisata", "liburan", "vacation", "trip", "explore",
          "vlog", "jalan-jalan"],
         "Konten ini berupa vlog perjalanan atau lifestyle"),
        # Motivasi & edukasi
        (["motivasi", "motivation", "inspire", "semangat", "tips", "tutorial",
          "belajar", "edukasi", "education", "ilmu"],
         "Konten ini bersifat edukatif atau motivasional"),
        # Keluarga & personal
        (["keluarga", "family", "anak", "baby", "sahabat", "teman", "friend",
          "couple", "selfie", "potrait"],
         "Konten ini menampilkan momen personal atau keluarga"),
        # Teknologi & AI (Twitter-specific)
        (["ai", "artificial intelligence", "machine learning", "gpt", "robot",
          "tesla", "spacex", "rocket", "space", "technology", "tech", "software",
          "engineering", "programming", "code", "startup", "silicon valley"],
         "Konten ini membahas topik teknologi, AI, atau inovasi"),
        # Politik & bisnis (Twitter-specific)
        (["politics", "government", "president", "election", "vote", "policy",
          "economy", "business", "market", "stock", "bitcoin", "crypto",
          "trump", "america", "freedom", "free speech", "democracy"],
         "Konten ini membahas topik politik, bisnis, atau isu sosial"),
        # Sains & lingkungan
        (["science", "climate", "environment", "planet", "universe", "nasa",
          "research", "study", "data", "facts"],
         "Konten ini membahas topik sains atau lingkungan"),
    ]

    # Cari konteks yang paling cocok
    opening = None
    for context_tags, context_desc in _CONTEXT_MAP:
        if any(tag in all_text_lower for tag in context_tags):
            opening = context_desc
            break

    if not opening:
        type_label = {"video": "Video", "image": "Foto", "carousel": "Foto carousel"}.get(media_type, "Konten")
        opening = f"{type_label} ini"

    # Susun deskripsi — jauh lebih kaya dan menjelaskan
    desc_sentences: List[str] = []

    # 1. Kalimat pembuka kontekstual
    desc_sentences.append(opening)

    # 2. Apa yang dilakukan/disampaikan user
    # Bersihkan URL dari caption untuk deskripsi
    caption_no_url = re.sub(r"https?://\S+", "", clean_caption).strip()
    caption_no_url = re.sub(r"\s+", " ", caption_no_url).strip()

    if caption_no_url and len(caption_no_url) > 3:
        if len(caption_no_url) > 50:
            desc_sentences.append(f'User mengungkapkan: "{caption_no_url[:200]}"')
        else:
            desc_sentences.append(f'dengan pesan: "{caption_no_url}"')
    elif not caption_no_url and clean_caption:
        # Caption hanya berisi URL — coba baca context dari OCR
        if not ocr_text:
            desc_sentences.append("Konten ini dibagikan tanpa teks caption (hanya tautan)")
    elif clean_caption:
        desc_sentences.append(f'dengan pesan: "{clean_caption[:100]}"')

    # 3. Detail topik — semua topik yang terdeteksi
    if top_topics:
        if len(top_topics) == 1:
            desc_sentences.append(f"Konten ini berfokus pada topik {top_topics[0]}")
        else:
            desc_sentences.append(f"Konten ini mencakup topik: {', '.join(top_topics)}")

    # 4. Konteks dari mentions
    if mentions:
        if len(mentions) == 1:
            desc_sentences.append(f"Video ini melibatkan atau menyebut akun @{mentions[0]}")
        else:
            m_str = ", ".join(f"@{m}" for m in mentions[:3])
            desc_sentences.append(f"Video ini melibatkan beberapa akun: {m_str}")

    # 5. OCR — teks yang terbaca dari konten
    if ocr_text and len(ocr_text.strip()) > 5:
        clean_ocr = ocr_text.strip()[:150]
        desc_sentences.append(f'Teks yang terdeteksi dalam visual konten: "{clean_ocr}"')

    # 6. Transcript audio — narasi yang diucapkan
    if transcript and len(transcript.strip()) > 10:
        clean_tr = transcript.strip()[:200]
        desc_sentences.append(f'Narasi yang diucapkan dalam video: "{clean_tr}"')

    # 7. Hashtag — konteks tambahan
    if hashtags:
        tag_str = " ".join(f"#{h}" for h in hashtags[:6])
        desc_sentences.append(f"Hashtag yang digunakan: {tag_str}")

    # 8. Engagement context
    if eng_level in ("VIRAL", "SANGAT TINGGI", "TINGGI"):
        desc_sentences.append(f"Konten ini mendapat engagement {eng_level.lower()} dari audiens")

    # 9. Sentimen dan nuansa
    sent_phrases = {
        "POSITIVE": "Secara keseluruhan, konten ini bersentimen positif — mengekspresikan semangat, kebahagiaan, atau hal yang menyenangkan",
        "NEGATIVE": "Secara keseluruhan, konten ini bersentimen negatif atau mengandung kritik, keluhan, maupun ekspresi yang kuat",
        "NEUTRAL":  "Secara keseluruhan, konten ini bersentimen netral — bersifat informatif atau deskriptif tanpa muatan emosi yang kuat",
    }
    desc_sentences.append(sent_phrases.get(ai_sent["label"], sent_phrases["NEUTRAL"]))

    ai_description = ". ".join(desc_sentences).rstrip(".") + "."

    # ── Sentiment comparison ──────────────────────────────────────────────
    cap_label = caption_sent["label"]
    ai_label  = ai_sent["label"]
    cap_score = caption_sent["score"]
    ai_score  = ai_sent["score"]

    # Normalise score ke 0-1 untuk perbandingan
    def _norm(label: str, score: float) -> float:
        if label == "POSITIVE":
            return score
        elif label == "NEGATIVE":
            return 1.0 - score
        return 0.5

    diff = round(abs(_norm(cap_label, cap_score) - _norm(ai_label, ai_score)), 4)

    sentiment_comparison = {
        "caption": cap_label,
        "ai":      ai_label,
        "match":   cap_label == ai_label,
        "difference": diff,
        "note": (
            "Sentimen caption dan AI konten sama" if cap_label == ai_label
            else f"Caption {cap_label.lower()} tapi AI mendeteksi {ai_label.lower()} dari keseluruhan konten"
        )
    }

    return {
        "caption_sentiment":    caption_sent,
        "ai_description":       ai_description,
        "ai_sentiment":         ai_sent,
        "sentiment_comparison": sentiment_comparison,
        "topics":               top_topics,
        "topic":                main_topic,       # backward compat
        "content_type":         media_type,
        "engagement_level":     eng_level,
        "hashtags":             hashtags,
        "has_mentions":         len(mentions) > 0,
        "word_count":           len(caption.split()) if caption else 0,
        # legacy fields untuk backward compat
        "content_summary":      clean_caption[:200] if clean_caption else (hashtags[0] if hashtags else ""),
        "key_themes":           top_topics,
        "has_hashtags":         len(hashtags) > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: hitung risk score dari activities
# ─────────────────────────────────────────────────────────────────────────────
def _compute_risk_score(
    activities: List[Dict],
    sentiment_breakdown: Dict[str, float]
) -> int:
    """
    Risk score 0–100 berdasarkan:
    - Persentase sentimen NEGATIVE
    - Kehadiran high-risk keywords
    - Kehadiran medium-risk keywords
    """
    base_risk = 0

    # Komponen 1: dari sentimen negatif (max 50 poin)
    neg_pct = sentiment_breakdown.get("NEGATIVE", 0)
    base_risk += int(neg_pct * 0.5)   # 100% negatif → +50

    # Komponen 2: scan keywords di semua teks (max 50 poin)
    all_text = " ".join(
        (
            a.get("combined_text") or   # ← pakai combined_text jika ada
            a.get("text") or
            a.get("content_text") or
            ""
        ).lower()
        for a in activities
    )

    high_hits   = sum(1 for kw in _HIGH_RISK_KEYWORDS   if kw in all_text)
    medium_hits = sum(1 for kw in _MEDIUM_RISK_KEYWORDS if kw in all_text)

    keyword_risk = min((high_hits * 15) + (medium_hits * 5), 50)
    base_risk += keyword_risk

    return min(max(base_risk, 0), 100)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: hitung personality OCEAN dari konten
# ─────────────────────────────────────────────────────────────────────────────
def _compute_personality(
    activities: List[Dict],
    sentiment_breakdown: Dict[str, float],
    profile: Dict
) -> Dict[str, int]:
    """
    Estimasi OCEAN (Big Five) berdasarkan pola konten.
    Nilai 0–100.
    """
    all_text = " ".join(
        (a.get("text") or "").lower() for a in activities
    )
    total_posts = max(len(activities), 1)

    # Hitung engagement rate rata-rata
    avg_engagement = 0
    if activities:
        total_eng = sum(
            a.get("metrics", {}).get("likes", 0) +
            a.get("metrics", {}).get("comments", 0)
            for a in activities
        )
        avg_engagement = total_eng / total_posts

    followers = profile.get("followers", 0)

    # Openness: penggunaan hashtag beragam, topik variatif
    all_hashtags = []
    for a in activities:
        all_hashtags.extend(a.get("hashtags") or [])
    unique_hashtags = len(set(h.lower() for h in all_hashtags))
    openness = min(40 + (unique_hashtags * 3), 95)

    # Conscientiousness: konsistensi posting (proxy: jumlah postingan)
    conscientiousness = min(30 + (total_posts * 4), 90)

    # Extraversion: followers banyak + engagement tinggi
    if followers > 1_000_000:
        extraversion = min(70 + int(math.log10(followers) * 3), 95)
    elif followers > 10_000:
        extraversion = min(50 + int(math.log10(followers) * 5), 90)
    else:
        extraversion = 40

    # Agreeableness: sentimen positif tinggi
    pos_pct = sentiment_breakdown.get("POSITIVE", 50)
    agreeableness = min(30 + int(pos_pct * 0.6), 95)

    # Neuroticism: sentimen negatif tinggi → neuroticism tinggi
    neg_pct = sentiment_breakdown.get("NEGATIVE", 0)
    neuroticism = min(10 + int(neg_pct * 0.7), 80)

    return {
        "openness": int(openness),
        "conscientiousness": int(conscientiousness),
        "extraversion": int(extraversion),
        "agreeableness": int(agreeableness),
        "neuroticism": int(neuroticism)
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: buat flags dan rekomendasi
# ─────────────────────────────────────────────────────────────────────────────
def _compute_flags(
    activities: List[Dict],
    sentiment_breakdown: Dict[str, float],
    risk_score: int,
    profile: Dict
) -> Dict[str, Any]:
    red_flags: List[str] = []
    green_flags: List[str] = []

    pos_pct = sentiment_breakdown.get("POSITIVE", 0)
    neg_pct = sentiment_breakdown.get("NEGATIVE", 0)
    followers = profile.get("followers", 0)
    verified = profile.get("verified", False)

    all_text = " ".join(
        (a.get("text") or "").lower() for a in activities
    )

    # ── Green flags ───────────────────────────────────────────────────────
    if pos_pct >= 60:
        green_flags.append(f"✅ Sentimen positif dominan ({pos_pct:.0f}%)")
    if risk_score <= 20:
        green_flags.append("✅ Risk score rendah — konten aman")
    if followers >= 1_000_000:
        green_flags.append(f"✅ Pengaruh besar ({followers:,} followers)")
    if verified:
        green_flags.append("✅ Akun terverifikasi resmi")
    if neg_pct == 0 and activities:
        green_flags.append("✅ Tidak ada konten negatif terdeteksi")

    # ── Red flags ─────────────────────────────────────────────────────────
    if neg_pct >= 40:
        red_flags.append(f"🚨 Sentimen negatif tinggi ({neg_pct:.0f}%)")
    if risk_score >= 70:
        red_flags.append(f"🚨 Risk score tinggi ({risk_score}/100)")
    elif risk_score >= 40:
        red_flags.append(f"⚠️ Risk score sedang ({risk_score}/100)")

    high_hits = [kw for kw in _HIGH_RISK_KEYWORDS if kw in all_text]
    if high_hits:
        red_flags.append(f"🚨 Kata berisiko tinggi terdeteksi: {', '.join(set(high_hits[:3]))}")

    medium_hits = [kw for kw in _MEDIUM_RISK_KEYWORDS if kw in all_text]
    if medium_hits:
        red_flags.append(f"⚠️ Kata kontroversial terdeteksi: {', '.join(set(medium_hits[:3]))}")

    # ── Rekomendasi ───────────────────────────────────────────────────────
    if risk_score <= 25:
        recommendation = "SANGAT DIREKOMENDASIKAN"
    elif risk_score <= 45:
        recommendation = "DIREKOMENDASIKAN"
    elif risk_score <= 65:
        recommendation = "PERTIMBANGKAN DENGAN HATI-HATI"
    else:
        recommendation = "TIDAK DIREKOMENDASIKAN"

    # ── Suitable roles ────────────────────────────────────────────────────
    suitable_roles: List[str] = []
    pos_pct_val = sentiment_breakdown.get("POSITIVE", 0)

    if pos_pct_val >= 60 and risk_score <= 40:
        suitable_roles.extend(["BRAND AMBASSADOR", "MARKETING"])
    if followers >= 100_000 and risk_score <= 50:
        suitable_roles.extend(["CREATIVE", "CONTENT CREATOR"])
    if verified and risk_score <= 30:
        suitable_roles.append("PUBLIC RELATIONS")
    if risk_score <= 20:
        suitable_roles.append("CORPORATE SPOKESPERSON")
    if not suitable_roles:
        suitable_roles.append("INTERNAL ROLE ONLY")

    # Deduplicate
    suitable_roles = list(dict.fromkeys(suitable_roles))

    return {
        "red_flags": red_flags,
        "green_flags": green_flags,
        "recommendation": recommendation,
        "suitable_roles": suitable_roles
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: buat summary objektif
# ─────────────────────────────────────────────────────────────────────────────
def _build_summary(
    username: str,
    platform: str,
    profile: Dict,
    activities: List[Dict],
    sentiment_breakdown: Dict[str, float],
    risk_score: int,
    risk_level: str
) -> str:
    """
    Summary berdasarkan FAKTA:
    - Nama, followers, platform
    - Jumlah postingan dianalisis
    - Sentimen dominan & persentase
    - Risk score & level
    - Topik/hashtag dominan
    """
    full_name  = profile.get("full_name") or username
    followers  = profile.get("followers", 0)
    verified   = profile.get("verified", False)
    total_p    = len(activities)

    pos_pct  = sentiment_breakdown.get("POSITIVE", 0)
    neg_pct  = sentiment_breakdown.get("NEGATIVE", 0)
    neu_pct  = sentiment_breakdown.get("NEUTRAL",  0)

    # Sentimen dominan
    if pos_pct >= neg_pct and pos_pct >= neu_pct:
        dominant_sentiment = "positif"
        dom_pct = pos_pct
    elif neg_pct >= pos_pct and neg_pct >= neu_pct:
        dominant_sentiment = "negatif"
        dom_pct = neg_pct
    else:
        dominant_sentiment = "netral"
        dom_pct = neu_pct

    # Kumpulkan topik / hashtag dominan
    all_hashtags: List[str] = []
    for a in activities:
        all_hashtags.extend(a.get("hashtags") or [])
    top_tags = []
    if all_hashtags:
        from collections import Counter
        counted = Counter(h.lower() for h in all_hashtags)
        top_tags = [f"#{tag}" for tag, _ in counted.most_common(3)]

    # Kumpulkan topik konten dominan dari content_analysis
    all_topics: List[str] = []
    engagement_levels: Dict[str, int] = {}
    viral_count = 0
    for a in activities:
        ca = a.get("content_analysis") or {}
        topic = ca.get("topic")
        if topic and topic != "umum":
            all_topics.append(topic)
        eng = ca.get("engagement_level", "")
        if eng:
            engagement_levels[eng] = engagement_levels.get(eng, 0) + 1
        if eng in ("VIRAL", "SANGAT TINGGI"):
            viral_count += 1

    dominant_topic = ""
    if all_topics:
        from collections import Counter
        topic_count = Counter(all_topics)
        dominant_topic = topic_count.most_common(1)[0][0]

    # Total engagement
    total_likes_all = sum(
        (a.get("metrics") or {}).get("likes", 0) for a in activities
    )
    avg_likes = int(total_likes_all / max(len(activities), 1))

    # Followers formatting
    if followers >= 1_000_000:
        followers_str = f"{followers/1_000_000:.1f} juta"
    elif followers >= 1_000:
        followers_str = f"{followers/1_000:.1f} ribu"
    else:
        followers_str = str(followers)

    verified_str = " (akun terverifikasi)" if verified else ""

    tag_str = f" Topik dominan: {', '.join(top_tags)}." if top_tags else ""

    # Bangun insight tambahan
    extra_insights = []
    if dominant_topic:
        extra_insights.append(f"Konten didominasi topik {dominant_topic}")
    if viral_count > 0:
        extra_insights.append(f"{viral_count} postingan mencapai engagement viral/sangat tinggi")
    if avg_likes > 0:
        if avg_likes >= 1_000_000:
            extra_insights.append(f"rata-rata {avg_likes/1_000_000:.1f}M likes per postingan")
        elif avg_likes >= 1_000:
            extra_insights.append(f"rata-rata {avg_likes/1_000:.0f}K likes per postingan")
    if top_tags:
        extra_insights.append(f"hashtag paling sering: {', '.join(top_tags[:2])}")

    extra_str = ". ".join(extra_insights)
    if extra_str:
        extra_str = " " + extra_str + "."

    summary = (
        f"@{username} ({full_name}) adalah pengguna {platform.upper()}"
        f"{verified_str} dengan {followers_str} followers. "
        f"Dari {total_p} postingan terakhir yang dianalisis, "
        f"{dom_pct:.0f}% konten bersentimen {dominant_sentiment} "
        f"({pos_pct:.0f}% positif, {neg_pct:.0f}% negatif, {neu_pct:.0f}% netral)."
        f"{extra_str}"
        f" Risk score: {risk_score}/100 ({risk_level})."
    )

    return summary.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI ANALISIS REPOST — dipanggil dari main.py
# ─────────────────────────────────────────────────────────────────────────────
def analyze_reposts(
    username: str,
    reposts: List[Dict]
) -> List[Dict]:
    """
    Analisis sentimen + buat deskripsi mudah dipahami untuk setiap repost.
    """
    if not reposts:
        return []

    _load_model()
    enriched: List[Dict] = []

    for r in reposts:
        caption   = r.get("original_caption") or ""
        hashtags  = r.get("hashtags") or []
        author    = r.get("original_author") or "kreator"
        metrics   = r.get("metrics") or {}
        likes     = metrics.get("likes", 0)
        plays     = metrics.get("plays", 0)

        # ── Sentimen berdasarkan caption ─────────────────────────────────
        sent = _analyze_single_sentiment(caption) if caption else {"label": "NEUTRAL", "score": 0.5}

        # ── Deteksi topik ─────────────────────────────────────────────────
        lower = caption.lower()
        topic_scores: Dict[str, int] = {}
        for topic, keywords in _TOPIC_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in lower)
            if hits > 0:
                topic_scores[topic] = hits
        main_topic = max(topic_scores, key=lambda k: topic_scores[k]) if topic_scores else "umum"

        # ── Buat deskripsi visual yang lebih natural ─────────────────────
        clean_caption = re.sub(r"[#@]\w+", "", caption).strip()
        clean_caption = re.sub(r"\s+", " ", clean_caption).strip()
        all_tags_lower = [h.lower() for h in hashtags]
        all_text_lower = (caption + " " + " ".join(hashtags)).lower()

        # Peta konteks → kalimat deskriptif (sama dengan _analyze_post_content)
        _REPOST_CONTEXT_MAP = [
            (["gunung", "mountain", "hiking", "pendaki", "summit", "puncak",
              "ranukumbolo", "rinjani", "semeru", "sumbing", "merbabu"],
             f"Video dari @{author} menampilkan aktivitas pendakian atau wisata pegunungan"),
            (["pantai", "beach", "laut", "ocean", "sunset", "sunrise"],
             f"Video dari @{author} menampilkan pemandangan pantai atau alam lautan"),
            (["alam", "nature", "hutan", "forest", "air terjun", "waterfall"],
             f"Video dari @{author} menampilkan keindahan alam terbuka"),
            (["food", "makanan", "kuliner", "makan", "masak", "cooking",
              "restaurant", "cafe", "kopi", "coffee"],
             f"Video dari @{author} berkaitan dengan kuliner atau makanan"),
            (["comedy", "komedi", "lucu", "funny", "humor", "meme", "viral", "prank"],
             f"Video dari @{author} berisi konten hiburan atau komedi"),
            (["gym", "workout", "fitness", "sport", "olahraga", "latihan",
              "football", "soccer", "basketball"],
             f"Video dari @{author} berkaitan dengan olahraga atau kebugaran"),
            (["musik", "music", "dance", "nyanyi", "singing", "lagu", "cover"],
             f"Video dari @{author} menampilkan musik atau tarian"),
            (["travel", "wisata", "liburan", "vacation", "trip", "explore", "vlog"],
             f"Video dari @{author} berupa vlog perjalanan atau eksplorasi"),
            (["motivasi", "motivation", "inspire", "tips", "tutorial", "belajar", "edukasi"],
             f"Video dari @{author} bersifat edukatif atau motivasional"),
            (["motor", "mobil", "otomotif", "racing", "drag", "modifikasi", "knalpot"],
             f"Video dari @{author} menampilkan konten otomotif atau kendaraan"),
            (["gaming", "game", "gamer", "esports", "main game", "gameplay"],
             f"Video dari @{author} berkaitan dengan gaming atau esports"),
        ]

        # Cari konteks yang cocok
        opening = None
        for context_tags, context_desc in _REPOST_CONTEXT_MAP:
            if any(tag in all_text_lower for tag in context_tags):
                opening = context_desc
                break

        if not opening:
            opening = f"Video dari @{author}"

        # Susun deskripsi repost yang kaya dan menjelaskan
        desc_parts: List[str] = []

        # 1. Siapa kreator aslinya dan apa konteksnya
        desc_parts.append(opening)

        # 2. Pesan/caption dari video asli
        if clean_caption and len(clean_caption) > 3:
            if len(clean_caption) > 50:
                desc_parts.append(f'Kreator asli mengungkapkan: "{clean_caption[:200]}"')
            else:
                desc_parts.append(f'dengan pesan: "{clean_caption}"')

        # 3. Topik konten
        if main_topic != "umum":
            desc_parts.append(f"Konten ini berfokus pada topik {main_topic}")

        # 4. Semua hashtag yang dipakai
        if hashtags:
            tag_str = " ".join(f"#{h}" for h in hashtags[:6])
            desc_parts.append(f"Hashtag yang digunakan: {tag_str}")

        # 5. Kenapa di-repost (inferensi dari sentimen dan topik)
        why_repost = {
            "hiburan":    f"Kemungkinan di-repost karena kontennya menghibur",
            "motivasi":   f"Kemungkinan di-repost karena kontennya inspiratif",
            "olahraga":   f"Kemungkinan di-repost karena relevan dengan minat olahraga",
            "perjalanan": f"Kemungkinan di-repost karena menampilkan destinasi atau pengalaman menarik",
            "kesehatan":  f"Kemungkinan di-repost karena berkaitan dengan gaya hidup sehat",
            "keluarga":   f"Kemungkinan di-repost karena kontennya personal dan relatable",
            "promosi":    f"Kemungkinan di-repost sebagai bagian dari promosi atau kolaborasi",
        }.get(main_topic, "")
        if why_repost:
            desc_parts.append(why_repost)

        # 6. Sentimen konten
        sent_phrases_repost = {
            "POSITIVE": "Konten ini bersentimen positif dan kemungkinan dianggap menginspirasi atau menghibur",
            "NEGATIVE": "Konten ini bersentimen negatif — mungkin berisi kritik, drama, atau konten kontroversial",
            "NEUTRAL":  "Konten ini bersentimen netral — bersifat informatif atau deskriptif",
        }
        desc_parts.append(sent_phrases_repost.get(sent["label"], sent_phrases_repost["NEUTRAL"]))

        content_description = ". ".join(desc_parts).rstrip(".") + "."

        enriched.append({
            **r,
            "sentiment":           sent,
            "topic":               main_topic,
            "content_description": content_description,
        })

    return enriched



def analyze_user_comments(
    username: str,
    user_comments: List[Dict]
) -> Dict[str, Any]:
    """
    Analisis 15 komentar terakhir yang ditulis si target.
    Mengembalikan sentimen, breakdown, topik, gaya bahasa, flags, summary.
    """
    if not user_comments:
        return {
            "comment_sentiment_overall":   "NEUTRAL",
            "comment_sentiment_score":     0.5,
            "comment_sentiment_breakdown": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
            "comment_topics":              [],
            "comment_style":               "tidak ada data",
            "comment_red_flags":           [],
            "comment_green_flags":         [],
            "comment_summary":             f"Tidak ada komentar @{username} yang terdeteksi.",
        }

    _load_model()

    enriched: List[Dict] = []
    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    scores: List[float] = []

    for c in user_comments:
        text = c.get("content") or ""
        sent = _analyze_single_sentiment(text)
        label = sent["label"]
        score = sent["score"]
        counts[label] += 1
        scores.append(score if label == "POSITIVE" else (1.0 - score if label == "NEGATIVE" else 0.5))
        enriched.append({**c, "sentiment": sent})

    total = max(len(enriched), 1)
    breakdown = {
        "POSITIVE": round(counts["POSITIVE"] / total * 100, 1),
        "NEGATIVE": round(counts["NEGATIVE"] / total * 100, 1),
        "NEUTRAL":  round(counts["NEUTRAL"]  / total * 100, 1),
    }

    # Overall
    if breakdown["POSITIVE"] >= breakdown["NEGATIVE"] and breakdown["POSITIVE"] >= breakdown["NEUTRAL"]:
        overall = "POSITIVE"
    elif breakdown["NEGATIVE"] >= breakdown["POSITIVE"] and breakdown["NEGATIVE"] >= breakdown["NEUTRAL"]:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"

    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.5

    # ── Topik dari komentar ───────────────────────────────────────────────
    all_comment_text = " ".join(c.get("content") or "" for c in user_comments).lower()
    topic_hits: Dict[str, int] = {}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in all_comment_text)
        if hits > 0:
            topic_hits[topic] = hits
    comment_topics = sorted(topic_hits, key=lambda k: topic_hits[k], reverse=True)[:3]

    # ── Gaya bahasa ───────────────────────────────────────────────────────
    avg_len = sum(len((c.get("content") or "").split()) for c in user_comments) / total
    has_emoji = any(re.search(r"[^\w\s,.'!?-]", c.get("content") or "") for c in user_comments)
    if avg_len <= 3:
        style = "singkat/ekspresif"
    elif avg_len <= 8:
        style = "kasual"
    else:
        style = "deskriptif"
    if has_emoji:
        style += " dengan emoji"

    # ── Flags dari komentar ───────────────────────────────────────────────
    comment_red_flags: List[str] = []
    comment_green_flags: List[str] = []

    high_hits_list = [kw for kw in _HIGH_RISK_KEYWORDS if kw in all_comment_text]
    if high_hits_list:
        comment_red_flags.append(f"🚨 Kata berisiko dalam komentar: {', '.join(set(high_hits_list[:3]))}")
    if breakdown["NEGATIVE"] >= 40:
        comment_red_flags.append(f"🚨 {breakdown['NEGATIVE']}% komentar bersentimen negatif")
    if breakdown["POSITIVE"] >= 60:
        comment_green_flags.append(f"✅ {breakdown['POSITIVE']}% komentar bersentimen positif")
    if not high_hits_list:
        comment_green_flags.append("✅ Tidak ada kata berisiko dalam komentar")

    # ── Summary ───────────────────────────────────────────────────────────
    topic_str  = f", topik: {', '.join(comment_topics)}" if comment_topics else ""
    summary = (
        f"Dari {len(user_comments)} komentar terakhir @{username}: "
        f"{breakdown['POSITIVE']:.0f}% positif, "
        f"{breakdown['NEGATIVE']:.0f}% negatif, "
        f"{breakdown['NEUTRAL']:.0f}% netral. "
        f"Gaya bahasa: {style}{topic_str}."
    )

    print(f"   ↳ Comment analysis: {overall} | {breakdown}")

    return {
        "comment_sentiment_overall":   overall,
        "comment_sentiment_score":     avg_score,
        "comment_sentiment_breakdown": breakdown,
        "comment_topics":              comment_topics,
        "comment_style":               style,
        "comment_red_flags":           comment_red_flags,
        "comment_green_flags":         comment_green_flags,
        "comment_summary":             summary,
        # Kembalikan juga komentar yang sudah di-enrich dengan sentiment
        "enriched_user_comments":      enriched,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI UTAMA — dipanggil dari main.py
# ─────────────────────────────────────────────────────────────────────────────
def analyze_user_behavior(
    username: str,
    platform: str,
    tweets_text: str = "",           # kept for backward compatibility
    activities: Optional[List[Dict]] = None,
    profile: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Analisis lengkap perilaku user berdasarkan activities list.

    Parameters
    ----------
    username    : nama pengguna
    platform    : tiktok / instagram / x / facebook
    tweets_text : teks gabungan (legacy, masih diterima)
    activities  : list aktivitas dari scraper_engine (format baru)
    profile     : dict profil dari scraper_engine

    Returns
    -------
    Dict analytics_insight dengan semua field lengkap
    """
    if activities is None:
        activities = []
    if profile is None:
        profile = {}

    print(f"🧠 [AI] Mulai analisis @{username} | {len(activities)} postingan")

    # ── Pre-load model ────────────────────────────────────────────────────
    _load_model()

    # ── Sentimen PER POSTINGAN ────────────────────────────────────────────
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    sentiment_scores: List[float] = []
    enriched_activities: List[Dict] = []

    for activity in activities:
        # ── Pilih input teks terbaik yang tersedia ────────────────────────
        # Prioritas: combined_text (caption+OCR+transcript) → caption → description
        # combined_text diisi oleh main.py setelah media extraction.
        # Jika extraction belum jalan / gagal, fallback ke caption (behavior lama).
        text = (
            activity.get("combined_text") or   # ← hasil gabungan OCR + transcript + caption
            activity.get("text") or            # ← caption mentah (fallback)
            activity.get("content_text") or
            activity.get("content_description") or
            ""
        )

        sent = _analyze_single_sentiment(text)
        label = sent["label"]
        score = sent["score"]

        sentiment_counts[label] += 1
        # normalise score ke arah 0–1 (POSITIVE mendekat 1, NEGATIVE mendekat 0)
        if label == "POSITIVE":
            sentiment_scores.append(score)
        elif label == "NEGATIVE":
            sentiment_scores.append(1.0 - score)
        else:
            sentiment_scores.append(0.5)

        # Update sentiment di activity dict (in-place)
        updated = dict(activity)
        updated["sentiment"] = {"label": label, "score": score}

        # ── Analisis isi konten per post ──────────────────────────────────
        content_analysis = _analyze_post_content(text, activity)
        updated["content_analysis"] = content_analysis

        # ── Sentimen per KOMENTAR ─────────────────────────────────────────
        enriched_comments = []
        for c in (activity.get("comments_data") or []):
            c_text = c.get("content") or c.get("text") or ""
            c_sent = _analyze_single_sentiment(c_text)
            enriched_c = dict(c)
            enriched_c["sentiment"] = c_sent
            enriched_comments.append(enriched_c)
        updated["comments_data"] = enriched_comments

        enriched_activities.append(updated)

    total = max(len(activities), 1)

    # ── Sentiment breakdown (persentase) ──────────────────────────────────
    sentiment_breakdown = {
        "POSITIVE": round(sentiment_counts["POSITIVE"] / total * 100, 1),
        "NEGATIVE": round(sentiment_counts["NEGATIVE"] / total * 100, 1),
        "NEUTRAL":  round(sentiment_counts["NEUTRAL"]  / total * 100, 1),
    }

    # ── Overall sentiment ─────────────────────────────────────────────────
    avg_score = round(sum(sentiment_scores) / len(sentiment_scores), 4) if sentiment_scores else 0.5

    if sentiment_breakdown["POSITIVE"] >= sentiment_breakdown["NEGATIVE"] and \
       sentiment_breakdown["POSITIVE"] >= sentiment_breakdown["NEUTRAL"]:
        sentiment_overall = "POSITIVE"
    elif sentiment_breakdown["NEGATIVE"] >= sentiment_breakdown["POSITIVE"] and \
         sentiment_breakdown["NEGATIVE"] >= sentiment_breakdown["NEUTRAL"]:
        sentiment_overall = "NEGATIVE"
    else:
        sentiment_overall = "NEUTRAL"

    print(
        f"   ↳ Sentimen: {sentiment_overall} | "
        f"POSITIVE={sentiment_breakdown['POSITIVE']}% "
        f"NEGATIVE={sentiment_breakdown['NEGATIVE']}% "
        f"NEUTRAL={sentiment_breakdown['NEUTRAL']}%"
    )

    # ── Risk score ────────────────────────────────────────────────────────
    risk_score = _compute_risk_score(enriched_activities, sentiment_breakdown)

    if risk_score <= 25:
        risk_level = "SANGAT RENDAH"
    elif risk_score <= 45:
        risk_level = "RENDAH"
    elif risk_score <= 65:
        risk_level = "SEDANG"
    elif risk_score <= 80:
        risk_level = "TINGGI"
    else:
        risk_level = "SANGAT TINGGI"

    print(f"   ↳ Risk: {risk_score}/100 ({risk_level})")

    # ── Personality ───────────────────────────────────────────────────────
    personality = _compute_personality(enriched_activities, sentiment_breakdown, profile)

    # ── Flags & rekomendasi ───────────────────────────────────────────────
    flags = _compute_flags(enriched_activities, sentiment_breakdown, risk_score, profile)

    # ── Summary objektif ──────────────────────────────────────────────────
    summary_text = _build_summary(
        username, platform, profile,
        enriched_activities, sentiment_breakdown,
        risk_score, risk_level
    )

    print(f"   ↳ Summary: {summary_text[:100]}...")

    return {
        # Aktivitas dengan sentimen sudah diisi
        "enriched_activities": enriched_activities,

        # Analytics
        "sentiment_overall": sentiment_overall,
        "sentiment_score": avg_score,
        "sentiment_breakdown": sentiment_breakdown,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "personality": personality,
        "summary_text": summary_text,
        "red_flags": flags["red_flags"],
        "green_flags": flags["green_flags"],
        "recommendation": flags["recommendation"],
        "suitable_roles": flags["suitable_roles"],
    }
