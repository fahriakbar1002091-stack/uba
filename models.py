# models.py - FULL VERSION DENGAN SEMUA FITUR
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.sql import func
from database import Base
from datetime import datetime

class Tenant(Base):
    __tablename__ = "tenants"

    tenant_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    daily_scan_limit = Column(Integer, default=50)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    api_key = Column(String(100), unique=True, nullable=True)
    is_active = Column(Boolean, default=True)  # Tambahkan ini

    def __repr__(self):
        return f"<Tenant(id={self.tenant_id}, name={self.name})>"

class SocialUser(Base):
    __tablename__ = "social_users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), nullable=False)
    platform = Column(String(30), nullable=False)
    target_type = Column(String(30), default="USER_PROFILE")
    last_scanned = Column(DateTime(timezone=True), server_default=func.now())
    
    # Profile Data
    full_name = Column(String(200), nullable=True)
    bio = Column(Text, nullable=True)
    followers = Column(Integer, default=0)
    following = Column(Integer, default=0)
    verified = Column(Boolean, default=False)
    profile_pic = Column(Text, nullable=True)  # URL bisa sangat panjang

    def __repr__(self):
        return f"<SocialUser(id={self.user_id}, username={self.username}, platform={self.platform})>"

class Activity(Base):
    __tablename__ = "activities"

    activity_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("social_users.user_id"), nullable=False)
    action_type = Column(String(50), nullable=False)  # POST, REPOST, COMMENT, STORY
    content_text = Column(Text, nullable=True)
    image_path = Column(Text, nullable=True)   # URL bisa sangat panjang
    post_url = Column(Text, nullable=True)     # URL bisa sangat panjang
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # AI Text Analysis
    ai_sentiment = Column(String(20), nullable=True)  # POSITIVE, NEGATIVE, NEUTRAL
    ai_risk_score = Column(Integer, default=50)

    # Media Extraction — diisi setelah OCR/transcript dari media URL
    combined_text    = Column(Text, nullable=True)   # caption + OCR + transcript
    ocr_text         = Column(Text, nullable=True)   # hasil OCR dari image/video frames
    audio_transcript = Column(Text, nullable=True)   # hasil speech-to-text dari video

    # AI Image Analysis
    ai_caption = Column(Text, nullable=True)
    ai_objects = Column(Text, nullable=True)   # JSON string
    ai_topics = Column(Text, nullable=True)    # JSON string
    ai_mood = Column(Text, nullable=True)      # JSON string
    ai_summary = Column(Text, nullable=True)
    analyzed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<Activity(id={self.activity_id}, type={self.action_type})>"

class AISummary(Base):
    __tablename__ = "ai_summaries"

    summary_id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.tenant_id"), nullable=False)
    user_id = Column(Integer, ForeignKey("social_users.user_id"), nullable=False)
    summary_text = Column(Text, nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Tambahan untuk AI Image
    image_analysis = Column(Text, nullable=True)  # JSON full result

    def __repr__(self):
        return f"<AISummary(id={self.summary_id}, user={self.user_id})>"

# Log untuk AI Image Analysis
class ImageAnalysisLog(Base):
    __tablename__ = "image_analysis_logs"

    log_id = Column(Integer, primary_key=True, index=True)
    activity_id = Column(Integer, ForeignKey("activities.activity_id"), nullable=True)
    image_path = Column(String(500), nullable=False)  # diperbesar dari 255
    ai_caption = Column(Text, nullable=True)
    ai_objects = Column(Text, nullable=True)
    ai_topics = Column(Text, nullable=True)
    ai_mood = Column(Text, nullable=True)
    ai_summary = Column(Text, nullable=True)
    analyzed_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ImageAnalysisLog(id={self.log_id})>"