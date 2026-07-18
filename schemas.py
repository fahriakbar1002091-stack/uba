# schemas.py - FULL VERSION DENGAN AI IMAGE ANALYSIS
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Any
from enum import Enum

class PlatformEnum(str, Enum):
    instagram = "instagram"
    x = "x"
    tiktok = "tiktok"
    threads = "threads"
    facebook = "facebook"

class ScanTypeEnum(str, Enum):
    user_profile = "USER_PROFILE"
    keyword_trend = "KEYWORD_TREND"

class DynamicScanRequest(BaseModel):
    tenant_id: int
    platform: PlatformEnum
    scan_type: ScanTypeEnum
    target: str

class TenantBase(BaseModel):
    name: str
    daily_scan_limit: Optional[int] = 50

class TenantCreate(TenantBase):
    pass

class TenantResponse(BaseModel):
    tenant_id: int
    name: str
    daily_scan_limit: int
    created_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True

class SocialUserBase(BaseModel):
    username: str
    platform: str
    target_type: str

class SocialUserCreate(SocialUserBase):
    pass

class SocialUserResponse(BaseModel):
    user_id: int
    username: str
    platform: str
    target_type: str
    last_scanned: datetime
    full_name: Optional[str]
    bio: Optional[str]
    followers: Optional[int]
    following: Optional[int]
    verified: Optional[bool]
    profile_pic: Optional[str]

    class Config:
        from_attributes = True

class ActivityBase(BaseModel):
    user_id: int
    action_type: str
    content_text: Optional[str]
    image_path: Optional[str]
    post_url: Optional[str]

class ActivityCreate(ActivityBase):
    pass

class ActivityResponse(ActivityBase):
    activity_id: int
    created_at: datetime
    
    # AI Text Analysis
    ai_sentiment: Optional[str]
    ai_risk_score: Optional[int]

    # AI Image Analysis
    ai_caption: Optional[str]
    ai_objects: Optional[List[str]]
    ai_topics: Optional[List[str]]
    ai_mood: Optional[List[str]]
    ai_summary: Optional[str]
    analyzed_at: Optional[datetime]

    class Config:
        from_attributes = True

class AISummaryResponse(BaseModel):
    summary_id: int
    user_id: int
    tenant_id: int
    summary_text: str
    generated_at: datetime
    image_analysis: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True

# New Schema for AI Image Analysis
class AIImageAnalysisResponse(BaseModel):
    ai_caption: str
    ai_objects: List[str]
    ai_topics: List[str]
    ai_mood: List[str]
    ai_summary: str
    analyzed_at: datetime

class ImageAnalysisRequest(BaseModel):
    image_path: str
    caption: Optional[str] = ""

class ImageAnalysisResponse(BaseModel):
    status: str
    filename: str
    ai_analysis: AIImageAnalysisResponse

# Response for Scan with AI
class ScanResponse(BaseModel):
    status: str
    timestamp: datetime
    scan_meta: Dict
    profile: Dict
    behavior_metrics: Dict
    analytics_insight: Dict
    ai_image_analysis: Optional[AIImageAnalysisResponse]

    class Config:
        from_attributes = True