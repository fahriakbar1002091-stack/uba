# main.py - Social Radar API
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import random
import os
import shutil
import json
import traceback

from database import get_db, engine
import models
import schemas
from scraper_engine import run_crawler
from ai_engine import analyze_user_behavior, analyze_user_comments, analyze_reposts
from ai_image_analysis import analyze_image_real, is_image_file
from media_extractor import extract_post_content, build_combined_text, check_dependencies

models.Base.metadata.create_all(bind=engine)

# ── Cek dependency media extraction saat startup ──────────────────────────
_media_deps = check_dependencies()
_extraction_enabled = os.getenv("MEDIA_EXTRACTION_ENABLED", "true").lower() == "true"

import logging as _logging
_logger = _logging.getLogger("social_radar")
_logger.info(f"Media extraction deps: {_media_deps}")
_logger.info(f"Media extraction enabled: {_extraction_enabled}")

# ── App setup ─────────────────────────────────────────────────────────────────
api_key_scheme = APIKeyHeader(name="X-API-KEY", auto_error=False)

app = FastAPI(
    title="Social Radar - Social Media Analytics Engine",
    version="3.0.0",
    description="Analisis perilaku user media sosial berbasis AI",
    docs_url="/docs",
    redoc_url="/redoc"
)

MASTER_API_KEY = "RAHASIA_AGENSIDETEKTIF_123"


def verify_api_key(api_key: str = Depends(api_key_scheme)):
    if not api_key or api_key != MASTER_API_KEY:
        raise HTTPException(status_code=401, detail="API Key tidak valid!")
    return api_key


# ── Global error handler — tampilkan detail error, bukan cuma '500' ──────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "ERROR",
            "error": type(exc).__name__,
            "detail": str(exc),
            "traceback": traceback.format_exc().splitlines()[-6:]  # 6 baris terakhir
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "ONLINE",
        "service": "Social Radar API v3.0",
        "docs": "/docs"
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG — lihat raw item dari Apify untuk deteksi struktur komentar
# Bisa diakses via header X-API-KEY ATAU query param ?key=
# Contoh: GET /api/v1/debug/tiktok-raw/khaby.lame?key=RAHASIA_AGENSIDETEKTIF_123
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/debug/tiktok-raw/{username}")
async def debug_tiktok_raw(
    username: str,
    key: str = "",                          # query param fallback
    api_key: str = Depends(api_key_scheme)  # header
):
    """
    Debug: dump SEMUA field mentah dari dataset Apify TikTok.
    Akses via header X-API-KEY atau ?key= di URL.
    """
    # Validasi key dari header atau query param
    effective_key = api_key or key
    if effective_key != MASTER_API_KEY:
        raise HTTPException(status_code=401, detail="API Key tidak valid!")

    import asyncio
    from functools import partial
    from apify_client import ApifyClient

    token = os.getenv("APIFY_TOKEN", "")
    if not token:
        raise HTTPException(status_code=500, detail="APIFY_TOKEN tidak ada di .env")

    def _fetch():
        client = ApifyClient(token)
        run = client.actor("clockworks/tiktok-scraper").call(
            run_input={
                "profiles": [username],
                "resultsPerPage": 3,
                "maxVideosPerProfile": 3,
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False,
                "shouldDownloadSubtitles": False,
                "includeComments": True,
                "maxComments": 5,
                "proxyConfiguration": {"useApifyProxy": True},
            }
        )
        dataset_id = getattr(run, "default_dataset_id", None)
        if not dataset_id:
            return []
        return client.dataset(dataset_id).list_items().items or []

    loop = asyncio.get_event_loop()
    items = await loop.run_in_executor(None, _fetch)

    debug_items = []
    for i, item in enumerate(items[:15]):
        # Cek semua field yang mungkin mengandung komentar
        all_comment_fields = {
            k: v for k, v in item.items()
            if "comment" in k.lower() or "reply" in k.lower()
        }

        debug_items.append({
            "index":       i,
            # Field tipe item — kunci utama untuk pisahkan video vs komentar
            "type":        item.get("type"),
            "itemType":    item.get("itemType"),
            "isRepost":    item.get("isRepost"),
            # ID fields
            "id":          item.get("id"),
            "videoId":     item.get("videoId"),
            "awemeId":     item.get("awemeId"),
            # Cek komentar nested
            "latestComments_count": len(item.get("latestComments") or []),
            "comments_count":       len(item.get("comments") or []),
            # Dump SEMUA field yang mengandung kata "comment" / "reply"
            "comment_related_fields": all_comment_fields,
            # Sample komentar pertama jika ada (nested)
            "latestComments_sample": (item.get("latestComments") or [])[:2],
            "comments_sample":       (item.get("comments") or [])[:2],
            # Teks postingan
            "text_snippet": str(item.get("text") or item.get("desc") or "")[:120],
            # SEMUA keys — untuk tahu field apa saja yang ada
            "all_keys": sorted(item.keys()),
        })

    return {
        "username":    username,
        "total_items": len(items),
        "hint": "Perhatikan 'type'/'itemType' tiap item dan 'all_keys' untuk lihat field komentar",
        "items":       debug_items
    }


# ── DEBUG REPOST — test parameter clockworks/tiktok-profile-scraper ──────────
@app.get("/api/v1/debug/tiktok-reposts/{username}")
async def debug_tiktok_reposts(
    username: str,
    key: str = "",
    api_key: str = Depends(api_key_scheme)
):
    """
    Debug: test scraping repost profil TikTok.
    Akses: GET /api/v1/debug/tiktok-reposts/williesalim?key=RAHASIA_AGENSIDETEKTIF_123
    """
    effective_key = api_key or key
    if effective_key != MASTER_API_KEY:
        raise HTTPException(status_code=401, detail="API Key tidak valid!")

    import asyncio
    from apify_client import ApifyClient

    token = os.getenv("APIFY_TOKEN", "")
    if not token:
        raise HTTPException(status_code=500, detail="APIFY_TOKEN tidak ada di .env")

    def _fetch_reposts():
        client = ApifyClient(token)
        run = client.actor("clockworks/tiktok-profile-scraper").call(
            run_input={
                "profiles":             [username],
                "profileSection":       "reposts",
                "resultsPerPage":       10,
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False,
            }
        )
        dataset_id = getattr(run, "default_dataset_id", None)
        if not dataset_id:
            return [], "no_dataset_id"
        items = client.dataset(dataset_id).list_items().items or []
        return items, "ok"

    loop = asyncio.get_event_loop()
    items, status = await loop.run_in_executor(None, _fetch_reposts)

    first_keys = sorted(items[0].keys()) if items else []
    first_sample = {}
    if items:
        for k in ["id", "text", "webVideoUrl", "createTimeISO", "authorMeta",
                  "isRepost", "repostSource", "type", "itemType", "errorCode",
                  "diggCount", "commentCount", "shareCount", "playCount"]:
            first_sample[k] = items[0].get(k)

    return {
        "username":          username,
        "fetch_status":      status,
        "total_items":       len(items),
        "first_item_keys":   first_keys,
        "first_item_sample": first_sample,
        "preview": [
            {
                "id":           item.get("id"),
                "text":         str(item.get("text") or "")[:80],
                "webVideoUrl":  item.get("webVideoUrl"),
                "author_name":  (item.get("authorMeta") or {}).get("name"),
                "author_unique":(item.get("authorMeta") or {}).get("uniqueId"),
                "isRepost":     item.get("isRepost"),
                "type":         item.get("type"),
                "errorCode":    item.get("errorCode"),
                "all_keys":     sorted(item.keys()),
            }
            for item in items[:10]
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# TENANTS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/tenants", response_model=schemas.TenantResponse, status_code=201)
def create_tenant(
    tenant: schemas.TenantCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    generated_key = f"KEY_TENANT_{random.randint(10000, 99999)}"
    db_tenant = models.Tenant(
        name=tenant.name,
        daily_scan_limit=tenant.daily_scan_limit,
        api_key=generated_key,
        is_active=True
    )
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    return db_tenant


@app.get("/api/v1/tenants", response_model=List[schemas.TenantResponse])
def get_all_tenants(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    return db.query(models.Tenant).all()


@app.get("/api/v1/tenants/{tenant_id}", response_model=schemas.TenantResponse)
def get_tenant(
    tenant_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    tenant = db.query(models.Tenant).filter(models.Tenant.tenant_id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant tidak ditemukan.")
    return tenant


# ─────────────────────────────────────────────────────────────────────────────
# SCAN TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/scan/trigger", status_code=201)
async def trigger_social_scan(
    payload: schemas.DynamicScanRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    # Validasi tenant
    tenant = db.query(models.Tenant).filter(
        models.Tenant.tenant_id == payload.tenant_id,
        models.Tenant.is_active == True
    ).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant tidak ditemukan atau tidak aktif.")

    # ── 1. Scrape ──────────────────────────────────────────────────────────
    scrape_result, source_status = await run_crawler(
        platform=payload.platform.value,
        target=payload.target,
        scan_type=payload.scan_type.value
    )

    scrape_profile    = scrape_result.get("profile", {})
    raw_activities    = scrape_result.get("activities", [])
    user_comments     = scrape_result.get("user_comments", [])
    reposts           = scrape_result.get("reposts", [])
    fb_photos         = scrape_result.get("photos", [])   # Facebook only
    scrape_errors     = scrape_result.get("scrape_errors", [])

    # ── 1b. Media Extraction — OCR + transcript per post ──────────────────
    # Dijalankan di thread executor agar tidak blocking event loop.
    # Dikendalikan oleh env var MEDIA_EXTRACTION_ENABLED (default: true).
    # Jika extraction gagal per-post, combined_text fallback ke caption.
    # Pipeline lama tetap berjalan normal meski extraction dimatikan.
    if _extraction_enabled and raw_activities:
        import asyncio
        from functools import partial

        loop = asyncio.get_event_loop()

        async def _extract_one(activity: dict) -> dict:
            """Jalankan extraction satu post di thread, dengan timeout 60 detik."""
            try:
                extraction = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        partial(extract_post_content, activity)
                    ),
                    timeout=60.0
                )
                ocr_text        = extraction.get("ocr_text", "")
                audio_transcript= extraction.get("audio_transcript", "")
                visual_desc     = extraction.get("visual_description", "")
                ext_errors      = extraction.get("extraction_errors", [])

                combined = build_combined_text(
                    caption_text      = activity.get("caption_text") or activity.get("text") or "",
                    ocr_text          = ocr_text,
                    audio_transcript  = audio_transcript,
                    visual_description= visual_desc,
                )

                return {
                    **activity,
                    "ocr_text":          ocr_text,
                    "audio_transcript":  audio_transcript,
                    "visual_description": visual_desc,
                    "combined_text":     combined or activity.get("text") or "",
                    "extraction_errors": ext_errors,
                }
            except asyncio.TimeoutError:
                return {
                    **activity,
                    "combined_text":     activity.get("text") or "",
                    "extraction_errors": ["Extraction timeout (>60s)"],
                }
            except Exception as exc:
                return {
                    **activity,
                    "combined_text":     activity.get("text") or "",
                    "extraction_errors": [f"Extraction error: {str(exc)}"],
                }

        # Jalankan semua post secara concurrent (tapi tetap di thread pool)
        raw_activities = list(await asyncio.gather(
            *[_extract_one(act) for act in raw_activities]
        ))

    # ── 2. Upsert SocialUser ───────────────────────────────────────────────
    try:
        db_user = db.query(models.SocialUser).filter(
            models.SocialUser.username == payload.target,
            models.SocialUser.platform == payload.platform.value
        ).first()

        if not db_user:
            db_user = models.SocialUser(
                username=payload.target,
                platform=payload.platform.value,
                target_type=payload.scan_type.value
            )
            db.add(db_user)
            db.flush()  # dapatkan user_id tanpa commit dulu

        db_user.full_name    = scrape_profile.get("full_name")    or db_user.full_name
        db_user.bio          = scrape_profile.get("bio")          or db_user.bio
        db_user.followers    = scrape_profile.get("followers",  0)
        db_user.following    = scrape_profile.get("following",  0)
        db_user.verified     = scrape_profile.get("verified",   False)
        db_user.profile_pic  = scrape_profile.get("profile_pic") or db_user.profile_pic
        db_user.last_scanned = datetime.now()
        db.commit()
        db.refresh(db_user)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error saat simpan user: {e}")

    # ── 3. Simpan activities ───────────────────────────────────────────────
    try:
        for item in raw_activities:
            ai_image = item.get("ai_image_analysis")
            db_activity = models.Activity(
                user_id          = db_user.user_id,
                action_type      = item.get("action_type", "POST"),
                content_text     = item.get("text", ""),
                combined_text    = item.get("combined_text") or item.get("text") or "",
                ocr_text         = item.get("ocr_text", ""),
                audio_transcript = item.get("audio_transcript", ""),
                image_path       = item.get("image_url"),
                post_url         = item.get("post_url"),
                ai_caption       = ai_image.get("ai_caption") if ai_image else None,
                ai_objects       = json.dumps(ai_image.get("ai_objects", [])) if ai_image else None,
                ai_topics        = json.dumps(ai_image.get("ai_topics",  [])) if ai_image else None,
                ai_mood          = json.dumps(ai_image.get("ai_mood",    [])) if ai_image else None,
                ai_summary       = ai_image.get("ai_summary")  if ai_image else None,
                analyzed_at      = datetime.now() if ai_image else None
            )
            db.add(db_activity)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error saat simpan activities: {e}")

    # ── 4. AI Analysis ────────────────────────────────────────────────────
    ai_result = analyze_user_behavior(
        username   = payload.target,
        platform   = payload.platform.value,
        activities = raw_activities,
        profile    = scrape_profile
    )

    enriched_activities = ai_result.pop("enriched_activities", raw_activities)

    # ── 4b. Analisis sentimen untuk user_comments ──────────────────────────
    # user_comments = komentar yang dibuat si target di postingannya sendiri
    from ai_engine import _analyze_single_sentiment
    enriched_user_comments = []
    for uc in user_comments:
        c_text = uc.get("content") or ""
        sent   = _analyze_single_sentiment(c_text) if c_text else {"label": "NEUTRAL", "score": 0.5}
        enriched_user_comments.append({**uc, "sentiment": sent})
    user_comments = enriched_user_comments

    # ── 4c. Analisis insight komentar user ─────────────────────────────────
    comment_insight = analyze_user_comments(
        username      = payload.target,
        user_comments = user_comments
    )
    # Komentar sudah di-enrich di dalam analyze_user_comments, pakai itu
    user_comments = comment_insight.pop("enriched_user_comments", user_comments)

    # ── 4d. Analisis repost ────────────────────────────────────────────────
    reposts = analyze_reposts(username=payload.target, reposts=reposts)

    # Update sentiment di DB (best-effort, tidak crash kalau gagal)
    try:
        saved = (
            db.query(models.Activity)
            .filter(models.Activity.user_id == db_user.user_id)
            .order_by(models.Activity.activity_id.desc())
            .limit(len(enriched_activities))
            .all()
        )
        for db_act, enriched in zip(saved, enriched_activities):
            sent = enriched.get("sentiment", {})
            db_act.ai_sentiment  = sent.get("label", "NEUTRAL")
            db_act.ai_risk_score = ai_result.get("risk_score", 50)
        db.commit()
    except Exception:
        db.rollback()  # non-critical, lanjut

    # ── 5. Simpan AI Summary ───────────────────────────────────────────────
    try:
        db_summary = models.AISummary(
            tenant_id   = tenant.tenant_id,
            user_id     = db_user.user_id,
            summary_text= ai_result.get("summary_text", "")
        )
        db.add(db_summary)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error saat simpan summary: {e}")

    # ── 6. Response ────────────────────────────────────────────────────────
    return {
        "status":    "SUCCESS",
        "timestamp": datetime.now().isoformat(),
        "scan_meta": {
            "platform":          payload.platform.value.upper(),
            "scan_type":         payload.scan_type.value,
            "target":            payload.target,
            "data_source_status": source_status,
            "db_user_id":        db_user.user_id,
            "scrape_errors":     scrape_errors
        },
        "profile": {
            "username":     scrape_profile.get("username",     payload.target),
            "full_name":    scrape_profile.get("full_name",    ""),
            "bio":          scrape_profile.get("bio",          ""),
            "about_me":     scrape_profile.get("about_me",     ""),
            "followers":    scrape_profile.get("followers",    0),
            "following":    scrape_profile.get("following",    0),
            "page_likes":   scrape_profile.get("page_likes",   0),
            "verified":     scrape_profile.get("verified",     False),
            "profile_pic":  scrape_profile.get("profile_pic",  ""),
            # Facebook
            "cover_photo":  scrape_profile.get("cover_photo",  ""),
            "address":      scrape_profile.get("address",      ""),
            "email":        scrape_profile.get("email",        ""),
            "phone":        scrape_profile.get("phone",        ""),
            "categories":   scrape_profile.get("categories",   []),
            "rating":       scrape_profile.get("rating",       ""),
            "rating_count": scrape_profile.get("rating_count", 0),
            # X/Twitter
            "user_id":      scrape_profile.get("user_id",      ""),
            "banner_image": scrape_profile.get("banner_image", ""),
            "tweet_count":  scrape_profile.get("tweet_count",  0),
            "data_limitations": scrape_profile.get("data_limitations", []),
            # Shared
            "location":     scrape_profile.get("location",     ""),
            "website":      scrape_profile.get("website",      ""),
            "join_date":    scrape_profile.get("join_date",    ""),
            "total_posts":  scrape_profile.get("total_posts",  len(enriched_activities)),
            "total_likes":  scrape_profile.get("total_likes",  0),
            "platform":     payload.platform.value,
            "last_updated": scrape_profile.get("last_updated", datetime.now().strftime("%Y-%m-%d"))
        },
        "behavior_metrics": {
            "total_extracted":   len(enriched_activities),
            "activities":        enriched_activities,
            "user_comments":     user_comments,
            "reposts":           reposts,
            "photos":            fb_photos,    # Facebook: foto publik dari album
        },
        "analytics_insight": {
            "sentiment_overall":   ai_result.get("sentiment_overall",   "NEUTRAL"),
            "sentiment_score":     ai_result.get("sentiment_score",     0.5),
            "sentiment_breakdown": ai_result.get("sentiment_breakdown", {}),
            "risk_score":          ai_result.get("risk_score",          0),
            "risk_level":          ai_result.get("risk_level",          "RENDAH"),
            "personality":         ai_result.get("personality",         {}),
            "summary_text":        ai_result.get("summary_text",        ""),
            "red_flags":           ai_result.get("red_flags",           []),
            "green_flags":         ai_result.get("green_flags",         []),
            "recommendation":      ai_result.get("recommendation",      ""),
            "suitable_roles":      ai_result.get("suitable_roles",      []),
            # ── Insight komentar user (15 terbaru) ────────────────────────
            "comment_sentiment_overall":   comment_insight.get("comment_sentiment_overall",   "NEUTRAL"),
            "comment_sentiment_score":     comment_insight.get("comment_sentiment_score",     0.5),
            "comment_sentiment_breakdown": comment_insight.get("comment_sentiment_breakdown", {}),
            "comment_topics":              comment_insight.get("comment_topics",              []),
            "comment_style":               comment_insight.get("comment_style",               ""),
            "comment_red_flags":           comment_insight.get("comment_red_flags",           []),
            "comment_green_flags":         comment_insight.get("comment_green_flags",         []),
            "comment_summary":             comment_insight.get("comment_summary",             ""),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCAN HISTORY
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/scan/history/{tenant_id}")
def get_scan_history(
    tenant_id: int,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    histories = (
        db.query(models.AISummary)
        .filter(models.AISummary.tenant_id == tenant_id)
        .order_by(models.AISummary.generated_at.desc())
        .all()
    )
    if not histories:
        raise HTTPException(status_code=404, detail="Belum ada history scan.")
    return {
        "status":        "SUCCESS",
        "tenant_id":     tenant_id,
        "total_records": len(histories),
        "data": [
            {
                "summary_id": h.summary_id,
                "user_id":    h.user_id,
                "summary_text": h.summary_text,
                "created_at": h.generated_at.strftime("%Y-%m-%d %H:%M:%S") if h.generated_at else ""
            }
            for h in histories
        ]
    }


@app.get("/api/v1/scan/user/{platform}/{username}")
def get_user_scan_result(
    platform: str,
    username: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Ambil hasil scan terakhir untuk user tertentu."""
    db_user = db.query(models.SocialUser).filter(
        models.SocialUser.username == username,
        models.SocialUser.platform == platform.lower()
    ).first()
    if not db_user:
        raise HTTPException(status_code=404, detail=f"User @{username} di {platform} belum pernah di-scan.")

    latest_summary = (
        db.query(models.AISummary)
        .filter(models.AISummary.user_id == db_user.user_id)
        .order_by(models.AISummary.generated_at.desc())
        .first()
    )
    activities = (
        db.query(models.Activity)
        .filter(models.Activity.user_id == db_user.user_id)
        .order_by(models.Activity.activity_id.desc())
        .limit(20)
        .all()
    )

    return {
        "status": "SUCCESS",
        "profile": {
            "username":    db_user.username,
            "full_name":   db_user.full_name,
            "bio":         db_user.bio,
            "followers":   db_user.followers,
            "following":   db_user.following,
            "verified":    db_user.verified,
            "profile_pic": db_user.profile_pic,
            "platform":    db_user.platform,
            "last_scanned": db_user.last_scanned.strftime("%Y-%m-%d %H:%M:%S") if db_user.last_scanned else ""
        },
        "summary_text": latest_summary.summary_text if latest_summary else "",
        "total_activities": len(activities),
        "activities": [
            {
                "activity_id":   a.activity_id,
                "action_type":   a.action_type,
                "content_text":  a.content_text,
                "post_url":      a.post_url,
                "ai_sentiment":  a.ai_sentiment,
                "ai_risk_score": a.ai_risk_score,
                "created_at":    a.created_at.strftime("%Y-%m-%d %H:%M:%S") if a.created_at else ""
            }
            for a in activities
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI IMAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/analyze-image", status_code=200)
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    caption: str = "",
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    if not is_image_file(image.filename):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (jpg, png, gif, webp).")

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, image.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    ai_result = analyze_image_real(file_path, caption)

    return {
        "status":      "SUCCESS",
        "filename":    image.filename,
        "ai_analysis": ai_result
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    uvicorn.run("main:app", host=host, port=port, reload=debug)
