"""
scraper_engine.py - Social Radar Scraper Engine v12.0
- run_crawler sekarang ASYNC — langsung await fetch tanpa asyncio.run()
- Tidak ada lagi nested event loop issue dengan FastAPIyt-dlp

"""

import os
import re
import time
import logging
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv()
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _empty_profile(username: str, platform: str) -> Dict[str, Any]:
    return {
        "username": username,
        "full_name": "",
        "bio": "",
        "followers": 0,
        "following": 0,
        "verified": False,
        "profile_pic": "",
        "total_posts": 0,
        "total_likes": 0,
        "platform": platform,
        "last_updated": time.strftime("%Y-%m-%d")
    }


def _normalise_post(raw: Dict, platform: str, username: str) -> Dict[str, Any]:
    text = raw.get("content") or raw.get("text") or ""

    media_type = raw.get("media_type") or ""
    if media_type in ("video", "reel"):
        content_type = "video"
    elif media_type in ("image", "photo"):
        content_type = "image"
    elif media_type == "carousel":
        content_type = "carousel"
    else:
        content_type = "video" if platform == "tiktok" else "image"

    if text:
        clean_text = re.sub(r"[#@]\w+", "", text).strip()
        if clean_text:
            content_description = clean_text[:300]
        else:
            hashtags_str = " ".join(f"#{h}" for h in (raw.get("hashtags") or [])[:5])
            content_description = f"Konten {content_type} {hashtags_str}".strip()
    else:
        content_description = f"Konten {content_type} tanpa caption"

    likes    = int(raw.get("likes")    or 0)
    comments = int(raw.get("comments") or 0)
    shares   = int(raw.get("shares")   or raw.get("reposts") or 0)
    plays    = int(raw.get("plays")    or 0)

    post_url = raw.get("post_url") or raw.get("url") or ""
    if not post_url:
        pid = raw.get("id") or ""
        if platform == "tiktok" and pid:
            post_url = f"https://www.tiktok.com/@{username}/video/{pid}"
        elif platform == "instagram" and pid:
            post_url = f"https://www.instagram.com/p/{pid}/"

    created_at = raw.get("created_at") or raw.get("timestamp") or time.strftime("%Y-%m-%d")

    # OCR yang sudah tersedia dari scraper (khusus Facebook — actor sudah sertakan ocrText)
    ocr_from_scraper = raw.get("ocr_from_media") or ""

    # combined_text default = caption; akan diperbarui main.py setelah media extraction
    # Untuk Facebook, langsung gabungkan OCR dari actor karena sudah tersedia
    combined_default = " | ".join(filter(None, [text[:700], ocr_from_scraper])) if ocr_from_scraper else text[:700]

    return {
        "action_type":         "POST",
        "text":                text[:700],
        "post_url":            post_url,
        "content_type":        content_type,
        "content_description": content_description,
        "image_url":           raw.get("image_url") or raw.get("cover_url") or "",
        "video_url":           raw.get("video_url") or "",
        "metrics": {
            "likes":      likes,
            "comments":   comments,
            "shares":     shares,
            "plays":      plays,
            "quotes":     int(raw.get("quotes")    or 0),
            "bookmarks":  int(raw.get("bookmarks") or 0),
        },
        "sentiment":    {"label": "NEUTRAL", "score": 0.5},
        "hashtags":     raw.get("hashtags")     or [],
        "mentions":     raw.get("mentions")     or [],
        "urls":         raw.get("urls")         or [],   # URL dalam tweet/post
        "music":        raw.get("music")        or {},
        "location":     raw.get("location")     or "",
        "duration_sec": raw.get("duration_sec") or 0,
        "comments_data": [],
        "is_ad":        bool(raw.get("is_ad") or raw.get("is_sponsored") or False),
        "is_reply":     bool(raw.get("is_reply")  or False),
        "is_repost":    bool(raw.get("is_repost") or False),
        "is_quote":     bool(raw.get("is_quote")  or False),
        "created_at":   str(created_at),
        # ── X/Twitter spesifik ────────────────────────────────────────────
        "lang":             raw.get("lang")             or "",  # bahasa tweet
        "source":           raw.get("source")           or "",  # Twitter Web App / iPhone / Android
        "conversation_id":  raw.get("conversation_id")  or "",
        "gif_url":          raw.get("gif_url")          or "",
        "all_media":        raw.get("all_media")        or [],  # list [{url, type}]
        "referenced_tweet_url": raw.get("referenced_tweet_url") or "",
        "in_reply_to_user":     raw.get("in_reply_to_user")     or "",
        # ── Media extraction fields ───────────────────────────────────────
        "caption_text":       text[:700],
        "ocr_text":           ocr_from_scraper,
        "audio_transcript":   "",
        "visual_description": "",
        "combined_text":      combined_default,
        "extraction_errors":  [],
    }


def _normalise_comments(raw_comments: List[Dict]) -> List[Dict]:
    result = []
    for c in raw_comments:
        if not isinstance(c, dict):
            continue
        result.append({
            "post_url":     c.get("post_url")     or "",
            "post_caption": c.get("post_caption") or "",
            "content":      c.get("content")      or c.get("text") or "",
            "likes":        int(c.get("likes")    or 0),
            "replies":      int(c.get("replies")  or 0),
            "quotes":       int(c.get("quotes")   or 0),
            "is_reply":     bool(c.get("is_reply", False)),
            "reply_to":     c.get("reply_to")     or "",
            "created_at":   str(c.get("created_at") or ""),
            "lang":         c.get("lang")         or "",   # X: bahasa reply
            "source":       c.get("source")       or "",   # X: perangkat
            "sentiment":    c.get("sentiment") or {"label": "NEUTRAL", "score": 0.5}
        })
    return result


def _build_result(profile, activities, user_comments, reposts, scrape_errors, status_str):
    return {
        "profile":       profile,
        "activities":    activities,
        "user_comments": user_comments,   # komentar target di postingan sendiri
        "reposts":       reposts,
        "scrape_errors": scrape_errors
    }, status_str


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: run_crawler — sekarang ASYNC
# ─────────────────────────────────────────────────────────────────────────────
async def run_crawler(
    platform: str,
    target: str,
    scan_type: str = "USER_PROFILE"
) -> Tuple[Dict[str, Any], str]:
    """
    Scraper Engine v12.0 — async, tidak pakai asyncio.run().
    Dipanggil dengan: await run_crawler(...)
    """
    p = platform.lower().strip()
    clean_target = target.replace("@", "").strip()

    logger.info(f"🚀 [{p.upper()}] Mulai scraping @{clean_target}")

    profile = _empty_profile(clean_target, p)

    try:
        # ── Pilih & panggil scraper ────────────────────────────────────────
        raw: Dict = {}

        if p == "tiktok":
            from fetch_tiktok import _tiktok_scrape_real
            raw = await _tiktok_scrape_real(clean_target, APIFY_TOKEN)

        elif p == "instagram":
            from fetch_instagram import _instagram_scrape_real
            raw = await _instagram_scrape_real(clean_target, APIFY_TOKEN)

        elif p in ("x", "twitter"):
            try:
                from fetch_tweet import _twitter_scrape_real
                raw = await _twitter_scrape_real(clean_target, APIFY_TOKEN)
            except ImportError:
                raw = {"status": "error", "scrape_errors": ["fetch_tweet tidak tersedia"]}

        elif p == "facebook":
            try:
                from fetch_facebook import _facebook_scrape_real
                raw = await _facebook_scrape_real(clean_target, APIFY_TOKEN)
            except ImportError:
                raw = {"status": "error", "scrape_errors": ["fetch_facebook tidak tersedia"]}

        else:
            return _build_result(profile, [], [], [], [f"Platform '{p}' tidak didukung"], "UNSUPPORTED_PLATFORM")

        # ── Cek status ────────────────────────────────────────────────────
        scrape_errors = raw.get("scrape_errors") or []

        if raw.get("status") == "error":
            logger.error(f"❌ Scrape error: {scrape_errors}")
            return _build_result(profile, [], [], [], scrape_errors, "SCRAPE_ERROR")

        # ── Isi profile ───────────────────────────────────────────────────
        profile["username"]     = raw.get("username")    or clean_target
        profile["full_name"]    = raw.get("full_name")   or clean_target
        profile["bio"]          = raw.get("bio")         or ""
        profile["followers"]    = int(raw.get("followers") or 0)
        profile["following"]    = int(raw.get("following") or 0)
        profile["verified"]     = bool(raw.get("verified") or False)
        profile["profile_pic"]  = raw.get("profile_pic") or ""
        profile["total_posts"]  = int(raw.get("total_posts") or len(raw.get("posts") or []))
        profile["total_likes"]  = int(raw.get("total_likes") or 0)
        profile["platform"]     = p
        profile["last_updated"] = time.strftime("%Y-%m-%d")

        # ── Field tambahan per platform ───────────────────────────────────
        if p == "facebook":
            profile["cover_photo"]  = raw.get("cover_photo")  or ""
            profile["about_me"]     = raw.get("about_me")     or ""
            profile["location"]     = raw.get("location")     or raw.get("address") or ""
            profile["address"]      = raw.get("address")      or ""
            profile["website"]      = raw.get("website")      or ""
            profile["email"]        = raw.get("email")        or ""
            profile["phone"]        = raw.get("phone")        or ""
            profile["categories"]   = raw.get("categories")   or []
            profile["rating"]       = raw.get("rating")       or ""
            profile["rating_count"] = int(raw.get("rating_count") or 0)
            profile["page_likes"]   = int(raw.get("page_likes") or 0)
            profile["join_date"]    = raw.get("join_date")    or ""

        elif p in ("x", "twitter"):
            profile["user_id"]       = raw.get("user_id")       or ""
            profile["banner_image"]  = raw.get("banner_image")  or ""
            profile["location"]      = raw.get("location")      or ""
            profile["website"]       = raw.get("website")       or ""
            profile["join_date"]     = raw.get("join_date")     or ""
            profile["tweet_count"]   = int(raw.get("tweet_count") or 0)
            # Field yang tidak tersedia karena limitasi Twitter/actor
            profile["data_limitations"] = raw.get("data_limitations") or []

        logger.info(
            f"✅ {profile['full_name']} | "
            f"Followers: {profile['followers']:,} | "
            f"Verified: {profile['verified']}"
        )

        # ── Normalise posts ───────────────────────────────────────────────
        raw_posts  = raw.get("posts") or []
        activities = [_normalise_post(rp, p, clean_target) for rp in raw_posts[:20]]

        logger.info(f"📊 Activities: {len(activities)}")

        if profile["total_likes"] == 0 and activities:
            profile["total_likes"] = sum(a["metrics"]["likes"] for a in activities)

        user_comments = _normalise_comments(raw.get("user_comments") or [])
        reposts       = raw.get("reposts") or []
        # Facebook photos — platform lain tidak ada field ini
        fb_photos     = raw.get("photos") or []

        status_str = "LIVE_DATA" if activities else "NO_POSTS_FOUND"
        logger.info(f"🎉 Selesai | {status_str}")

        result_data, status = _build_result(
            profile, activities,
            user_comments,
            reposts, scrape_errors, status_str
        )
        # Inject photos ke result jika ada
        if fb_photos:
            result_data["photos"] = fb_photos
        return result_data, status

    except Exception as e:
        import traceback
        logger.error(f"🚨 run_crawler error: {e}\n{traceback.format_exc()}")
        return _build_result(profile, [], [], [], [str(e)], "ERROR")
