"""
fetch_facebook.py - Facebook Scraper via Apify v2.0
Actors:
  - apify/facebook-posts-scraper   → posts publik, repost detection via sharedPost
  - apify/facebook-pages-scraper   → profile lengkap (bio, cover, lokasi, website, dll)
  - apify/facebook-comments-scraper → user_comments (filter: komentar oleh target sendiri)
  - apify/facebook-photos-scraper  → foto publik (best-effort, skip jika gagal)

Semua field profile yang diambil:
  name, profile_pic, cover_photo, bio, address/location, website, email,
  phone, creation_date, categories, about_me (teks panjang), rating,
  followers, likes (page likes), followings

Limitasi Facebook (tidak bisa diambil scraper apapun):
  - Pekerjaan / jabatan       → TIDAK tersedia di API publik
  - Pendidikan                → TIDAK tersedia di API publik
  - Daftar teman              → selalu privat
  - Grup yang diikuti         → privat kecuali grupnya sendiri publik
  - Halaman yang disukai      → privat
"""

import re
import asyncio
import time as time_module
from typing import Dict, List, Any
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _parse_fb_date(raw: Any) -> str:
    if not raw:
        return ""
    s = str(raw)
    # Unix timestamp
    if s.isdigit():
        try:
            return time_module.strftime("%Y-%m-%d", time_module.localtime(int(s)))
        except Exception:
            return s
    # ISO string → ambil 10 karakter pertama
    return s[:10] if len(s) >= 10 else s


def _build_empty_result(username: str, error: str) -> Dict:
    return {
        "status": "error", "platform": "facebook", "username": username,
        # profile
        "full_name": "", "bio": "", "about_me": "", "followers": 0,
        "following": 0, "page_likes": 0, "verified": False,
        "join_date": "", "profile_pic": "", "cover_photo": "",
        "location": "", "address": "", "website": "", "email": "",
        "phone": "", "categories": [], "rating": "", "rating_count": 0,
        # stats
        "total_posts": 0, "total_likes": 0, "total_comments": 0,
        "total_shares": 0, "has_text": False,
        # data
        "posts": [], "photos": [], "user_comments": [], "reposts": [],
        "scrape_errors": [error]
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNC SCRAPER — dijalankan di thread executor agar tidak blocking event loop
# ─────────────────────────────────────────────────────────────────────────────
def _facebook_scrape_sync(username: str, apify_token: str) -> Dict:
    if not apify_token:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        apify_token = os.getenv("APIFY_TOKEN", "")

    if not apify_token:
        return _build_empty_result(username, "APIFY_TOKEN tidak ditemukan di .env")

    try:
        from apify_client import ApifyClient
    except ImportError:
        return _build_empty_result(username, "apify_client tidak terinstall — jalankan: pip install apify-client")

    clean_username = username.replace("@", "").strip()
    profile_url = (
        clean_username
        if clean_username.startswith("http")
        else f"https://www.facebook.com/{clean_username}"
    )
    print(f"📡 [Facebook] Scraping {profile_url}...")

    try:
        client = ApifyClient(apify_token)

        # ══════════════════════════════════════════════════════════════════
        # RUN 1 — Posts (wajib, tidak boleh skip)
        # ══════════════════════════════════════════════════════════════════
        print(f"   ↳ [1/4] Mengambil posts via apify/facebook-posts-scraper...")
        run_posts = client.actor("apify/facebook-posts-scraper").call(
            run_input={
                "startUrls":          [{"url": profile_url}],
                "resultsLimit":       20,
                "proxyConfiguration": {"useApifyProxy": True},
            }
        )
        posts_dataset_id = getattr(run_posts, "default_dataset_id", None)
        if not posts_dataset_id:
            return _build_empty_result(clean_username, "Posts dataset ID tidak ditemukan")

        raw_items: List[Dict] = client.dataset(posts_dataset_id).list_items().items or []
        print(f"   ↳ Total raw items: {len(raw_items)}")

        if not raw_items:
            return _build_empty_result(
                clean_username,
                f"@{clean_username} tidak ditemukan, akun privat, atau Facebook memblokir akses."
            )

        # ══════════════════════════════════════════════════════════════════
        # RUN 2 — Profile detail lengkap
        # ══════════════════════════════════════════════════════════════════
        # Field yang dikonfirmasi dari apify/facebook-pages-scraper:
        #   title, intro, followers, likes, followings,
        #   profilePictureUrl, coverPhotoUrl, websites, website,
        #   email, phone, address, creation_date,
        #   about_me (dict: {text, urls}), categories, rating, ratingOverall,
        #   ratingCount, pageId, facebookId
        full_name    = ""
        bio          = ""
        about_me_txt = ""
        followers    = 0
        followings   = 0
        page_likes   = 0
        verified     = False
        profile_pic  = ""
        cover_photo  = ""
        location     = ""
        address      = ""
        website      = ""
        email        = ""
        phone        = ""
        categories: List[str] = []
        rating       = ""
        rating_count = 0
        join_date    = ""

        print(f"   ↳ [2/4] Mengambil profile detail via apify/facebook-pages-scraper...")
        try:
            run_profile = client.actor("apify/facebook-pages-scraper").call(
                run_input={
                    "startUrls":          [{"url": profile_url}],
                    "proxyConfiguration": {"useApifyProxy": True},
                }
            )
            ds_profile = getattr(run_profile, "default_dataset_id", None)
            if ds_profile:
                profile_items = client.dataset(ds_profile).list_items().items or []
                if profile_items:
                    pi = profile_items[0]

                    # Nama
                    full_name = _safe_str(pi.get("title") or pi.get("name"))

                    # Bio / intro singkat
                    bio = _safe_str(pi.get("intro") or pi.get("description"))

                    # About me (teks panjang)
                    about_raw = pi.get("about_me")
                    if isinstance(about_raw, dict):
                        about_me_txt = _safe_str(about_raw.get("text"))
                    elif isinstance(about_raw, str):
                        about_me_txt = about_raw.strip()

                    # Followers & likes
                    followers  = _safe_int(pi.get("followers"))
                    page_likes = _safe_int(pi.get("likes"))
                    followings = _safe_int(pi.get("followings"))

                    # Foto profil & cover
                    profile_pic = _safe_str(pi.get("profilePictureUrl") or pi.get("profilePic"))
                    cover_photo = _safe_str(pi.get("coverPhotoUrl"))

                    # Lokasi & alamat
                    address  = _safe_str(pi.get("address"))
                    # address biasanya format: "Jalan X, Kota, Negara url_maps"
                    # ambil bagian sebelum "http" jika ada
                    if "http" in address:
                        address = address.split("http")[0].strip().rstrip(",").strip()
                    location = address  # alias

                    # Kontak
                    # websites bisa list atau string
                    raw_websites = pi.get("websites") or []
                    if isinstance(raw_websites, list) and raw_websites:
                        website = _safe_str(raw_websites[0])
                    elif isinstance(raw_websites, str):
                        website = raw_websites.strip()
                    else:
                        website = _safe_str(pi.get("website"))

                    email = _safe_str(pi.get("email"))
                    phone = _safe_str(pi.get("phone"))

                    # Tanggal pembuatan
                    join_date = _safe_str(pi.get("creation_date"))

                    # Kategori
                    raw_cats = pi.get("categories") or []
                    if isinstance(raw_cats, list):
                        categories = [_safe_str(c) for c in raw_cats if c]
                    elif isinstance(raw_cats, str):
                        categories = [raw_cats.strip()]

                    # Rating (khusus halaman bisnis)
                    rating_overall = pi.get("ratingOverall")
                    rating_cnt     = _safe_int(pi.get("ratingCount"))
                    if rating_overall:
                        rating = f"{rating_overall}% recommend ({rating_cnt} reviews)"
                        rating_count = rating_cnt
                    elif pi.get("rating"):
                        rating = _safe_str(pi.get("rating"))

                    # Verified — facebook-pages-scraper tidak expose field ini
                    # tapi CONFIRMED_OWNER_LABEL menandakan halaman terverifikasi
                    verified = bool(
                        pi.get("CONFIRMED_OWNER_LABEL") or
                        pi.get("verified") or
                        pi.get("is_verified")
                    )

                    print(
                        f"   ↳ Profile: {full_name} | "
                        f"Followers: {followers:,} | "
                        f"Likes: {page_likes:,} | "
                        f"Cover: {'✓' if cover_photo else '✗'}"
                    )
        except Exception as e_profile:
            print(f"   ⚠️ Profile scraper error (non-fatal): {e_profile}")

        # Fallback jika profile scraper gagal — ambil dari data post
        if not full_name:
            first_user  = (raw_items[0].get("user") or {})
            full_name   = _safe_str(first_user.get("name") or raw_items[0].get("pageName") or clean_username)
            profile_pic = profile_pic or _safe_str(first_user.get("profilePic"))

        # ══════════════════════════════════════════════════════════════════
        # PARSE POSTS & REPOSTS dari raw_items
        # ══════════════════════════════════════════════════════════════════
        posts:   List[Dict] = []
        reposts: List[Dict] = []

        for item in raw_items:
            post_id  = _safe_str(item.get("postId") or item.get("id"))
            post_url = _safe_str(
                item.get("url") or item.get("topLevelUrl") or
                (f"https://www.facebook.com/{clean_username}/posts/{post_id}" if post_id else "")
            )
            if not post_url:
                continue

            text       = _safe_str(item.get("text"))
            likes      = _safe_int(item.get("likes"))
            comments   = _safe_int(item.get("comments"))
            shares     = _safe_int(item.get("shares"))
            plays      = _safe_int(item.get("viewsCount"))
            created_at = _parse_fb_date(item.get("time") or item.get("timestamp"))

            # Media
            media_list                  = item.get("media") or []
            image_url = video_url = cover_url = ""
            ocr_texts: List[str]        = []

            for media in media_list:
                if not isinstance(media, dict):
                    continue
                # Thumbnail / foto
                thumb = (
                    media.get("thumbnail") or
                    (media.get("photo_image") or {}).get("uri") or
                    media.get("url") or ""
                )
                if thumb and not cover_url:
                    cover_url = image_url = _safe_str(thumb)
                # Video
                if "Video" in _safe_str(media.get("__typename")) or media.get("videoUrl"):
                    video_url = _safe_str(media.get("videoUrl") or media.get("url"))
                # OCR dari scraper
                ocr = _safe_str(media.get("ocrText"))
                if ocr:
                    ocr_texts.append(ocr)

            # Media type
            if video_url or "reel" in post_url or "videos" in post_url:
                media_type = "video"
            elif media_list:
                media_type = "image"
            else:
                media_type = "text"

            hashtags = re.findall(r"#(\w+)", text)
            mentions = re.findall(r"@(\w+)", text)

            # ── Repost detection (multi-signal) ───────────────────────────
            # Signal 1: field sharedPost dari actor (paling akurat)
            shared_post_data = item.get("sharedPost")
            is_repost = bool(shared_post_data)

            # Signal 2: pageName berbeda dari target username
            # (post dari halaman lain yang di-share)
            if not is_repost:
                item_page = _safe_str(item.get("pageName")).lower()
                if item_page and item_page != clean_username.lower():
                    is_repost = True
                    # Buat sharedPost dummy dari data item
                    shared_post_data = {
                        "text":  text,
                        "user":  {"name": _safe_str(item.get("pageName"))},
                    }

            # Signal 3: facebookUrl berbeda dari profile_url
            # (scraper kadang return post dari halaman lain saat scrape profil)
            if not is_repost:
                item_fb_url = _safe_str(item.get("facebookUrl")).lower().rstrip("/")
                target_fb_url = profile_url.lower().rstrip("/")
                if (
                    item_fb_url and target_fb_url and
                    item_fb_url != target_fb_url and
                    f"facebook.com/{clean_username.lower()}" not in item_fb_url
                ):
                    is_repost = True
                    shared_post_data = shared_post_data or {"text": text, "user": {}}

            shared_post_data = shared_post_data or {}

            # Reactions detail
            reactions = {
                "like":  _safe_int(item.get("reactionLikeCount")),
                "love":  _safe_int(item.get("reactionLoveCount")),
                "care":  _safe_int(item.get("reactionCareCount")),
                "haha":  _safe_int(item.get("reactionHahaCount")),
                "wow":   _safe_int(item.get("reactionWowCount")),
                "sad":   _safe_int(item.get("reactionSadCount")),
                "angry": _safe_int(item.get("reactionAngryCount")),
            }

            post_entry = {
                "id":          post_id,
                "content":     text,
                "likes":       likes,
                "comments":    comments,
                "shares":      shares,
                "plays":       plays,
                "post_url":    post_url,
                "video_url":   video_url,
                "image_url":   image_url,
                "cover_url":   cover_url,
                "created_at":  created_at,
                "hashtags":    hashtags,
                "mentions":    mentions,
                "media_type":  media_type,
                "duration_sec": 0,
                "reactions":   reactions,
                "ocr_from_media": " | ".join(ocr_texts),
                "comments_data": [],
                "is_ad":       bool((item.get("pageAdLibrary") or {}).get("is_business_page_active")),
                "is_repost":   is_repost,
                "music":       {"title": "", "author": ""}
            }

            if is_repost:
                shared = shared_post_data
                shared_user = shared.get("user") or {}
                reposts.append({
                    "video_url":           post_url,
                    "original_author":     _safe_str(shared_user.get("name")),
                    "original_author_url": _safe_str(shared_user.get("url") or shared_user.get("profileUrl")),
                    "original_caption":    _safe_str(shared.get("text") or text)[:300],
                    "repost_date":         created_at,
                    "hashtags":            hashtags,
                    "metrics":             {"likes": likes, "comments": comments, "shares": shares, "plays": plays},
                    "reactions":           reactions,
                    "sentiment":           {"label": "NEUTRAL", "score": 0.5},
                    "content_description": "",
                })
            else:
                posts.append(post_entry)

        # ══════════════════════════════════════════════════════════════════
        # RUN 3 — Foto publik (best-effort)
        # ══════════════════════════════════════════════════════════════════
        photos: List[Dict] = []
        print(f"   ↳ [3/4] Mengambil foto via apify/facebook-photos-scraper...")
        try:
            run_photos = client.actor("apify/facebook-photos-scraper").call(
                run_input={
                    "startUrls":          [{"url": profile_url}],
                    "resultsLimit":       12,
                    "proxyConfiguration": {"useApifyProxy": True},
                }
            )
            ds_photos = getattr(run_photos, "default_dataset_id", None)
            if ds_photos:
                photo_items = client.dataset(ds_photos).list_items().items or []
                print(f"   ↳ Foto ditemukan: {len(photo_items)}")
                for ph in photo_items:
                    photo_url    = _safe_str(
                        ph.get("imageUrl") or ph.get("url") or ph.get("src")
                    )
                    photo_page   = _safe_str(ph.get("photoUrl") or ph.get("postUrl") or ph.get("facebookUrl"))
                    caption      = _safe_str(ph.get("caption") or ph.get("text"))
                    photo_date   = _parse_fb_date(ph.get("date") or ph.get("timestamp"))
                    photo_likes  = _safe_int(ph.get("likes") or ph.get("likesCount"))
                    photo_album  = _safe_str(ph.get("albumName") or ph.get("album"))

                    if not photo_url:
                        continue
                    photos.append({
                        "image_url":  photo_url,
                        "photo_page": photo_page,
                        "caption":    caption,
                        "album":      photo_album,
                        "likes":      photo_likes,
                        "date":       photo_date,
                    })
        except Exception as e_photos:
            print(f"   ⚠️ Photo scraper error (non-fatal): {e_photos}")

        # ══════════════════════════════════════════════════════════════════
        # RUN 4 — Komentar: ambil semua, pisah jadi comments_data & user_comments
        # ══════════════════════════════════════════════════════════════════
        # Output facebook-comments-scraper yang dikonfirmasi:
        #   profileName, profileId, profileUrl, text, date, likesCount,
        #   commentUrl, facebookUrl (= post url), threadingDepth, facebookId
        #
        # Strategi:
        #   - Semua komentar → simpan ke comments_data per post (max 5 per post)
        #   - Filter komentar oleh si target → simpan ke user_comments
        #
        # Filter identitas target (multi-signal):
        #   1. profileId == target_fb_id  (paling akurat)
        #   2. profileName == full_name   (exact match)
        #   3. profileName == clean_username (username sebagai nama)
        #   4. profileUrl mengandung clean_username (link profil target)
        #   5. partial match nama — ambil jika salah satu token nama ada
        # ══════════════════════════════════════════════════════════════════
        user_comments: List[Dict] = []

        # Ambil facebookId target dari post item pertama sebagai anchor
        target_fb_id = _safe_str(
            (raw_items[0].get("user") or {}).get("id") or
            raw_items[0].get("facebookId")
        )

        # Token nama target (untuk partial match)
        # misal full_name = "Anies Rasyid Baswedan" → tokens = {"anies", "baswedan"}
        name_tokens: set = set()
        if full_name:
            name_tokens = {t.lower() for t in full_name.split() if len(t) > 2}

        # URL profil target (bisa jadi https://www.facebook.com/username atau id)
        target_profile_urls = {
            profile_url.lower().rstrip("/"),
            f"https://www.facebook.com/{clean_username}".lower(),
        }
        if target_fb_id:
            target_profile_urls.add(f"https://www.facebook.com/{target_fb_id}".lower())

        # Gunakan 5 post teratas (hemat kredit Apify)
        top_post_urls = [p["post_url"] for p in posts[:5] if p["post_url"]]
        # Buat lookup post by url untuk isi comments_data
        post_url_index: Dict[str, int] = {p["post_url"]: i for i, p in enumerate(posts)}

        if top_post_urls:
            print(f"   ↳ [4/4] Mengambil komentar untuk {len(top_post_urls)} post teratas...")
            try:
                run_comments = client.actor("apify/facebook-comments-scraper").call(
                    run_input={
                        "startUrls":          [{"url": u} for u in top_post_urls],
                        "resultsLimit":       100,  # ambil lebih banyak agar filter bisa jalan
                        "proxyConfiguration": {"useApifyProxy": True},
                    }
                )
                ds_comments = getattr(run_comments, "default_dataset_id", None)
                if ds_comments:
                    comment_items = client.dataset(ds_comments).list_items().items or []
                    print(f"   ↳ Total komentar raw: {len(comment_items)}")

                    target_name_lower     = full_name.lower() if full_name else ""
                    target_username_lower = clean_username.lower()

                    # Counter comments_data per post (max 5 per post)
                    comments_per_post: Dict[str, int] = {}

                    for ci in comment_items:
                        if ci.get("errorCode"):
                            continue

                        c_author      = _safe_str(
                            ci.get("profileName") or
                            (ci.get("author") or {}).get("name") or
                            ci.get("authorName")
                        )
                        c_author_lower = c_author.lower()
                        c_author_id    = _safe_str(
                            ci.get("profileId") or
                            (ci.get("author") or {}).get("id")
                        )
                        c_author_url   = _safe_str(
                            ci.get("profileUrl") or
                            (ci.get("author") or {}).get("url") or ""
                        ).lower().rstrip("/")

                        c_text     = _safe_str(ci.get("text") or ci.get("message"))
                        c_url      = _safe_str(ci.get("commentUrl") or ci.get("url"))
                        c_post_url = _safe_str(ci.get("facebookUrl") or ci.get("inputUrl") or ci.get("postUrl"))
                        c_likes    = _safe_int(ci.get("likesCount") or ci.get("likes"))
                        c_date     = _parse_fb_date(ci.get("date") or ci.get("timestamp"))
                        is_reply   = _safe_int(ci.get("threadingDepth")) > 0

                        if not c_text:
                            continue

                        # ── Simpan ke comments_data per post (semua komentar) ──
                        # Cari post yang paling cocok dengan c_post_url
                        matched_post_url = ""
                        for pu in post_url_index:
                            if c_post_url and (
                                c_post_url == pu or
                                pu in c_post_url or
                                c_post_url in pu
                            ):
                                matched_post_url = pu
                                break

                        if matched_post_url:
                            idx = post_url_index[matched_post_url]
                            cnt = comments_per_post.get(matched_post_url, 0)
                            if cnt < 5:
                                posts[idx]["comments_data"].append({
                                    "author":     c_author,
                                    "content":    c_text,
                                    "likes":      c_likes,
                                    "date":       c_date,
                                    "is_reply":   is_reply,
                                    "comment_url": c_url,
                                })
                                comments_per_post[matched_post_url] = cnt + 1

                        # ── Filter: apakah komentar ini dari si target? ────────
                        # Multi-signal detection — salah satu match sudah cukup
                        is_target = False

                        # Signal 1: Facebook ID exact match (paling reliable)
                        if target_fb_id and c_author_id and c_author_id == target_fb_id:
                            is_target = True

                        # Signal 2: Nama persis sama (case-insensitive)
                        elif target_name_lower and c_author_lower == target_name_lower:
                            is_target = True

                        # Signal 3: Username sama dengan nama komentar
                        elif target_username_lower and c_author_lower == target_username_lower:
                            is_target = True

                        # Signal 4: URL profil komentar mengandung username target
                        elif c_author_url and target_username_lower in c_author_url:
                            is_target = True

                        # Signal 5: URL profil komentar match dengan target profile URL
                        elif c_author_url and c_author_url in target_profile_urls:
                            is_target = True

                        # Signal 6: Partial name match — minimal 2 token nama cocok
                        elif name_tokens:
                            c_tokens = {t.lower() for t in c_author.split() if len(t) > 2}
                            overlap = name_tokens & c_tokens
                            if len(overlap) >= min(2, len(name_tokens)):
                                is_target = True

                        if not is_target:
                            continue

                        # Cari caption post yang dikomentari
                        post_caption = ""
                        if matched_post_url:
                            post_caption = posts[post_url_index[matched_post_url]]["content"][:100]
                        else:
                            # fallback: cari manual
                            for p in posts:
                                if p["post_url"] and c_post_url and (
                                    c_post_url == p["post_url"] or
                                    p["post_url"] in c_post_url
                                ):
                                    post_caption = p["content"][:100]
                                    break

                        user_comments.append({
                            "post_url":     c_post_url or c_url,
                            "post_caption": post_caption,
                            "post_author":  clean_username,
                            "content":      c_text,
                            "likes":        c_likes,
                            "is_reply":     is_reply,
                            "reply_to":     "",
                            "created_at":   c_date,
                            "sentiment":    {"label": "NEUTRAL", "score": 0.5}
                        })

                    print(f"   ↳ user_comments (by target): {len(user_comments)}")
                    total_cd = sum(len(p["comments_data"]) for p in posts)
                    print(f"   ↳ comments_data (others): {total_cd} across {len(posts)} posts")

            except Exception as e_comments:
                print(f"   ⚠️ Comment scraper error (non-fatal): {e_comments}")

        # ── Trim & hitung total ───────────────────────────────────────────
        user_comments = user_comments[:15]
        reposts       = reposts[:10]
        photos        = photos[:12]

        total_likes    = sum(p["likes"]    for p in posts)
        total_comments = sum(p["comments"] for p in posts)
        total_shares   = sum(p["shares"]   for p in posts)

        print(
            f"   ↳ Posts: {len(posts)} | "
            f"Photos: {len(photos)} | "
            f"user_comments: {len(user_comments)} | "
            f"reposts: {len(reposts)}"
        )

        return {
            "status":      "success",
            "platform":    "facebook",
            "username":    clean_username,

            # ── Profile lengkap ──────────────────────────────────────────
            "full_name":   full_name,
            "bio":         bio,           # intro/deskripsi singkat
            "about_me":    about_me_txt,  # teks panjang dari tab About
            "followers":   followers,
            "following":   followings,
            "page_likes":  page_likes,    # jumlah likes halaman (berbeda dari followers)
            "verified":    verified,
            "join_date":   join_date,     # creation_date
            "profile_pic": profile_pic,
            "cover_photo": cover_photo,   # foto sampul
            "location":    location,      # kota/lokasi
            "address":     address,       # alamat lengkap
            "website":     website,
            "email":       email,
            "phone":       phone,
            "categories":  categories,    # kategori halaman, misal ["Page", "News"]
            "rating":      rating,        # % recommend (halaman bisnis)
            "rating_count": rating_count,

            # ── Statistik ────────────────────────────────────────────────
            "total_posts":    len(posts),
            "total_likes":    total_likes,
            "total_comments": total_comments,
            "total_shares":   total_shares,
            "has_text":       any(p["content"] for p in posts),

            # ── Data ─────────────────────────────────────────────────────
            "posts":         posts,
            "photos":        photos,
            "user_comments": user_comments,
            "reposts":       reposts,
            "scrape_errors": []
        }

    except Exception as e:
        import traceback
        err = f"Facebook scrape error: {str(e)}"
        print(f"   ❌ {err}")
        print(traceback.format_exc())
        return _build_empty_result(clean_username, err)


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC WRAPPER — dipanggil dari scraper_engine.py
# ─────────────────────────────────────────────────────────────────────────────
async def _facebook_scrape_real(username: str, apify_token: str = "") -> Dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(_facebook_scrape_sync, username, apify_token)
    )
