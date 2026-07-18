"""
fetch_instagram.py - Instagram Scraper via Apify
Actor utama : apify/instagram-scraper
Actor komentar: apify/instagram-comments-scraper
apify_client 3.x + async-safe

Fitur:
- Profile lengkap (followers, bio, verified, profile_pic)
- 20 post terbaru dengan metrics lengkap
- Komentar per post (5 post teratas)
- owner_replies: komentar yang dibuat target di postingannya sendiri
- Deteksi repost dari caption (credit, via @, repost)
- content_analysis: topik, sentimen, deskripsi — diisi oleh ai_engine
"""

import os
import re
import time
import asyncio
from typing import Dict, List, Any
from functools import partial


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default


def _build_empty_result(username: str, error: str) -> Dict:
    return {
        "status": "error",
        "platform": "instagram",
        "username": username,
        "full_name": "",
        "bio": "",
        "followers": 0,
        "following": 0,
        "likes": 0,
        "verified": False,
        "join_date": "",
        "profile_pic": "",
        "total_posts": 0,
        "posts": [],
        "user_comments": [],
        "reposts": [],
        "total_likes": 0,
        "total_comments": 0,
        "total_shares": 0,
        "has_text": False,
        "scrape_errors": [error]
    }


def _parse_timestamp(ts: Any) -> str:
    if not ts:
        return ""
    s = str(ts)
    if s.isdigit():
        try:
            return time.strftime("%Y-%m-%d", time.localtime(int(s)))
        except Exception:
            return s
    return s[:10] if len(s) >= 10 else s


def _detect_repost_from_caption(text: str) -> Dict:
    """
    Deteksi apakah postingan ini adalah repost/credit dari akun lain.
    Instagram tidak punya fitur repost native, tapi konten sering pakai pola:
    - "credit: @username"
    - "via @username"
    - "repost @username"
    - "📸 @username"
    - "@username 's video"
    """
    if not text:
        return {"is_repost": False, "original_author": ""}

    patterns = [
        r"credit[:\s]+@(\w+)",
        r"via\s+@(\w+)",
        r"repost[:\s]+@(\w+)",
        r"re-?post[:\s]+@(\w+)",
        r"source[:\s]+@(\w+)",
        r"from[:\s]+@(\w+)",
        r"📸\s*@(\w+)",
        r"📹\s*@(\w+)",
        r"🎥\s*@(\w+)",
        r"cr[:\s]+@(\w+)",
        r"©\s*@(\w+)",
    ]

    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return {"is_repost": True, "original_author": match.group(1)}

    return {"is_repost": False, "original_author": ""}


def _parse_posts_from_items(items: List[Dict], clean_username: str) -> List[Dict]:
    """Parse list item dari Apify dataset → list post standar."""
    posts: List[Dict] = []
    for item in items:
        if item.get("type") == "user" or item.get("itemType") == "user":
            continue

        item_owner = (
            item.get("ownerUsername") or
            (item.get("owner") or {}).get("username") or
            ""
        ).lower()
        if item_owner and item_owner != clean_username.lower():
            continue

        post_id   = str(item.get("id") or item.get("postId") or "")
        shortcode = item.get("shortCode") or item.get("shortcode") or post_id
        if not post_id and not shortcode:
            continue

        text: str = (
            item.get("caption") or
            item.get("text") or
            item.get("alt") or
            item.get("accessibility_caption") or
            ""
        ).strip()

        likes    = _safe_int(item.get("likesCount")    or item.get("likes"))
        comments = _safe_int(item.get("commentsCount") or item.get("comments"))

        plays = _safe_int(
            item.get("videoViewCount") or
            item.get("videoViews")     or
            item.get("videoPlayCount") or
            item.get("video_view_count") or
            item.get("playCount")      or
            item.get("views")
        )

        shares = _safe_int(
            item.get("sharesCount") or
            item.get("shareCount")  or
            item.get("repostCount") or
            item.get("shares")
        )

        post_url: str = (
            item.get("url") or
            item.get("postUrl") or
            (f"https://www.instagram.com/p/{shortcode}/" if shortcode else "")
        )
        image_url: str = (
            item.get("displayUrl") or
            item.get("imageUrl") or
            item.get("thumbnailUrl") or
            ""
        ) or ""
        video_url: str = item.get("videoUrl") or item.get("videoSrc") or ""

        type_raw = item.get("type") or item.get("mediaType") or ""
        if type_raw in ("Video", "video") or item.get("isVideo") or video_url:
            media_type = "video"
        elif type_raw in ("Sidecar", "sidecar", "GraphSidecar"):
            media_type = "carousel"
        else:
            media_type = "image"

        timestamp = _parse_timestamp(
            item.get("timestamp") or item.get("takenAtTimestamp")
        )

        # Hashtags
        hashtags: List[str] = re.findall(r"#(\w+)", text)
        raw_hashtags = item.get("hashtags") or []
        if raw_hashtags and isinstance(raw_hashtags, list):
            parsed_tags = []
            for h in raw_hashtags:
                t = h.get("name") or h.get("hashtag") or h.get("tag") or "" if isinstance(h, dict) else h.lstrip("#") if isinstance(h, str) else ""
                if t:
                    parsed_tags.append(t)
            if parsed_tags:
                hashtags = parsed_tags

        # Mentions
        mentions: List[str] = re.findall(r"@(\w+)", text)
        raw_mentions = item.get("mentions") or item.get("taggedUsers") or []
        if raw_mentions and isinstance(raw_mentions, list):
            parsed_mentions = []
            for m in raw_mentions:
                u = m.get("username") or m.get("name") or "" if isinstance(m, dict) else m.lstrip("@") if isinstance(m, str) else ""
                if u:
                    parsed_mentions.append(u)
            if parsed_mentions:
                mentions = parsed_mentions

        duration_sec = _safe_int(
            item.get("videoDuration") or
            item.get("duration") or
            item.get("video_duration") or
            (item.get("videoMeta") or {}).get("duration")
        )

        # Komentar nested dari apify/instagram-scraper posts output
        # Field: latestComments (list of dicts), firstComment (string)
        raw_comments_data = (
            item.get("latestComments") or
            item.get("previewComments") or
            []
        )
        if isinstance(raw_comments_data, str):
            raw_comments_data = []

        comments_data: List[Dict] = []

        # Tambahkan firstComment jika ada (string tunggal)
        first_comment_str = item.get("firstComment") or ""
        if first_comment_str and isinstance(first_comment_str, str) and first_comment_str.strip():
            comments_data.append({
                "username":   "unknown",
                "content":    first_comment_str.strip(),
                "likes":      0,
                "created_at": "",
                "is_reply":   False,
                "reply_to":   ""
            })

        # Parse latestComments (bisa sampai ~20 dari actor)
        for c in raw_comments_data[:20]:
            if isinstance(c, dict):
                c_content = c.get("text") or c.get("comment") or c.get("content") or ""
                if not c_content:
                    continue
                c_likes = _safe_int(
                    c.get("likesCount") or c.get("likeCount") or
                    c.get("likes") or c.get("diggCount")
                )
                c_user = (
                    c.get("ownerUsername") or c.get("username") or
                    (c.get("owner") or {}).get("username") or "unknown"
                )
                comments_data.append({
                    "username":   c_user,
                    "content":    c_content,
                    "likes":      c_likes,
                    "created_at": _parse_timestamp(c.get("timestamp") or c.get("created_at")),
                    "is_reply":   bool(c.get("isReply") or False),
                    "reply_to":   c.get("replyToUsername") or ""
                })

        location: str = (
            item.get("locationName") or
            (item.get("location") or {}).get("name") or
            ""
        )

        # Deteksi repost dari caption
        repost_info = _detect_repost_from_caption(text)

        posts.append({
            "id":            post_id or shortcode,
            "content":       text,
            "likes":         likes,
            "comments":      comments,
            "shares":        shares,
            "plays":         plays,
            "post_url":      post_url,
            "video_url":     video_url,
            "image_url":     image_url,
            "cover_url":     image_url,
            "created_at":    timestamp,
            "hashtags":      hashtags,
            "mentions":      mentions,
            "media_type":    media_type,
            "duration_sec":  duration_sec,
            "location":      location,
            "comments_data": comments_data,
            "is_sponsored":  bool(item.get("isSponsored") or False),
            "is_repost":     repost_info["is_repost"],
            "repost_author": repost_info["original_author"],
            "music": {
                "title":  ((item.get("musicInfo") or {}).get("music_info") or {}).get("title")  or "",
                "author": ((item.get("musicInfo") or {}).get("music_info") or {}).get("author") or "",
            }
        })
    return posts


def _instagram_scrape_sync(username: str, apify_token: str) -> Dict:
    if not apify_token:
        from dotenv import load_dotenv
        load_dotenv()
        apify_token = os.getenv("APIFY_TOKEN", "")

    if not apify_token:
        return _build_empty_result(username, "APIFY_TOKEN tidak ditemukan di .env")

    try:
        from apify_client import ApifyClient
    except ImportError:
        return _build_empty_result(username, "apify_client tidak terinstall")

    clean_username = username.replace("@", "").strip()
    profile_url    = f"https://www.instagram.com/{clean_username}/"
    print(f"📡 [Instagram] Scraping @{clean_username}...")

    from dotenv import load_dotenv
    load_dotenv()
    ig_session = os.getenv("INSTAGRAM_SESSION", "")

    client = ApifyClient(apify_token)

    # ── Attempt 1: resultsType = posts ────────────────────────────────────
    items: List[Dict] = []
    run_input_posts = {
        "directUrls":   [profile_url],
        "resultsType":  "posts",
        "resultsLimit": 20,
        "addParentData": True,
        "maxComments":  10,           # ← ambil komentar nested per post
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"]
        },
    }
    if ig_session:
        run_input_posts["sessionCookies"] = [
            {"name": "sessionid", "value": ig_session, "domain": ".instagram.com"}
        ]

    try:
        print(f"   ↳ [Attempt 1] resultsType=posts...")
        run1 = client.actor("apify/instagram-scraper").call(run_input=run_input_posts)
        ds1  = getattr(run1, "default_dataset_id", None)
        if ds1:
            items = client.dataset(ds1).list_items().items or []
        print(f"   ↳ Attempt 1 hasil: {len(items)} item")
    except Exception as e:
        print(f"   ⚠️ Attempt 1 error: {e}")

    # ── Attempt 2: resultsType = details ─────────────────────────────────
    profile_raw: Dict = {}
    if not items:
        run_input_details = {
            "directUrls":  [profile_url],
            "resultsType": "details",
            "resultsLimit": 1,
            "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
        }
        if ig_session:
            run_input_details["sessionCookies"] = [
                {"name": "sessionid", "value": ig_session, "domain": ".instagram.com"}
            ]
        try:
            print(f"   ↳ [Attempt 2] resultsType=details...")
            run2 = client.actor("apify/instagram-scraper").call(run_input=run_input_details)
            ds2  = getattr(run2, "default_dataset_id", None)
            if ds2:
                detail_items = client.dataset(ds2).list_items().items or []
                if detail_items:
                    profile_raw = detail_items[0]
                    latest = (
                        profile_raw.get("latestPosts") or
                        (profile_raw.get("edge_owner_to_timeline_media") or {}).get("edges") or []
                    )
                    for edge in latest:
                        node = edge.get("node") or edge if isinstance(edge, dict) else {}
                        node["ownerUsername"]  = profile_raw.get("username", clean_username)
                        node["ownerFullName"]  = profile_raw.get("fullName", "")
                        node["followersCount"] = profile_raw.get("followersCount", 0)
                        node["followsCount"]   = profile_raw.get("followsCount", 0)
                        node["biography"]      = profile_raw.get("biography", "")
                        node["verified"]       = profile_raw.get("isVerified", False)
                        node["profilePicUrl"]  = profile_raw.get("profilePicUrl", "")
                        items.append(node)
            print(f"   ↳ Attempt 2 hasil: {len(items)} item")
        except Exception as e:
            print(f"   ⚠️ Attempt 2 error: {e}")

    if not items and not profile_raw:
        return _build_empty_result(
            clean_username,
            f"@{clean_username} tidak ditemukan, akun privat, atau Instagram memblokir request."
        )

    # ── Parse profile ─────────────────────────────────────────────────────
    first = items[0] if items else profile_raw
    for candidate in items:
        candidate_owner = (
            candidate.get("ownerUsername") or
            (candidate.get("owner") or {}).get("username") or ""
        ).lower()
        if candidate_owner == clean_username.lower():
            first = candidate
            break

    owner: Dict = first.get("owner") or first.get("ownerProfile") or profile_raw or {}

    full_name: str  = (first.get("ownerFullName") or owner.get("fullName") or owner.get("full_name") or profile_raw.get("fullName") or clean_username)
    bio: str        = (first.get("biography") or owner.get("biography") or profile_raw.get("biography") or "")
    followers: int  = _safe_int(first.get("followersCount") or owner.get("followersCount") or profile_raw.get("followersCount") or (owner.get("edge_followed_by") or {}).get("count"))
    following: int  = _safe_int(first.get("followsCount") or owner.get("followsCount") or profile_raw.get("followsCount") or (owner.get("edge_follow") or {}).get("count"))
    verified: bool  = bool(first.get("verified") or owner.get("is_verified") or owner.get("verified") or profile_raw.get("isVerified") or False)
    profile_pic: str = (first.get("profilePicUrl") or owner.get("profilePicUrl") or owner.get("profile_pic_url_hd") or profile_raw.get("profilePicUrl") or "") or ""
    total_ig_posts: int = _safe_int(first.get("postsCount") or owner.get("postsCount") or profile_raw.get("postsCount") or (owner.get("edge_owner_to_timeline_media") or {}).get("count"))

    print(f"   ↳ Profile: {full_name} | Followers: {followers:,} | Verified: {verified}")

    # ── Parse posts ───────────────────────────────────────────────────────
    posts = _parse_posts_from_items(items, clean_username)

    if not posts:
        print(f"   ⚠️ Tidak ada postingan yang bisa diparse")
        return {
            "status": "success", "platform": "instagram",
            "username": clean_username, "full_name": full_name,
            "bio": bio, "followers": followers, "following": following,
            "likes": 0, "verified": verified, "join_date": "",
            "profile_pic": profile_pic, "total_posts": total_ig_posts,
            "posts": [], "user_comments": [], "reposts": [],
            "total_likes": 0, "total_comments": 0, "total_shares": 0,
            "has_text": False,
            "scrape_errors": ["Posts tidak tersedia — akun mungkin privat"]
        }

    # ── Ambil komentar lebih banyak via apify/instagram-comments-scraper ──
    # Ambil untuk 5 post terbaru agar coverage lebih luas
    top_post_urls = [p["post_url"] for p in posts[:5] if p["post_url"]]

    # Buat mapping shortcode → post_url untuk matching komentar
    shortcode_to_url: Dict[str, str] = {}
    for p in posts[:5]:
        purl = p["post_url"]
        # Extract shortcode dari URL: https://www.instagram.com/p/SHORTCODE/
        sc_match = re.search(r"/p/([A-Za-z0-9_-]+)/?", purl)
        if sc_match:
            shortcode_to_url[sc_match.group(1)] = purl

    comments_by_post: Dict[str, List[Dict]] = {}

    if top_post_urls:
        print(f"   ↳ Mengambil komentar untuk {len(top_post_urls)} post teratas...")
        try:
            comment_input = {
                "directUrls":            top_post_urls,
                "resultsType":           "comments",
                "resultsLimit":          30,
                "includeNestedComments": True,
            }
            # Session meningkatkan keberhasilan signifikan
            if ig_session:
                comment_input["sessionCookies"] = [
                    {"name": "sessionid", "value": ig_session, "domain": ".instagram.com"}
                ]

            comment_run = client.actor("apify/instagram-scraper").call(
                run_input=comment_input
            )
            c_dataset_id = getattr(comment_run, "default_dataset_id", None)
            if c_dataset_id:
                comment_items = client.dataset(c_dataset_id).list_items().items or []
                print(f"   ↳ Total raw komentar: {len(comment_items)}")

                for c_item in comment_items:
                    if c_item.get("errorCode"):
                        continue

                    # Output apify/instagram-scraper comments:
                    # postUrl, commentUrl, id, text, ownerUsername, timestamp, likesCount
                    # Cari post_url yang sesuai dari field postUrl
                    post_url_key = ""

                    # Method 1: postUrl langsung dari item (field utama di output actor ini)
                    item_post_url = c_item.get("postUrl") or ""
                    if item_post_url:
                        # Normalise: pastikan format sama dengan top_post_urls
                        item_post_url = item_post_url.rstrip("/") + "/"
                        for pu in top_post_urls:
                            if item_post_url == pu.rstrip("/") + "/":
                                post_url_key = pu
                                break
                        # Jika tidak cocok persis, coba shortcode matching
                        if not post_url_key:
                            sc_m = re.search(r"/p/([A-Za-z0-9_-]+)/?", item_post_url)
                            if sc_m and sc_m.group(1) in shortcode_to_url:
                                post_url_key = shortcode_to_url[sc_m.group(1)]

                    # Method 2: shortCode dari commentUrl
                    if not post_url_key:
                        comment_url = c_item.get("commentUrl") or ""
                        sc_m = re.search(r"/p/([A-Za-z0-9_-]+)/", comment_url)
                        if sc_m and sc_m.group(1) in shortcode_to_url:
                            post_url_key = shortcode_to_url[sc_m.group(1)]

                    # Method 3: shortCode field langsung
                    if not post_url_key:
                        sc = c_item.get("shortCode") or c_item.get("shortcode") or ""
                        if sc and sc in shortcode_to_url:
                            post_url_key = shortcode_to_url[sc]

                    # Method 4: fallback ke post pertama
                    if not post_url_key and top_post_urls:
                        post_url_key = top_post_urls[0]

                    if not post_url_key:
                        continue

                    if post_url_key not in comments_by_post:
                        comments_by_post[post_url_key] = []

                    c_user = (
                        c_item.get("ownerUsername") or
                        c_item.get("username") or
                        (c_item.get("owner") or {}).get("username") or
                        "unknown"
                    )
                    c_text  = c_item.get("text") or c_item.get("comment") or ""
                    c_likes = _safe_int(c_item.get("likesCount") or c_item.get("likeCount"))
                    c_time  = _parse_timestamp(c_item.get("timestamp"))
                    is_reply = bool(c_item.get("parentCommentId") or c_item.get("isReply"))
                    reply_to = c_item.get("parentOwnerUsername") or ""

                    # Simpan komentar top-level
                    comments_by_post[post_url_key].append({
                        "username":   c_user,
                        "content":    c_text,
                        "likes":      c_likes,
                        "created_at": c_time,
                        "is_reply":   is_reply,
                        "reply_to":   reply_to
                    })

                    # Simpan juga replies nested (output actor punya field "replies")
                    for reply in (c_item.get("replies") or []):
                        if not isinstance(reply, dict):
                            continue
                        r_user = (
                            reply.get("ownerUsername") or
                            reply.get("username") or
                            (reply.get("owner") or {}).get("username") or
                            "unknown"
                        )
                        r_text  = reply.get("text") or reply.get("comment") or ""
                        r_likes = _safe_int(reply.get("likesCount") or reply.get("likeCount"))
                        r_time  = _parse_timestamp(reply.get("timestamp"))
                        comments_by_post[post_url_key].append({
                            "username":   r_user,
                            "content":    r_text,
                            "likes":      r_likes,
                            "created_at": r_time,
                            "is_reply":   True,
                            "reply_to":   c_user
                        })

                total_c = sum(len(v) for v in comments_by_post.values())
                print(f"   ↳ Komentar berhasil: {total_c} dari {len(comments_by_post)} post")
            else:
                print("   ⚠️ Comment scraper: tidak ada dataset ID")
        except Exception as e:
            print(f"   ⚠️ Comment scraper error (non-fatal): {e}")

    # Gabungkan komentar baru ke posts
    for p in posts:
        purl = p["post_url"]
        if purl in comments_by_post:
            # Merge: gabungkan komentar baru dengan yang sudah ada, hindari duplikat
            existing_contents = {c["content"] for c in p["comments_data"]}
            for new_c in comments_by_post[purl]:
                if new_c["content"] not in existing_contents:
                    p["comments_data"].append(new_c)
                    existing_contents.add(new_c["content"])

    # ── owner_replies: komentar yang dibuat si target di postingannya sendiri
    owner_replies: List[Dict] = []
    for p in posts:
        for c in p["comments_data"]:
            commenter = (c.get("username") or "").lower().strip()
            if commenter == clean_username.lower():
                owner_replies.append({
                    "post_url":     p["post_url"],
                    "post_caption": p["content"][:100] if p["content"] else "",
                    "post_author":  clean_username,
                    "content":      c["content"],
                    "likes":        c["likes"],
                    "is_reply":     c.get("is_reply", False),
                    "reply_to":     c.get("reply_to", ""),
                    "created_at":   c["created_at"],
                    "sentiment":    {"label": "NEUTRAL", "score": 0.5}
                })

    # Sort terbaru dulu, ambil 15
    owner_replies.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    owner_replies = owner_replies[:15]

    # ── Repost: deteksi dari caption + agregat ────────────────────────────
    reposts: List[Dict] = []
    for p in posts:
        if p.get("is_repost") and p.get("repost_author"):
            reposts.append({
                "video_url":        p["post_url"],
                "original_author":  p["repost_author"],
                "original_caption": p["content"][:300],
                "repost_date":      p["created_at"],
                "hashtags":         p["hashtags"],
                "metrics": {
                    "likes":    p["likes"],
                    "comments": p["comments"],
                    "shares":   p["shares"],
                    "plays":    p["plays"],
                },
                "sentiment":           {"label": "NEUTRAL", "score": 0.5},
                "content_description": f"Konten Instagram yang di-repost dari @{p['repost_author']}",
            })

    reposts = reposts[:10]

    total_likes    = sum(p["likes"]    for p in posts)
    total_comments = sum(p["comments"] for p in posts)

    print(f"   ↳ Posts: {len(posts)} | Likes: {total_likes:,} | Owner replies: {len(owner_replies)} | Reposts: {len(reposts)}")

    return {
        "status":         "success",
        "platform":       "instagram",
        "username":       clean_username,
        "full_name":      full_name,
        "bio":            bio,
        "followers":      followers,
        "following":      following,
        "likes":          0,
        "verified":       verified,
        "join_date":      "",
        "profile_pic":    profile_pic,
        "total_posts":    total_ig_posts or len(posts),
        "posts":          posts,
        "user_comments":  owner_replies,
        "reposts":        reposts,
        "total_likes":    total_likes,
        "total_comments": total_comments,
        "total_shares":   0,
        "has_text":       any(p["content"] for p in posts),
        "scrape_errors":  []
    }


async def _instagram_scrape_real(username: str, apify_token: str = "") -> Dict:
    """Async wrapper — blocking Apify call dijalankan di thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(_instagram_scrape_sync, username, apify_token)
    )
