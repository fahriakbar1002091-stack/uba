"""
fetch_tiktok.py - TikTok Scraper via Apify
Actor: clockworks/tiktok-scraper
apify_client 3.x compatible + async-safe (runs blocking IO in thread executor)
"""

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
        "platform": "tiktok",
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
        "comments_on_posts": [],
        "reposts": [],
        "total_likes": 0,
        "total_comments": 0,
        "total_shares": 0,
        "has_text": False,
        "scrape_errors": [error]
    }


def _tiktok_scrape_sync(username: str, apify_token: str) -> Dict:
    """
    Synchronous scraper — dijalankan di thread executor agar tidak
    memblokir event loop FastAPI.
    """
    try:
        from apify_client import ApifyClient
    except ImportError:
        return _build_empty_result(username, "apify_client tidak terinstall")

    clean_username = username.replace("@", "").strip()
    print(f"📡 [TikTok] Scraping @{clean_username}...")

    try:
        client = ApifyClient(apify_token)

        print(f"   ↳ Menjalankan actor clockworks/tiktok-scraper...")
        run = client.actor("clockworks/tiktok-scraper").call(
            run_input={
                "profiles": [clean_username],
                "resultsPerPage": 20,
                "maxVideosPerProfile": 20,
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False,
                "shouldDownloadSubtitles": False,
                "proxyConfiguration": {"useApifyProxy": True},
            }
        )

        if not run:
            return _build_empty_result(clean_username, "Apify run gagal - tidak ada response")

        dataset_id = getattr(run, "default_dataset_id", None)
        if not dataset_id:
            return _build_empty_result(clean_username, "Apify run gagal - dataset ID tidak ditemukan")

        items: List[Dict] = client.dataset(dataset_id).list_items().items or []
        print(f"   ↳ Total item raw: {len(items)}")

        if not items:
            return _build_empty_result(clean_username, f"@{clean_username} tidak ditemukan atau tidak ada data")

        # ── Parse profile ─────────────────────────────────────────────────
        first = items[0]
        author_meta: Dict = (
            first.get("authorMeta") or
            (first.get("userInfo") or {}).get("user") or
            first.get("author") or
            {}
        )

        full_name: str = (
            author_meta.get("name") or
            author_meta.get("nickName") or
            author_meta.get("nickname") or
            clean_username
        )
        bio: str = (
            author_meta.get("signature") or
            author_meta.get("bio") or
            author_meta.get("description") or
            ""
        )
        followers: int = _safe_int(
            author_meta.get("fans") or
            author_meta.get("followerCount") or
            author_meta.get("followers") or
            (first.get("authorStats") or {}).get("followerCount")
        )
        following: int = _safe_int(
            author_meta.get("following") or
            author_meta.get("followingCount") or
            (first.get("authorStats") or {}).get("followingCount")
        )
        verified: bool = bool(
            author_meta.get("verified") or
            first.get("verified") or
            False
        )
        profile_pic: str = (
            author_meta.get("avatar") or
            author_meta.get("avatarLarger") or
            author_meta.get("avatarMedium") or
            author_meta.get("avatarThumb") or
            first.get("avatarLarger") or
            first.get("avatarMedium") or
            ""
        ) or ""
        total_heart: int = _safe_int(
            author_meta.get("heart") or
            author_meta.get("heartCount") or
            author_meta.get("likeCount")
        )

        print(f"   ↳ Profile: {full_name} | Followers: {followers:,} | Verified: {verified}")

        # ── Parse posts ───────────────────────────────────────────────────
        # Berdasarkan debug: semua item adalah VIDEO (type=null).
        # Field yang tersedia: id, text, diggCount, commentCount, shareCount,
        #   playCount, repostCount, createTime, hashtags, mentions,
        #   mediaUrls, videoMeta, musicMeta, webVideoUrl
        # Komentar TIDAK ada di sini — harus ambil terpisah via comment scraper.
        posts: List[Dict] = []
        video_ids: List[str] = []  # kumpulkan untuk ambil komentar

        for item in items:
            video_id = str(item.get("id") or "")
            if not video_id:
                continue

            video_ids.append(video_id)

            text: str = (item.get("text") or "").strip()

            # Metrics — field name REAL dari debug output
            digg_count    = _safe_int(item.get("diggCount"))
            comment_count = _safe_int(item.get("commentCount"))
            share_count   = _safe_int(item.get("shareCount"))
            play_count    = _safe_int(item.get("playCount"))
            repost_count  = _safe_int(item.get("repostCount"))
            collect_count = _safe_int(item.get("collectCount"))

            post_url: str = item.get("webVideoUrl") or f"https://www.tiktok.com/@{clean_username}/video/{video_id}"

            # Cover/thumbnail dari mediaUrls
            media_urls = item.get("mediaUrls") or []
            cover_url: str = media_urls[0] if media_urls else ""

            # Hashtags — clockworks menyediakan field 'hashtags' sebagai list dict
            hashtags: List[str] = []
            for h in (item.get("hashtags") or []):
                if isinstance(h, dict):
                    tag = h.get("name") or h.get("title") or ""
                    if tag:
                        hashtags.append(tag)
                elif isinstance(h, str):
                    hashtags.append(h)
            # Fallback dari teks jika hashtags kosong
            if not hashtags:
                hashtags = re.findall(r"#(\w+)", text)

            # Mentions — field 'mentions' atau 'detailedMentions'
            mentions: List[str] = []
            for m in (item.get("detailedMentions") or item.get("mentions") or []):
                if isinstance(m, dict):
                    name = m.get("uniqueId") or m.get("name") or ""
                    if name:
                        mentions.append(name)
                elif isinstance(m, str):
                    mentions.append(m)
            if not mentions:
                mentions = re.findall(r"@(\w+)", text)

            # Tanggal — gunakan createTimeISO atau createTime (unix timestamp)
            create_time = item.get("createTimeISO") or ""
            if not create_time:
                ts = item.get("createTime")
                if ts and str(ts).isdigit():
                    try:
                        create_time = time.strftime("%Y-%m-%d", time.localtime(int(ts)))
                    except Exception:
                        create_time = str(ts)
            else:
                # createTimeISO format: "2026-07-01T12:00:00.000Z"
                create_time = create_time[:10]

            # Tipe media
            video_meta: Dict = item.get("videoMeta") or {}
            is_slideshow = bool(item.get("isSlideshow"))
            media_type = "image" if is_slideshow else "video"
            duration = _safe_int(video_meta.get("duration"))

            # Music
            music_meta: Dict = item.get("musicMeta") or {}

            posts.append({
                "id":            video_id,
                "content":       text,
                "likes":         digg_count,
                "comments":      comment_count,
                "shares":        share_count,
                "plays":         play_count,
                "reposts":       repost_count,
                "saves":         collect_count,
                "post_url":      post_url,
                "video_url":     post_url,
                "cover_url":     cover_url,
                "created_at":    create_time,
                "hashtags":      hashtags,
                "mentions":      mentions,
                "media_type":    media_type,
                "duration_sec":  duration,
                "comments_data": [],  # diisi setelah ambil dari comment scraper
                "is_ad":         bool(item.get("isAd")),
                "is_pinned":     bool(item.get("isPinned")),
                "is_repost":     False,
                "music": {
                    "title":  music_meta.get("musicName")   or "",
                    "author": music_meta.get("musicAuthor") or "",
                }
            })

        # ── Ambil komentar via clockworks/tiktok-comments-scraper ────────
        # Strategi: ambil 10 video teratas dengan 50 komentar + replies per video
        # Lebih besar coverage → lebih besar kemungkinan owner muncul di komentar
        comments_by_video: Dict[str, List[Dict]] = {}
        top_video_ids = video_ids[:10]  # perluas ke 10 video

        if top_video_ids:
            print(f"   ↳ Mengambil komentar untuk {len(top_video_ids)} video teratas...")
            try:
                post_urls_for_comments = [
                    f"https://www.tiktok.com/@{clean_username}/video/{vid}"
                    for vid in top_video_ids
                ]
                comment_run = client.actor("clockworks/tiktok-comments-scraper").call(
                    run_input={
                        "postURLs":             post_urls_for_comments,
                        "commentsPerPost":      50,   # naikkan ke 50 per video
                        "maxRepliesPerComment": 10,   # lebih banyak reply per komentar
                    }
                )
                c_dataset_id = getattr(comment_run, "default_dataset_id", None)
                if c_dataset_id:
                    comment_items = client.dataset(c_dataset_id).list_items().items or []
                    print(f"   ↳ Total raw komentar: {len(comment_items)}")

                    for c_item in comment_items:
                        if c_item.get("errorCode"):
                            continue

                        # Ekstrak video ID dari videoWebUrl
                        video_web_url = c_item.get("videoWebUrl") or ""
                        vid_id = ""
                        parts = video_web_url.rstrip("/").split("/")
                        if parts:
                            candidate = parts[-1]
                            if candidate in top_video_ids:
                                vid_id = candidate
                        if not vid_id:
                            for v in top_video_ids:
                                if v in video_web_url:
                                    vid_id = v
                                    break
                        if not vid_id:
                            continue

                        if vid_id not in comments_by_video:
                            comments_by_video[vid_id] = []

                        c_text     = c_item.get("text") or ""
                        c_user     = c_item.get("uniqueId") or "unknown"
                        c_likes    = _safe_int(c_item.get("diggCount"))
                        c_replies  = _safe_int(c_item.get("replyCommentTotal"))
                        c_time_iso = c_item.get("createTimeISO") or ""
                        c_date     = c_time_iso[:10] if len(c_time_iso) >= 10 else ""

                        # Simpan komentar top-level
                        comments_by_video[vid_id].append({
                            "username":   c_user,
                            "content":    c_text,
                            "likes":      c_likes,
                            "replies":    c_replies,
                            "created_at": c_date,
                            "is_reply":   False,
                            "reply_to":   ""
                        })

                        # ── Parse reply threads ───────────────────────────
                        # Reply dari owner sering ada di sini
                        for reply in (c_item.get("replies") or []):
                            if not isinstance(reply, dict):
                                continue
                            r_user     = reply.get("uniqueId") or "unknown"
                            r_text     = reply.get("text") or ""
                            r_likes    = _safe_int(reply.get("diggCount"))
                            r_time_iso = reply.get("createTimeISO") or ""
                            r_date     = r_time_iso[:10] if len(r_time_iso) >= 10 else ""

                            comments_by_video[vid_id].append({
                                "username":   r_user,
                                "content":    r_text,
                                "likes":      r_likes,
                                "replies":    0,
                                "created_at": r_date,
                                "is_reply":   True,
                                "reply_to":   c_user   # balasan ke siapa
                            })

                    # Hitung total termasuk replies
                    total_c = sum(len(v) for v in comments_by_video.values())
                    print(f"   ↳ Video dengan komentar: {len(comments_by_video)}/{len(top_video_ids)} | Total entry: {total_c}")
                else:
                    print("   ⚠️ Comment scraper: tidak ada dataset ID")

            except Exception as e:
                print(f"   ⚠️ Comment scraper error: {e}")

        # ── Agregat: filter komentar dari owner sendiri ──────────────────
        # Cek baik komentar top-level maupun reply thread
        # Ambil 15 komentar terbaru yang ditulis si target
        owner_replies: List[Dict] = []

        # Pasang komentar dari comment scraper ke masing-masing post
        for p in posts:
            vid = p["id"]
            if vid in comments_by_video:
                p["comments_data"] = comments_by_video[vid]

        total_comments_fetched = sum(len(p["comments_data"]) for p in posts)
        print(f"   ↳ Total komentar terpasang: {total_comments_fetched}")

        # Filter komentar yang dibuat oleh si target sendiri
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
                        "replies":      c.get("replies", 0),
                        "is_reply":     c.get("is_reply", False),
                        "reply_to":     c.get("reply_to", ""),
                        "created_at":   c["created_at"],
                        "sentiment":    {"label": "NEUTRAL", "score": 0.5}
                    })

        print(f"   ↳ Owner replies ditemukan: {len(owner_replies)}")

        # Sort terbaru dulu, ambil 15
        owner_replies.sort(key=lambda x: x.get("created_at") or "", reverse=True)
        owner_replies = owner_replies[:15]

        # ── Fallback: jika owner tidak pernah reply, ambil komentar terpopuler ──
        # Ini memastikan user_comments tidak pernah kosong jika ada komentar
        if not owner_replies and total_comments_fetched > 0:
            all_comments_flat: List[Dict] = []
            for p in posts:
                for c in p["comments_data"]:
                    if c.get("content"):
                        all_comments_flat.append({
                            "post_url":     p["post_url"],
                            "post_caption": p["content"][:100] if p["content"] else "",
                            "post_author":  clean_username,
                            "content":      c["content"],
                            "likes":        c["likes"],
                            "replies":      c.get("replies", 0),
                            "is_reply":     c.get("is_reply", False),
                            "reply_to":     c.get("reply_to", ""),
                            "created_at":   c["created_at"],
                            "from_username": c.get("username", ""),
                            "sentiment":    {"label": "NEUTRAL", "score": 0.5},
                            "note":         "komentar dari pengikut (target tidak terdeteksi reply)"
                        })
            # Sort by likes terbanyak, ambil 10 terbaik
            all_comments_flat.sort(key=lambda x: x.get("likes", 0), reverse=True)
            owner_replies = all_comments_flat[:10]
            print(f"   ↳ Fallback: menampilkan {len(owner_replies)} komentar terpopuler")

        # ── Ambil REPOST via scrapelabsapi/tiktok-repost-scraper ─────────
        reposts: List[Dict] = []
        print(f"   ↳ Mengambil repost @{clean_username}...")
        try:
            profile_url = f"https://www.tiktok.com/@{clean_username}"
            repost_run = client.actor("scrapelabsapi/tiktok-repost-scraper").call(
                run_input={
                    "urls":     [{"url": profile_url}],
                    "maxItems": 10,
                }
            )
            r_dataset_id = getattr(repost_run, "default_dataset_id", None)
            if r_dataset_id:
                repost_items = client.dataset(r_dataset_id).list_items().items or []
                print(f"   ↳ Total raw repost: {len(repost_items)}")

                for r_item in repost_items:
                    if r_item.get("errorCode"):
                        print(f"   ⚠️ Repost error: {r_item.get('errorCode')}")
                        continue

                    r_id   = str(r_item.get("id") or "")
                    r_text = (r_item.get("desc") or r_item.get("text") or "").strip()

                    # author = kreator asli video yang di-repost
                    r_author = r_item.get("author") or {}
                    original_author = (
                        r_author.get("uniqueId") or
                        r_author.get("nickname") or
                        ""
                    ).strip()

                    # Skip jika video milik sendiri (bukan repost orang lain)
                    if original_author.lower() == clean_username.lower():
                        continue

                    r_url = (
                        r_item.get("webVideoUrl") or
                        (f"https://www.tiktok.com/@{original_author}/video/{r_id}"
                         if r_id and original_author else "")
                    )

                    # Tanggal
                    r_time = r_item.get("createTimeISO") or ""
                    if not r_time:
                        ts = r_item.get("createTime")
                        if ts and str(ts).isdigit():
                            try:
                                r_time = time.strftime("%Y-%m-%d", time.localtime(int(ts)))
                            except Exception:
                                pass
                    else:
                        r_time = r_time[:10]

                    # Metrics — output actor pakai field "stats"
                    r_stats    = r_item.get("stats") or {}
                    r_likes    = _safe_int(r_stats.get("diggCount")    or r_stats.get("likeCount"))
                    r_comments = _safe_int(r_stats.get("commentCount"))
                    r_shares   = _safe_int(r_stats.get("shareCount"))
                    r_plays    = _safe_int(r_stats.get("playCount"))

                    # Hashtags dari teks
                    r_hashtags: List[str] = re.findall(r"#(\w+)", r_text)

                    reposts.append({
                        "video_url":        r_url,
                        "original_author":  original_author,
                        "original_caption": r_text[:300],
                        "repost_date":      r_time,
                        "hashtags":         r_hashtags,
                        "metrics": {
                            "likes":    r_likes,
                            "comments": r_comments,
                            "shares":   r_shares,
                            "plays":    r_plays,
                        },
                        "sentiment":           {"label": "NEUTRAL", "score": 0.5},
                        "content_description": "",
                    })

                print(f"   ↳ Repost berhasil: {len(reposts)}")
            else:
                print("   ⚠️ Repost scraper: tidak ada dataset ID")

        except Exception as e:
            print(f"   ⚠️ Repost scraper error (non-fatal): {e}")

        # Sort reposts by repost_date descending, ambil 10 terbaru
        reposts.sort(key=lambda x: x.get("repost_date") or "", reverse=True)
        reposts = reposts[:10]

        print(f"   ↳ Owner replies: {len(owner_replies)} | Reposts: {len(reposts)}")

        total_likes    = sum(p["likes"]    for p in posts)
        total_comments = sum(p["comments"] for p in posts)
        total_shares   = sum(p["shares"]   for p in posts)

        print(f"   ↳ Posts: {len(posts)} | Likes: {total_likes:,} | Comments: {total_comments:,}")

        return {
            "status":            "success",
            "platform":          "tiktok",
            "username":          clean_username,
            "full_name":         full_name,
            "bio":               bio,
            "followers":         followers,
            "following":         following,
            "likes":             total_heart,
            "verified":          verified,
            "join_date":         "",
            "profile_pic":       profile_pic,
            "total_posts":       len(posts),
            "posts":             posts,
            "user_comments":     owner_replies,   # balasan owner di postingan sendiri
            "comments_on_posts": [],              # dihapus — tidak relevan
            "reposts":           reposts,
            "total_likes":       total_likes,
            "total_comments":    total_comments,
            "total_shares":      total_shares,
            "has_text":          any(p["content"] for p in posts),
            "scrape_errors":     []
        }

    except Exception as e:
        import traceback
        err = f"TikTok scrape error: {str(e)}"
        print(f"   ❌ {err}")
        print(traceback.format_exc())
        return _build_empty_result(clean_username, err)


async def _tiktok_scrape_real(username: str, apify_token: str) -> Dict:
    """
    Async wrapper — jalankan blocking Apify call di thread executor
    agar tidak memblokir event loop FastAPI.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(_tiktok_scrape_sync, username, apify_token)
    )
