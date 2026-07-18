"""
fetch_tweet.py - Twitter/X Scraper via Apify v4.0
Actor utama: apidojo/twitter-profile-scraper
  - Support native retweet via includeNativeRetweets=true
  - Support replies via getReplies=true (type="reply" items)
  - Tidak butuh auth, tidak butuh proxy setup

FIELD PROFILE YANG DIAMBIL:
  user_id, username, display_name, bio, followers, following,
  verified (legacy + blue), profile_pic, banner_image,
  location, website, join_date (via getAboutData),
  tweet_count (dari profil item pertama)

FIELD TWEET YANG DIAMBIL:
  id, text/fullText, created_at, lang, source, tweet_url,
  hashtags, mentions, urls_in_tweet,
  media (list url + type),
  likes, retweets, replies, quotes, bookmarks, views,
  is_retweet, is_reply, is_quote, conversation_id,
  in_reply_to_status_id, referenced_tweet_url

REPOST (RETWEET):
  Dari tweet yang is_retweet=True atau text "RT @"
  original_author, original_caption, original_tweet_url

USER COMMENTS:
  Reply dari si target di postingannya sendiri
  → items dengan type="reply" atau isReply=True
  → filter author.userName == target
  → dari conversation threads (getReplies=true)

YANG TIDAK BISA DIAMBIL (limitasi Twitter/actor):
  - Poll detail          → tidak di-expose
  - Community Note       → tidak di-expose
  - Followers/Following list → butuh auth ($100/mo)
  - Like list            → butuh auth
  - Geo/Coordinates      → hampir tidak ada user aktifkan
  - media_count, lists_count → tidak di-expose
  - Akun < 10K followers → kadang timeline kosong (Twitter restriction)
"""

import re
import asyncio
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


def _parse_twitter_date(raw: Any) -> str:
    """Parse berbagai format tanggal Twitter ke YYYY-MM-DD."""
    if not raw:
        return ""
    s = str(raw).strip()
    # RFC 2822: "Wed Mar 05 14:23:01 +0000 2026"
    try:
        import email.utils
        return email.utils.parsedate_to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        pass
    # ISO / substring
    if len(s) >= 10:
        return s[:10]
    return s


def _build_empty_result(username: str, error: str) -> Dict:
    return {
        "status": "error", "platform": "x", "username": username,
        "full_name": "", "bio": "", "location": "", "website": "",
        "followers": 0, "following": 0, "tweet_count": 0,
        "verified": False, "join_date": "", "profile_pic": "",
        "banner_image": "", "user_id": "",
        "total_posts": 0, "total_likes": 0, "total_comments": 0,
        "total_shares": 0, "has_text": False,
        "posts": [], "user_comments": [], "reposts": [],
        "scrape_errors": [error],
        "data_limitations": [error]
    }


def _parse_media(media_raw: Any) -> Dict[str, Any]:
    """
    apidojo mengembalikan media sebagai list URL string.
    Contoh: ["https://pbs.twimg.com/media/xxx.jpg", "https://.../video.mp4"]
    """
    urls = media_raw if isinstance(media_raw, list) else []
    image_url = ""
    video_url = ""
    gif_url   = ""
    all_media: List[Dict] = []

    for m in urls:
        m_str = _safe_str(m)
        if not m_str:
            continue
        m_lower = m_str.lower()

        if any(x in m_lower for x in (".mp4", ".m3u8", "video", "ext_tw_video")):
            mtype = "video"
            video_url = video_url or m_str
        elif ".gif" in m_lower or "tweet_video" in m_lower:
            mtype = "gif"
            gif_url = gif_url or m_str
        else:
            mtype = "image"
            image_url = image_url or m_str

        all_media.append({"url": m_str, "type": mtype})

    return {
        "image_url":  image_url,
        "video_url":  video_url,
        "gif_url":    gif_url,
        "all_media":  all_media,
        "media_type": (
            "video" if video_url else
            "gif"   if gif_url   else
            "image" if image_url else
            "text"
        )
    }


def _extract_entities(item: Dict) -> Dict:
    """
    Ekstrak hashtags, mentions, dan URLs dari field entities
    atau fallback ke regex dari teks tweet.
    apidojo output: entities.hashtags[], entities.user_mentions[], entities.urls[]
    """
    entities   = item.get("entities") or {}
    text       = _safe_str(item.get("fullText") or item.get("text"))

    # Hashtags
    raw_tags = entities.get("hashtags") or []
    if raw_tags and isinstance(raw_tags[0], dict):
        hashtags = [_safe_str(h.get("text") or h.get("tag")) for h in raw_tags]
    elif raw_tags:
        hashtags = [_safe_str(h) for h in raw_tags]
    else:
        hashtags = re.findall(r"#(\w+)", text)

    # Mentions
    raw_mentions = entities.get("user_mentions") or []
    if raw_mentions and isinstance(raw_mentions[0], dict):
        mentions = [_safe_str(m.get("screen_name") or m.get("username")) for m in raw_mentions]
    elif raw_mentions:
        mentions = [_safe_str(m) for m in raw_mentions]
    else:
        mentions = re.findall(r"@(\w+)", text)

    # URLs — ambil expanded_url, bukan t.co
    raw_urls = entities.get("urls") or []
    urls = []
    for u in raw_urls:
        if isinstance(u, dict):
            expanded = _safe_str(u.get("expanded_url") or u.get("url"))
            if expanded and "t.co" not in expanded:
                urls.append(expanded)
        elif isinstance(u, str) and "t.co" not in u:
            urls.append(u)

    # Fallback: regex dari teks
    if not urls:
        urls = [u for u in re.findall(r"https?://\S+", text) if "t.co" not in u]

    return {
        "hashtags": [h for h in hashtags if h],
        "mentions": [m for m in mentions if m],
        "urls":     urls
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNC SCRAPER
# ─────────────────────────────────────────────────────────────────────────────
def _twitter_scrape_sync(username: str, apify_token: str) -> Dict:
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
        return _build_empty_result(username, "apify_client tidak terinstall — pip install apify-client")

    clean_username = username.replace("@", "").strip()
    print(f"📡 [X/Twitter] Scraping @{clean_username}...")

    try:
        client = ApifyClient(apify_token)

        # ══════════════════════════════════════════════════════════════════
        # RUN — apidojo/twitter-profile-scraper
        #
        # includeNativeRetweets=true → retweet masuk sebagai tweet item
        #   dengan text "RT @username: ..."
        # getReplies=true            → reply thread dari tiap tweet yang
        #   punya replyCount >= minReplyCount masuk sebagai type="reply"
        # getAboutData=true          → profile metadata: join_date, location, dll
        #
        # Pricing (free plan $5/bulan):
        #   $0.016 per profile (40 tweet gratis)
        #   $0.016 per reply query (36 reply gratis)
        #   $0.0004 per item tambahan
        #   maxItems=60 → estimasi ~$0.016 + sedikit, aman di free plan
        # ══════════════════════════════════════════════════════════════════
        print(f"   ↳ Mengambil tweets + replies via apidojo/twitter-profile-scraper...")
        run = client.actor("apidojo/twitter-profile-scraper").call(
            run_input={
                "twitterHandles":       [clean_username],
                "maxItems":             60,    # 40 gratis + 20 extra = ~$0.024
                "includeNativeRetweets": True, # ambil retweet
                "getReplies":           True,  # ambil reply threads
                "minReplyCount":        1,     # semua tweet yang punya >= 1 reply
                "getAboutData":         True,  # join_date, location, dll (+$0.002)
            }
        )

        dataset_id = getattr(run, "default_dataset_id", None)
        if not dataset_id:
            return _build_empty_result(clean_username, "Dataset ID tidak ditemukan")

        items: List[Dict] = client.dataset(dataset_id).list_items().items or []
        print(f"   ↳ Total item raw: {len(items)}")

        if not items:
            return _build_empty_result(
                clean_username,
                f"@{clean_username} tidak ditemukan atau akun < 10K followers "
                f"(Twitter platform restriction — timeline tidak tersedia tanpa auth)."
            )

        # ── Ekstrak profile dari author di item pertama ───────────────────
        # apidojo embed author info di tiap tweet item
        first_item   = items[0]
        author_raw   = first_item.get("author") or {}
        about_raw    = author_raw.get("about") or {}

        user_id      = _safe_str(author_raw.get("id") or author_raw.get("userId"))
        full_name    = _safe_str(author_raw.get("name") or clean_username)
        bio          = _safe_str(author_raw.get("description") or author_raw.get("bio"))
        followers    = _safe_int(author_raw.get("followers") or author_raw.get("followersCount"))
        following    = _safe_int(author_raw.get("following") or author_raw.get("friendsCount"))
        verified     = bool(
            author_raw.get("isBlueVerified") or
            author_raw.get("isVerified") or
            author_raw.get("verified")
        )
        profile_pic  = _safe_str(
            author_raw.get("profilePicture") or
            author_raw.get("profile_image_url")
        )
        banner_image = _safe_str(
            author_raw.get("profileBanner") or
            author_raw.get("profile_banner_url")
        )

        # Field dari about (requires getAboutData=true)
        join_date    = _parse_twitter_date(
            about_raw.get("accountCreatedAt") or
            author_raw.get("created_at") or ""
        )
        location     = _safe_str(
            about_raw.get("location") or
            author_raw.get("location") or ""
        )
        website      = _safe_str(
            author_raw.get("url") or
            author_raw.get("entities_url") or ""
        )

        # tweet_count tidak selalu ada — fallback ke count items
        tweet_count  = _safe_int(
            author_raw.get("statusesCount") or
            author_raw.get("tweet_count") or
            author_raw.get("tweetsCount") or 0
        )

        print(
            f"   ↳ Profile: {full_name} | "
            f"Followers: {followers:,} | "
            f"Verified: {verified} | "
            f"Banner: {'✓' if banner_image else '✗'} | "
            f"JoinDate: {join_date or '?'}"
        )

        # ── Parse semua item ──────────────────────────────────────────────
        # apidojo item types:
        #   type="tweet"  → original tweet
        #   type="reply"  → reply (dari getReplies=true)
        #   tidak ada type → tweet biasa
        posts:            List[Dict] = []
        reposts:          List[Dict] = []
        raw_user_comments: List[Dict] = []
        seen_ids:         set         = set()

        for item in items:
            item_type  = _safe_str(item.get("type")).lower()   # "tweet" / "reply" / ""
            item_id    = _safe_str(item.get("id"))
            if not item_id or item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            # Field utama — apidojo pakai likeCount, retweetCount, dll
            text       = _safe_str(item.get("fullText") or item.get("text"))
            likes      = _safe_int(item.get("likeCount")     or item.get("likes"))
            retweets   = _safe_int(item.get("retweetCount")  or item.get("retweets"))
            replies_ct = _safe_int(item.get("replyCount")    or item.get("replies"))
            quotes     = _safe_int(item.get("quoteCount")    or item.get("quotes"))
            bookmarks  = _safe_int(item.get("bookmarkCount") or item.get("bookmarks"))
            views      = _safe_int(item.get("viewCount")     or item.get("views"))
            created_at = _parse_twitter_date(item.get("createdAt") or item.get("created_at"))
            lang       = _safe_str(item.get("lang") or item.get("language"))
            source     = _safe_str(item.get("source"))
            conv_id    = _safe_str(item.get("conversationId") or item.get("conversation_id"))

            tweet_url = _safe_str(
                item.get("url") or
                item.get("twitterUrl") or
                item.get("tweet_url") or
                f"https://x.com/{clean_username}/status/{item_id}"
            )

            # Author embedded di tiap item
            item_author    = item.get("author") or {}
            item_handle    = _safe_str(
                item_author.get("userName") or
                item_author.get("username") or
                item_author.get("user_handle") or
                item.get("user_handle") or
                ""
            ).lower().lstrip("@")

            # Media
            media_info = _parse_media(
                item.get("media") or item.get("media_urls") or []
            )
            image_url  = media_info["image_url"]
            video_url  = media_info["video_url"]
            gif_url    = media_info["gif_url"]
            all_media  = media_info["all_media"]
            media_type = media_info["media_type"]

            # Entities
            ents     = _extract_entities(item)
            hashtags = ents["hashtags"]
            mentions = ents["mentions"]
            urls     = ents["urls"]

            # Referenced tweet
            in_reply_to_id   = _safe_str(
                item.get("inReplyToStatusId") or
                item.get("in_reply_to_status_id")
            )
            in_reply_to_user = _safe_str(
                item.get("inReplyToUser") or
                item.get("in_reply_to_screen_name") or
                (item.get("inReplyToUser") or {}).get("userName") if isinstance(item.get("inReplyToUser"), dict) else ""
            )
            referenced_url   = _safe_str(
                item.get("quotedTweetUrl") or
                item.get("quoted_tweet_url") or
                item.get("referenced_tweet_url") or ""
            )

            # ── Deteksi tipe ──────────────────────────────────────────────
            # Retweet: field isRetweet atau text "RT @"
            is_retweet = (
                bool(item.get("isRetweet") or item.get("is_retweet")) or
                text.startswith("RT @")
            )
            # Reply: type="reply", isReply=True, atau in_reply_to_id ada
            is_reply = (
                item_type == "reply" or
                bool(item.get("isReply") or item.get("is_reply")) or
                bool(in_reply_to_id) or
                (text.startswith("@") and not text.startswith("RT @") and len(text.split()) > 1)
            )
            is_quote = bool(item.get("isQuote") or item.get("is_quote") or item.get("is_quote_tweet"))

            # ── Bangun post entry ─────────────────────────────────────────
            post = {
                "id":               item_id,
                "content":          text,
                "tweet_url":        tweet_url,
                "post_url":         tweet_url,
                "conversation_id":  conv_id,
                "created_at":       created_at,
                "lang":             lang,
                "source":           source,
                "likes":            likes,
                "comments":         replies_ct,
                "shares":           retweets,
                "plays":            views,
                "quotes":           quotes,
                "bookmarks":        bookmarks,
                "image_url":        image_url,
                "video_url":        video_url,
                "gif_url":          gif_url,
                "cover_url":        image_url or video_url or gif_url,
                "all_media":        all_media,
                "media_type":       media_type,
                "duration_sec":     0,
                "hashtags":         hashtags,
                "mentions":         mentions,
                "urls":             urls,
                "is_reply":         is_reply,
                "is_repost":        is_retweet,
                "is_quote":         is_quote,
                "referenced_tweet_url": referenced_url,
                "in_reply_to_user": in_reply_to_user,
                "in_reply_to_id":   in_reply_to_id,
                "sentiment":        {"label": "NEUTRAL", "score": 0.5},
                "ai_description":   "",
                "comments_data":    [],
                "music":            {"title": "", "author": ""}
            }

            # ── Routing berdasarkan tipe ──────────────────────────────────
            if is_retweet:
                # Ekstrak info retweet
                rt_match    = re.match(r"RT @(\w+):\s*", text)
                orig_author = rt_match.group(1) if rt_match else ""
                orig_text   = re.sub(r"^RT @\w+:\s*", "", text).strip()

                reposts.append({
                    "video_url":           tweet_url,
                    "original_author":     orig_author,
                    "original_author_url": f"https://x.com/{orig_author}" if orig_author else "",
                    "original_tweet_url":  referenced_url,
                    "original_caption":    orig_text[:300],
                    "repost_date":         created_at,
                    "hashtags":            hashtags,
                    "mentions":            mentions,
                    "metrics": {
                        "likes":    likes,
                        "comments": replies_ct,
                        "shares":   retweets,
                        "plays":    views,
                        "quotes":   quotes
                    },
                    "sentiment":           {"label": "NEUTRAL", "score": 0.5},
                    "content_description": "",
                })
                # Jangan masukkan retweet ke posts

            elif is_reply:
                # Periksa apakah reply ini dari si target sendiri
                author_match = (
                    item_handle == clean_username.lower() or
                    # fallback: jika author tidak ada di item, asumsikan dari target
                    # (karena kita scrape profil target)
                    not item_handle
                )

                if author_match:
                    raw_user_comments.append({
                        "post_url":     tweet_url,
                        "post_caption": "",       # diisi setelah loop
                        "post_author":  clean_username,
                        "content":      text,
                        "likes":        likes,
                        "replies":      replies_ct,
                        "quotes":       quotes,
                        "is_reply":     True,
                        "reply_to":     in_reply_to_user,
                        "in_reply_to_id": in_reply_to_id,
                        "created_at":   created_at,
                        "lang":         lang,
                        "source":       source,
                        "sentiment":    {"label": "NEUTRAL", "score": 0.5}
                    })
                # Masukkan ke posts juga (reply termasuk aktivitas user)
                posts.append(post)

            else:
                posts.append(post)

        # ── Isi post_caption untuk user_comments ──────────────────────────
        # Cari tweet original yang di-reply menggunakan conversation_id
        # Buat index: tweet_id → content untuk lookup cepat
        tweet_content_index: Dict[str, str] = {}
        for p in posts:
            if not p.get("is_reply") and not p.get("is_repost"):
                pid = p.get("id") or ""
                if pid:
                    tweet_content_index[pid] = p["content"][:120]

        user_comments: List[Dict] = []
        for uc in raw_user_comments:
            # Cari caption dari tweet yang dibalas
            ref_id = uc.get("in_reply_to_id") or ""
            caption = tweet_content_index.get(ref_id, "")
            uc["post_caption"] = caption
            user_comments.append(uc)

        # ── Trim ─────────────────────────────────────────────────────────
        user_comments = user_comments[:15]
        reposts       = reposts[:10]

        # Posts final = semua yang bukan retweet, max 20
        final_posts = [p for p in posts if not p.get("is_repost")][:20]
        original_posts = [p for p in final_posts if not p.get("is_reply")]

        total_likes    = sum(p["likes"]    for p in final_posts)
        total_comments = sum(p["comments"] for p in final_posts)
        total_shares   = sum(p["shares"]   for p in final_posts)

        print(
            f"   ↳ Posts: {len(final_posts)} (original: {len(original_posts)}) | "
            f"user_comments: {len(user_comments)} | "
            f"reposts: {len(reposts)}"
        )

        data_limitations = [
            "poll_data: Twitter tidak expose poll via unauthenticated API",
            "community_note: tidak di-expose actor",
            "followers_list: butuh Twitter API resmi ($100/mo)",
            "following_list: butuh Twitter API resmi ($100/mo)",
            "likes_list: butuh Twitter API resmi",
            "alt_text_media: tidak di-expose actor",
            "geo_coordinates: hampir tidak ada user yang aktifkan",
            "media_count / lists_count: tidak di-expose actor",
            "view_count: bisa 0 untuk beberapa tweet (Twitter restriction)",
            "akun < 10K followers: timeline mungkin kosong (Twitter platform restriction)",
        ]

        return {
            "status":   "success",
            "platform": "x",
            "username": clean_username,

            # Profile
            "user_id":      user_id,
            "full_name":    full_name,
            "bio":          bio,
            "location":     location,
            "website":      website,
            "followers":    followers,
            "following":    following,
            "tweet_count":  tweet_count,
            "verified":     verified,
            "join_date":    join_date,
            "profile_pic":  profile_pic,
            "banner_image": banner_image,

            # Statistik
            "total_posts":    len(final_posts),
            "total_likes":    total_likes,
            "total_comments": total_comments,
            "total_shares":   total_shares,
            "has_text":       any(p["content"] for p in final_posts),

            # Data
            "posts":         final_posts,
            "user_comments": user_comments,
            "reposts":       reposts,

            # Meta
            "scrape_errors":    [],
            "data_limitations": data_limitations,
        }

    except Exception as e:
        import traceback
        err = f"Twitter/X scrape error: {str(e)}"
        print(f"   ❌ {err}")
        print(traceback.format_exc())
        return _build_empty_result(clean_username, err)


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
async def _twitter_scrape_real(username: str, apify_token: str = "") -> Dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(_twitter_scrape_sync, username, apify_token)
    )
