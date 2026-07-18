"""
media_extractor.py - Social Radar Media Content Extraction Layer
================================================================
Mengekstrak teks dari isi media post secara nyata:
  - IMAGE / CAROUSEL : OCR dari URL gambar nyata
  - VIDEO            : OCR dari frames + speech-to-text dari audio nyata

Tidak ada dummy/mock/fabricated data.
Jika ekstraksi gagal → kembalikan string kosong + catat error.

Dependencies (install jika belum ada):
    pip install pytesseract Pillow requests
    pip install openai-whisper           # untuk transcript audio
    pip install opencv-python            # untuk frame extraction
    pip install yt-dlp                   # untuk download video

Tesseract binary harus terinstall di sistem:
    Windows : https://github.com/UB-Mannheim/tesseract/wiki
    Linux   : sudo apt install tesseract-ocr
"""

from __future__ import annotations

import os
import io
import re
import time
import logging
import tempfile
import subprocess
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
def _empty_extraction() -> Dict:
    """Struktur kosong yang aman jika ekstraksi tidak dilakukan / gagal."""
    return {
        "ocr_text":        "",
        "audio_transcript": "",
        "visual_description": "",
        "extraction_errors": []
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: download image ke bytes (tanpa simpan ke disk)
# ─────────────────────────────────────────────────────────────────────────────
def _download_image_bytes(url: str, timeout: int = 15) -> Optional[bytes]:
    """Download image dari URL. Return bytes atau None jika gagal."""
    try:
        import requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            logger.debug(f"Bukan image content-type: {content_type} dari {url[:80]}")
            return None
        return resp.content
    except Exception as e:
        logger.debug(f"Gagal download image {url[:80]}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: OCR satu image bytes
# ─────────────────────────────────────────────────────────────────────────────
def _ocr_image_bytes(image_bytes: bytes) -> str:
    """
    Jalankan OCR pada image bytes menggunakan pytesseract.
    Return teks hasil OCR (bisa kosong jika tidak ada teks di gambar).
    """
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))

        # Konversi ke RGB jika perlu (RGBA / palette tidak didukung OCR dengan baik)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Config: PSM 3 = auto page segmentation, PSM 11 = sparse text (cocok untuk social media)
        custom_config = r"--oem 3 --psm 3"
        raw_text: str = pytesseract.image_to_string(img, config=custom_config)

        # Bersihkan whitespace berlebih
        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        return cleaned

    except ImportError:
        # pytesseract tidak terinstall — tidak crash, return kosong
        logger.warning("pytesseract tidak terinstall. OCR dilewati.")
        return ""
    except Exception as e:
        logger.debug(f"OCR error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CORE: ekstraksi IMAGE / CAROUSEL
# ─────────────────────────────────────────────────────────────────────────────
def extract_image(
    image_url: str,
    carousel_urls: Optional[List[str]] = None
) -> Dict:
    """
    Ekstraksi teks dari satu image atau semua slide carousel.

    Parameters
    ----------
    image_url      : URL gambar utama (cover / thumbnail)
    carousel_urls  : list URL semua slide carousel (opsional)

    Returns
    -------
    dict dengan key: ocr_text, audio_transcript, visual_description, extraction_errors
    """
    result = _empty_extraction()
    errors: List[str] = []

    # Kumpulkan semua URL yang perlu di-OCR
    urls_to_process: List[str] = []
    if carousel_urls:
        urls_to_process = [u for u in carousel_urls if u]
    if image_url and image_url not in urls_to_process:
        urls_to_process.insert(0, image_url)

    if not urls_to_process:
        errors.append("Tidak ada image URL tersedia")
        result["extraction_errors"] = errors
        return result

    ocr_parts: List[str] = []

    for idx, url in enumerate(urls_to_process[:5]):   # max 5 slide
        img_bytes = _download_image_bytes(url)
        if img_bytes is None:
            errors.append(f"Gagal download image slide {idx + 1}: {url[:80]}")
            continue

        ocr_result = _ocr_image_bytes(img_bytes)
        if ocr_result:
            ocr_parts.append(ocr_result)

    result["ocr_text"]          = " | ".join(ocr_parts) if ocr_parts else ""
    result["extraction_errors"] = errors
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: download video ke file sementara
# ─────────────────────────────────────────────────────────────────────────────
def _download_video_to_tempfile(video_url: str, max_size_mb: int = 50) -> Optional[str]:
    """
    Download video dari URL ke file sementara.
    Return path file sementara atau None jika gagal / terlalu besar.

    Mencoba urutan:
      1. requests langsung (untuk URL CDN biasa)
      2. yt-dlp (untuk URL platform yang memerlukan session)
    """
    # ── Attempt 1: requests langsung ─────────────────────────────────────
    try:
        import requests
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(video_url, headers=headers, timeout=30, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        is_video = any(t in content_type for t in ("video/", "application/octet-stream", "binary/"))
        if not is_video and "html" in content_type:
            raise ValueError(f"Response bukan video: {content_type}")

        # Cek ukuran
        content_length = int(resp.headers.get("Content-Length", 0))
        if content_length > max_size_mb * 1024 * 1024:
            raise ValueError(f"Video terlalu besar: {content_length // (1024*1024)}MB > {max_size_mb}MB")

        suffix = ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > max_size_mb * 1024 * 1024:
                        raise ValueError(f"Video melebihi batas {max_size_mb}MB saat download")
            temp_path = f.name

        logger.debug(f"Video didownload via requests: {temp_path} ({downloaded // 1024}KB)")
        return temp_path

    except Exception as e:
        logger.debug(f"Download via requests gagal: {e}. Mencoba yt-dlp...")

    # ── Attempt 2: yt-dlp ─────────────────────────────────────────────────
    try:
        import yt_dlp   # type: ignore

        suffix = ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            temp_path = f.name

        ydl_opts = {
    "outtmpl":       temp_path.replace(".mp4", ".%(ext)s"),
    "format":        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    "quiet":         True,
    "no_warnings":   True,
    "max_filesize":  max_size_mb * 1024 * 1024,
    "cookiefile":    "cookies.txt",  # ← TAMBAHKAN INI!
    }
        # yt-dlp mungkin ganti ekstensi
        for ext in ("mp4", "webm", "mkv", "mov"):
            candidate = temp_path.replace(".mp4", f".{ext}")
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                logger.debug(f"Video didownload via yt-dlp: {candidate}")
                return candidate

        return None

    except ImportError:
        logger.debug("yt-dlp tidak terinstall.")
        return None
    except Exception as e:
        logger.debug(f"yt-dlp gagal: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: ekstrak frames dari video dan OCR
# ─────────────────────────────────────────────────────────────────────────────
def _extract_frames_ocr(video_path: str, num_frames: int = 4) -> str:
    """
    Ekstrak `num_frames` frame dari video dan jalankan OCR.
    Return teks gabungan dari semua frame.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("opencv-python tidak terinstall. Frame OCR dilewati.")
        return ""

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return ""

        # Ambil frame di posisi yang tersebar merata
        frame_positions = [
            int(total_frames * i / (num_frames + 1))
            for i in range(1, num_frames + 1)
        ]

        ocr_parts: List[str] = []
        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue

            # Encode frame ke PNG bytes untuk di-OCR
            success, encoded = cv2.imencode(".png", frame)
            if not success:
                continue

            frame_bytes = encoded.tobytes()
            ocr_text = _ocr_image_bytes(frame_bytes)
            if ocr_text and ocr_text not in ocr_parts:
                ocr_parts.append(ocr_text)

        cap.release()
        return " | ".join(ocr_parts)

    except Exception as e:
        logger.debug(f"Frame extraction error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: transcript audio dari video
# ─────────────────────────────────────────────────────────────────────────────
def _transcribe_audio(video_path: str) -> str:
    """
    Transkripsi audio dari video menggunakan OpenAI Whisper (local model).
    Return teks transcript atau string kosong jika gagal.
    """
    try:
        import whisper  # type: ignore
    except ImportError:
        logger.warning("openai-whisper tidak terinstall. Transcript dilewati.")
        return ""

    try:
        # Load model kecil — cepat, cukup akurat untuk social media
        # Model "tiny" atau "base" cocok untuk production agar tidak terlalu lambat
        whisper_model_size = os.getenv("WHISPER_MODEL", "base")

        model = whisper.load_model(whisper_model_size)
        result = model.transcribe(
            video_path,
            fp16=False,       # fp16=True hanya jika ada GPU
            language=None,    # auto-detect language
            verbose=False
        )
        transcript: str = result.get("text", "").strip()
        logger.debug(f"Whisper transcript ({len(transcript)} chars)")
        return transcript

    except Exception as e:
        logger.debug(f"Whisper transcription error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# CORE: ekstraksi VIDEO
# ─────────────────────────────────────────────────────────────────────────────
def extract_video(
    video_url: str,
    cover_url: str = "",
    max_size_mb: int = 40
) -> Dict:
    """
    Ekstraksi teks dari video:
      1. Download video ke tempfile
      2. OCR dari beberapa frame
      3. Transcript audio dengan Whisper
      4. Jika download gagal, fallback OCR ke cover/thumbnail

    Parameters
    ----------
    video_url   : URL direct video (mp4/webm) atau URL platform
    cover_url   : URL thumbnail/cover sebagai fallback OCR
    max_size_mb : batas ukuran video yang didownload

    Returns
    -------
    dict dengan key: ocr_text, audio_transcript, visual_description, extraction_errors
    """
    result = _empty_extraction()
    errors: List[str] = []
    temp_path: Optional[str] = None

    if not video_url:
        errors.append("Tidak ada video URL tersedia")
        # Fallback ke thumbnail OCR
        if cover_url:
            thumb_result = extract_image(cover_url)
            result["ocr_text"]          = thumb_result["ocr_text"]
            result["extraction_errors"] = errors + thumb_result["extraction_errors"]
        else:
            result["extraction_errors"] = errors
        return result

    try:
        # ── Download video ────────────────────────────────────────────────
        logger.debug(f"Downloading video: {video_url[:80]}...")
        temp_path = _download_video_to_tempfile(video_url, max_size_mb=max_size_mb)

        if not temp_path or not os.path.exists(temp_path):
            errors.append(f"Gagal download video dari {video_url[:80]}")
            # Fallback: OCR dari thumbnail
            if cover_url:
                thumb_result = extract_image(cover_url)
                result["ocr_text"]          = thumb_result["ocr_text"]
                result["extraction_errors"] = errors + thumb_result["extraction_errors"]
            else:
                result["extraction_errors"] = errors
            return result

        file_size_kb = os.path.getsize(temp_path) // 1024
        logger.debug(f"Video downloaded: {file_size_kb}KB → {temp_path}")

        # ── OCR dari frames ───────────────────────────────────────────────
        ocr_text = _extract_frames_ocr(temp_path, num_frames=4)

        # ── Transcript audio ──────────────────────────────────────────────
        transcript = _transcribe_audio(temp_path)

        result["ocr_text"]          = ocr_text
        result["audio_transcript"]  = transcript
        result["extraction_errors"] = errors

    except Exception as e:
        errors.append(f"Video extraction error: {str(e)}")
        result["extraction_errors"] = errors

    finally:
        # Hapus file sementara — selalu, meski error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT: extract_post_content
# ─────────────────────────────────────────────────────────────────────────────
def extract_post_content(activity: Dict) -> Dict:
    """
    Entry point utama — dipanggil per post/activity dari pipeline.

    Membaca field dari activity dict (output scraper_engine._normalise_post):
      - content_type : "video" / "image" / "carousel"
      - video_url    : URL direct video
      - image_url    : URL gambar utama
      - cover_url    : URL thumbnail

    Returns
    -------
    dict: ocr_text, audio_transcript, visual_description, extraction_errors
    """
    content_type = (activity.get("content_type") or "").lower()
    video_url    = activity.get("video_url") or ""
    image_url    = activity.get("image_url") or activity.get("cover_url") or ""
    cover_url    = activity.get("cover_url") or image_url

    if content_type == "video" or (video_url and content_type != "image"):
        return extract_video(video_url=video_url, cover_url=cover_url)

    elif content_type in ("image", "carousel", "photo"):
        return extract_image(image_url=image_url)

    else:
        # Tipe tidak dikenal — coba image jika ada URL
        if image_url:
            return extract_image(image_url=image_url)
        return _empty_extraction()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build combined_text dari semua sumber
# ─────────────────────────────────────────────────────────────────────────────
def build_combined_text(
    caption_text: str,
    ocr_text: str,
    audio_transcript: str,
    visual_description: str = ""
) -> str:
    """
    Gabungkan semua sumber teks yang benar-benar tersedia.
    Urutan: caption → OCR → transcript → visual description.
    Field kosong diabaikan — tidak ada padding palsu.
    """
    parts: List[str] = []

    if caption_text and caption_text.strip():
        parts.append(caption_text.strip())

    if ocr_text and ocr_text.strip():
        # Hindari duplikasi jika OCR hasilnya sama dengan caption
        ocr_clean = ocr_text.strip()
        if ocr_clean not in (parts[0] if parts else ""):
            parts.append(f"[OCR] {ocr_clean}")

    if audio_transcript and audio_transcript.strip():
        parts.append(f"[TRANSCRIPT] {audio_transcript.strip()}")

    if visual_description and visual_description.strip():
        parts.append(f"[VISUAL] {visual_description.strip()}")

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECK — dijalankan sekali saat import untuk info saja
# ─────────────────────────────────────────────────────────────────────────────
def check_dependencies() -> Dict[str, bool]:
    """
    Cek apakah dependency opsional tersedia.
    Dipanggil dari main.py saat startup untuk logging informatif.
    """
    status = {}

    try:
        import pytesseract
        import PIL
        status["ocr_pytesseract"] = True
    except ImportError:
        status["ocr_pytesseract"] = False

    try:
        import cv2
        status["video_opencv"] = True
    except ImportError:
        status["video_opencv"] = False

    try:
        import whisper
        status["transcript_whisper"] = True
    except ImportError:
        status["transcript_whisper"] = False

    try:
        import yt_dlp
        status["video_ytdlp"] = True
    except ImportError:
        status["video_ytdlp"] = False

    try:
        import requests
        status["requests"] = True
    except ImportError:
        status["requests"] = False

    return status
