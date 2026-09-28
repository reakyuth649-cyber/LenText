"""LensText Telegram Bot — OCR for photos and video frames, powered by Gemini.

Send a photo (or an image file) -> get the text back.
Send a video, with an optional timestamp as caption (e.g. `2.5` or `1:05`) -> OCR of that frame.
"""

import asyncio
import functools
import logging
import os
import tempfile
import time

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    PicklePersistence,
    filters,
)

load_dotenv()

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
ALLOWED_USER_IDS = {
    int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").replace(" ", "").split(",") if x
}
DATA_FILE = os.environ.get("DATA_FILE", "lenstext_data.pkl")

# Same list as the web app. "Detect language" = auto.
LANGUAGES = [
    "Detect language", "Khmer", "English", "Spanish", "French", "German", "Chinese",
    "Japanese", "Korean", "Russian", "Portuguese", "Italian", "Arabic",
]
DEFAULT_LANGUAGE = "Khmer"
HISTORY_LIMIT = 50
TG_MAX_DOWNLOAD = 20 * 1024 * 1024  # Bot API getFile limit
TG_MAX_MESSAGE = 4096

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("lenstext")

gemini = genai.Client(api_key=GEMINI_API_KEY)


# ---------- helpers ----------

def guarded(fn):
    """Restrict a handler to ALLOWED_USER_IDS (if configured)."""

    @functools.wraps(fn)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if ALLOWED_USER_IDS and (user is None or user.id not in ALLOWED_USER_IDS):
            if update.effective_message:
                await update.effective_message.reply_text("Sorry, this bot is private.")
            return
        return await fn(update, context)

    return wrapper


def build_prompt(language: str) -> str:
    if language == "Detect language":
        instruction = "Identify and extract all text from this image accurately, automatically detecting the language."
    else:
        instruction = f"Extract all text from this image accurately in {language}."
    return (
        f"{instruction} Maintain the original layout and formatting. "
        "Output only the extracted text. If there is no text, say 'No text found'."
    )


async def run_ocr(image_bytes: bytes, mime_type: str, language: str) -> str:
    response = await gemini.aio.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            build_prompt(language),
        ],
    )
    return (response.text or "").strip()


def parse_seconds(caption: str | None) -> float | None:
    """'' -> 0, '2.5' -> 2.5, '1:05' -> 65, '1:02:03' -> 3723. Invalid -> None."""
    if not caption or not caption.strip():
        return 0.0
    token = caption.split()[0]
    try:
        total = 0.0
        for part in token.split(":"):
            total = total * 60 + float(part)
        return total if total >= 0 else None
    except ValueError:
        return None


def extract_frame(video_bytes: bytes, seconds: float) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        path = f.name
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps > 0 and frames > 0 and seconds > frames / fps:
            raise ValueError(f"That video is only {frames / fps:.1f}s long.")
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Couldn't read a frame at that timestamp.")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("Couldn't encode the extracted frame.")
        return buf.tobytes()
    finally:
        cap.release()
        os.unlink(path)


async def send_long(message, text: str):
    """Reply, splitting at Telegram's 4096-char limit (prefer line breaks)."""
    while text:
        if len(text) <= TG_MAX_MESSAGE:
            await message.reply_text(text)
            return
        cut = text.rfind("\n", 0, TG_MAX_MESSAGE)
        if cut < TG_MAX_MESSAGE // 2:
            cut = TG_MAX_MESSAGE
        await message.reply_text(text[:cut])
        text = text[cut:].lstrip("\n")


def get_lang(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get("lang", DEFAULT_LANGUAGE)


def save_history(context: ContextTypes.DEFAULT_TYPE, text: str, language: str):
    history = context.user_data.setdefault("history", [])
    history.insert(0, {"text": text, "language": language, "ts": time.time()})
    del history[HISTORY_LIMIT:]


async def ocr_and_reply(update: Update, context: ContextTypes.DEFAULT_TYPE, image_bytes: bytes, mime: str):
    message = update.effective_message
    language = get_lang(context)
    try:
        text = await run_ocr(image_bytes, mime, language)
    except Exception:
        log.exception("Gemini OCR failed")
        await message.reply_text("Failed to extract text. Please try a different image and try again.")
        return
    if not text:
        await message.reply_text("No text could be extracted.")
        return
    if text.lower().rstrip(".") != "no text found":
        save_history(context, text, language)
    await send_long(message, text)


# ---------- commands ----------

@guarded
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "LensText — AI OCR\n\n"
        "Send me a photo (or an image file) and I'll extract the text.\n"
        "Send a video and I'll read one frame — add a timestamp as the caption "
        "(e.g. 2.5 or 1:05); default is the first frame.\n\n"
        f"Current language: {get_lang(context)}\n\n"
        "/lang — change OCR language\n"
        "/history — your last extractions\n"
        "/help — show this message"
    )


@guarded
async def lang_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose the OCR language:", reply_markup=lang_keyboard(get_lang(context)))


def lang_keyboard(current: str) -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(("✓ " if name == current else "") + name, callback_data=f"lang:{i}")
        for i, name in enumerate(LANGUAGES)
    ]
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(rows)


@guarded
async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        name = LANGUAGES[int(query.data.split(":", 1)[1])]
    except (ValueError, IndexError):
        await query.answer()
        return
    context.user_data["lang"] = name
    await query.answer(f"Language: {name}")
    await query.edit_message_text(f"OCR language set to: {name}", reply_markup=lang_keyboard(name))


@guarded
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    history = context.user_data.get("history", [])
    if not history:
        await update.message.reply_text("No history yet — send me an image first.")
        return
    lines = []
    for i, item in enumerate(history[:5], 1):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(item["ts"]))
        preview = " ".join(item["text"].split())
        preview = preview[:120] + ("…" if len(preview) > 120 else "")
        lines.append(f"{i}. [{item['language']}] {when}\n{preview}")
    await send_long(update.message, "\n\n".join(lines))


# ---------- media ----------

@guarded
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    await message.chat.send_action(ChatAction.TYPING)
    tg_file = await message.photo[-1].get_file()  # largest size
    data = bytes(await tg_file.download_as_bytearray())
    await ocr_and_reply(update, context, data, "image/jpeg")


@guarded
async def handle_image_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    doc = message.document
    if doc.file_size and doc.file_size > TG_MAX_DOWNLOAD:
        await message.reply_text("That file is over Telegram's 20 MB bot download limit.")
        return
    await message.chat.send_action(ChatAction.TYPING)
    tg_file = await doc.get_file()
    data = bytes(await tg_file.download_as_bytearray())
    await ocr_and_reply(update, context, data, doc.mime_type or "image/jpeg")


@guarded
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    media = message.video or message.video_note or message.document
    if media.file_size and media.file_size > TG_MAX_DOWNLOAD:
        await message.reply_text("That video is over Telegram's 20 MB bot download limit. Try a shorter clip.")
        return
    seconds = parse_seconds(message.caption)
    if seconds is None:
        await message.reply_text("I couldn't read that timestamp. Use seconds (2.5) or mm:ss (1:05) as the caption.")
        return
    await message.chat.send_action(ChatAction.TYPING)
    tg_file = await media.get_file()
    video_bytes = bytes(await tg_file.download_as_bytearray())
    try:
        frame = await asyncio.to_thread(extract_frame, video_bytes, seconds)
    except ValueError as e:
        await message.reply_text(str(e))
        return
    except Exception:
        log.exception("Frame extraction failed")
        await message.reply_text("I couldn't read that video file.")
        return
    await ocr_and_reply(update, context, frame, "image/jpeg")


@guarded
async def handle_other(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me a photo, an image file, or a video. /help for details.")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.error("Unhandled error", exc_info=context.error)


def main():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .persistence(PicklePersistence(filepath=DATA_FILE))
        .build()
    )
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("lang", lang_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CallbackQueryHandler(lang_callback, pattern=r"^lang:\d+$"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_image_document))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE | filters.Document.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_other))
    app.add_error_handler(on_error)
    log.info("LensText bot running (model: %s)", MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
