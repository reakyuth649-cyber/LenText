# LensText — Telegram Bot

Telegram version of the LensText web app: send a photo or a video, get the text back (Gemini OCR, Khmer by default).

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) (`/newbot`) and copy the token.
2. Get a Gemini API key from [AI Studio](https://aistudio.google.com/apikey).
3. Install and configure:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env    # then fill in the two keys
   ```
4. Run:
   ```bash
   python bot.py
   ```

Docker: `docker build -t lenstext-bot . && docker run -d --env-file .env -v lenstext-data:/data -e DATA_FILE=/data/lenstext_data.pkl lenstext-bot`

## Usage

| Action | Result |
|---|---|
| Send a photo | OCR of the photo |
| Send an image as a *file* | OCR at full quality (no Telegram compression) — best for small text |
| Send a video | OCR of the first frame |
| Send a video with caption `2.5` or `1:05` | OCR of the frame at that time |
| `/lang` | Pick OCR language (default: Khmer; also auto-detect, English, Chinese, etc.) |
| `/history` | Your last 5 extractions |

## Notes

- Telegram bots can only download files up to **20 MB**.
- Language choice and history are stored per user in `lenstext_data.pkl`.
- Set `ALLOWED_USER_IDS` if you don't want strangers spending your Gemini quota.
- The web app's crop/zoom tools don't have a Telegram equivalent — crop the image before sending, or send it as a file.
