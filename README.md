# Region Detector

Watches a part of your screen for codes and sends them to Discord when they show up.

## What it does

You pick an area on your screen, and it watches that spot for text changes. When it finds new codes, it sends a Discord message with a screenshot. Pretty useful for monitoring game codes or whatever.

## Setup

You'll need Python and Tesseract OCR installed first.

**Get Tesseract:**
- Windows: Download from [here](https://github.com/UB-Mannheim/tesseract/wiki)
- Mac: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

**Install the thing:**
```bash
git clone https://github.com/yourusername/region-detector.git
cd region-detector
python setup.py
```

## How to use it

1. Run `python region_detector.py`
2. Click "Select Region" and drag to pick what part of your screen to watch
3. Put in your Discord webhook URL
4. It starts watching automatically

Press Ctrl+C to stop it.

## Discord webhook

Go to your Discord server → Server Settings → Integrations → Webhooks → New Webhook. Copy the URL and paste it when the app asks.

## Notes

- Takes screenshots every 5 seconds
- Keeps the last 10 screenshots, deletes older ones
- Won't spam you with the same code twice
- Everything stays on your computer except the Discord messages

## If something breaks

- Make sure Tesseract is installed and working
- Check your Discord webhook URL is right
- Try selecting a bigger/clearer area to monitor
- The text needs to be readable for OCR to work

That's pretty much it. Works on Windows, Mac, and Linux.