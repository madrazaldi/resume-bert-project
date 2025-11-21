"""
External OCR via OCR.Space API.
"""

import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class OcrEngine:
    """
    Calls OCR.Space to extract text from an image.
    """

    def __init__(self):
        self.api_key = os.getenv("OCR_SPACE_API_KEY")
        self.available = bool(self.api_key)
        if not self.available:
            print("[OCR] OCR_SPACE_API_KEY not set; OCR endpoints will be unavailable.")

    def image_to_text(self, data: bytes) -> Optional[str]:
        if not self.available:
            return None

        url = "https://api.ocr.space/parse/image"
        files = {"file": ("upload.png", data, "image/png")}
        payload = {
            "language": "eng",
            "OCREngine": 2,
            "isOverlayRequired": False,
            "scale": True,
        }
        headers = {"apikey": self.api_key}

        try:
            resp = requests.post(url, files=files, data=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            if result.get("IsErroredOnProcessing"):
                print(f"[OCR] API error: {result.get('ErrorMessage')}")
                return None
            parsed = result.get("ParsedResults") or []
            if not parsed:
                return None
            text = parsed[0].get("ParsedText", "").strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            print(f"[OCR] External OCR failed: {exc}")
            return None
