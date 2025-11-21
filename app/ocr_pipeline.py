"""
Deep-learning OCR pipeline using Microsoft's TrOCR model.
"""

import io
from typing import Optional

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OcrEngine:
    """
    Minimal wrapper around TrOCR for turning resume images into text.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        self.model_name = model_name
        self.available = False
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(DEVICE)
            self.model.eval()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            print(f"[OCR] Failed to load {model_name}: {exc}")
            self.available = False

    @torch.no_grad()
    def image_to_text(self, data: bytes) -> Optional[str]:
        if not self.available:
            return None
        image = Image.open(io.BytesIO(data)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
        generated_ids = self.model.generate(pixel_values, max_length=512)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
