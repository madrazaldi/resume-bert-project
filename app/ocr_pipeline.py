"""
Deep-learning OCR pipeline using Microsoft's TrOCR model.
"""

import io
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
)


class OcrEngine:
    """
    Minimal wrapper around TrOCR for turning resume images into text.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-small-printed",  # printed is trained for full-page printed text
        use_fast: bool = False,
    ):
        self.model_name = model_name
        self.available = False
        try:
            self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=use_fast)
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
        image = self._prepare_image(data)

        # TrOCR is line-oriented; segment lines to avoid collapsing whole documents into one token.
        lines = self._split_lines(image)

        # Batch lines to keep GPU/MPS busy and cut latency.
        pixel_values = self.processor(
            images=lines,
            return_tensors="pt",
            padding=True,  # ensure batch has consistent shapes
        ).pixel_values.to(DEVICE)
        generated_ids = self.model.generate(
            pixel_values,
            max_new_tokens=96,  # still bounded, handles denser lines
            num_beams=2,  # light beam search for quality on printed text
            do_sample=False,
            early_stopping=True,
        )
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        texts = [t.strip() for t in decoded if t.strip()]
        if not texts:
            return None
        return "\n".join(texts)

    def _prepare_image(self, data: bytes, max_side: int = 1100) -> Image.Image:
        """
        Basic preprocessing to keep OCR fast and legible:
        - Convert to RGB
        - Resize so the longest side is at most `max_side` pixels
        - Light sharpening to help printed text
        """
        image = Image.open(io.BytesIO(data)).convert("RGB")

        # Downscale very large images to reduce computation
        w, h = image.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, resample=Image.Resampling.BICUBIC)

        # Subtle sharpen to improve contrast for printed text
        image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
        return image

    def _split_lines(self, image: Image.Image) -> List[Image.Image]:
        """
        Naive line segmentation based on horizontal ink density.
        Works reasonably for well-formatted resumes without extra deps (no OpenCV).
        """
        grayscale = image.convert("L")
        data = np.array(grayscale)

        # Invert so that text pixels are high values; threshold to find ink per row.
        ink = 1.0 - (data.astype(np.float32) / 255.0)
        row_density = ink.mean(axis=1)

        # Consider rows with some ink as part of a line.
        threshold = max(0.05, row_density.mean() * 0.5)
        active_rows = row_density > threshold

        # Find contiguous segments of active rows.
        segments: List[Tuple[int, int]] = []
        start = None
        for idx, active in enumerate(active_rows.tolist()):
            if active and start is None:
                start = idx
            elif not active and start is not None:
                segments.append((start, idx))
                start = None
        if start is not None:
            segments.append((start, len(active_rows)))

        # Merge very small gaps to avoid over-segmentation.
        merged: List[Tuple[int, int]] = []
        min_gap = 3
        for seg in segments:
            if not merged:
                merged.append(seg)
                continue
            prev_start, prev_end = merged[-1]
            if seg[0] - prev_end <= min_gap:
                merged[-1] = (prev_start, seg[1])
            else:
                merged.append(seg)

        # Crop line images with a small vertical margin.
        margin = 2
        lines: List[Image.Image] = []
        for top, bottom in merged:
            top = max(0, top - margin)
            bottom = min(data.shape[0], bottom + margin)
            if bottom - top < 5:  # skip tiny noise
                continue
            line_img = image.crop((0, top, image.width, bottom))
            lines.append(line_img)

        # Fallback to whole image if segmentation fails
        if not lines:
            lines = [image]
        return lines
