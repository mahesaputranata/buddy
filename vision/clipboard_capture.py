from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image, ImageGrab
from datetime import datetime

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def grab_clipboard_image(
    out_dir: Union[str, Path] = "screenshots",
    min_interval_s: float = 0.3,
    _state: dict = {},
) -> Optional[Tuple[str, str]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = ImageGrab.grabclipboard()

    img = None

    # Case A: actual image
    if isinstance(data, Image.Image):
        img = data

    # Case B: clipboard contains file path(s)
    elif isinstance(data, list) and data:
        p = Path(data[0])
        if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            img = Image.open(p).convert("RGB")

    if img is None:
        return None

    # Convert to bytes (PNG) for hashing + stable saving
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    h = _sha256_bytes(png_bytes)

    now = time.time()
    last_t = _state.get("t", 0.0)
    last_h = _state.get("h", "")

    # Debounce: if same image triggered too quickly, ignore
    if h == last_h and (now - last_t) < min_interval_s:
        return None

    _state["t"] = now
    _state["h"] = h

    now_dt = datetime.now()
    ts = now_dt.strftime("%Y%m%d_%H%M%S_") + f"{now_dt.microsecond // 1000:03d}"
    out_path = out_dir / f"clip_{ts}_{h[:8]}.png"

    out_path.write_bytes(png_bytes) 

    return str(out_path.resolve()), h  