from __future__ import annotations
import pytesseract
import cv2

def setup_tesseract(tesseract_cmd: str | None):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def ocr_text(
    img,
    psm: int = 7,
    digits_only: bool = True,
    oem: int | None = None,
    timeout_sec: float | None = None,
) -> str:
    """
    Recebe uma imagem (idealmente binária/preprocessada) e retorna texto.
    """
    config_parts = [f"--psm {psm}"]
    if oem is not None:
        config_parts.append(f"--oem {oem}")
    if digits_only:
        # restringe a saída a dígitos; melhora bastante quando é só número
        config_parts.append("-c tessedit_char_whitelist=0123456789")

    config = " ".join(config_parts)

    # Se vier binária, ok. Se vier BGR, também funciona, mas ideal é binária/gray.
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        if timeout_sec is None:
            text = pytesseract.image_to_string(img, config=config)
        else:
            text = pytesseract.image_to_string(img, config=config, timeout=timeout_sec)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            raise TimeoutError("OCR timeout") from exc
        raise
    return text.strip()
