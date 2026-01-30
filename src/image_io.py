from pathlib import Path
import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(input_dir: Path):
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p

def read_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}")
    return img

