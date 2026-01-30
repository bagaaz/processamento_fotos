import cv2

from src.preprocess import red_text_mask

def center_crop(bgr_img, crop_w: float, crop_h: float):
    """
    Recorta uma região central presumindo que a placa está no meio.
    crop_w e crop_h são frações (0..1).
    """
    h, w = bgr_img.shape[:2]
    cw = int(w * crop_w)
    ch = int(h * crop_h)

    x1 = max(0, (w - cw) // 2)
    y1 = max(0, (h - ch) // 2)
    x2 = min(w, x1 + cw)
    y2 = min(h, y1 + ch)
    return bgr_img[y1:y2, x1:x2]

def try_find_plate_like_region(bgr_img):
    """
    Tentativa opcional (mais “inteligente”): localizar um retângulo bem definido
    (tipo uma placa) via bordas/contornos.

    Se falhar, retorna None.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    h, w = bgr_img.shape[:2]
    img_area = h * w

    # Heurísticas: placa costuma ter área intermediária e formato retangular
    best = None
    best_score = 0.0

    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < img_area * 0.01 or area > img_area * 0.40:
            continue

        aspect = cw / max(ch, 1)
        if aspect < 1.2 or aspect > 8.0:
            continue

        # score simples: área * retangularidade
        score = area
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return None

    x, y, cw, ch = best
    return bgr_img[y:y+ch, x:x+cw]

def try_find_red_text_region(bgr_img):
    """
    Localiza região com texto vermelho (inscrição) em qualquer parte da imagem.
    Retorna o ROI recortado ou None.
    """
    mask = red_text_mask(bgr_img)
    if cv2.countNonZero(mask) == 0:
        return None

    # Dilata horizontalmente para unir dígitos próximos
    join = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)), iterations=1)

    cnts, _ = cv2.findContours(join, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    h, w = bgr_img.shape[:2]
    img_area = h * w

    best = None
    best_score = 0.0

    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < img_area * 0.0005 or area > img_area * 0.60:
            continue
        aspect = cw / max(ch, 1)
        if aspect < 1.0 or aspect > 15.0:
            continue
        score = area
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return None

    x, y, cw, ch = best
    pad_x = int(cw * 0.05)
    pad_y = int(ch * 0.35)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    return bgr_img[y1:y2, x1:x2]
