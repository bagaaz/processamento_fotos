import cv2
import numpy as np

def rotate_bound(img, angle_degrees: float):
    """
    Rota sem cortar conteúdo (expande canvas).
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos = abs(m[0, 0])
    sin = abs(m[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, m, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def deskew_by_red_mask(bgr_img, max_abs_angle: float = 25.0):
    """
    Usa o texto vermelho para estimar inclinação e corrige levemente.
    """
    mask = red_text_mask(bgr_img)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr_img, 0.0

    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    angle = rect[2]
    w, h = rect[1]

    # Normaliza para faixa [-45, 45]
    if w < h:
        angle = angle + 90.0

    if abs(angle) < 2.0 or abs(angle) > max_abs_angle:
        return bgr_img, 0.0

    return rotate_bound(bgr_img, angle), angle

def red_text_mask(bgr_img):
    """
    Mascara regiões vermelhas (inscrição) em placas de fundo branco.
    Retorna uma máscara binária (255 = vermelho detectado).
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    h, w = bgr_img.shape[:2]

    def build_mask(s_min: int, v_min: int):
        # Vermelho em HSV fica próximo de 0 e 180.
        lower1 = np.array([0, s_min, v_min], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, s_min, v_min], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask1, mask2)

    mask = build_mask(70, 50)
    min_pixels = max(30, int(0.0001 * h * w))
    if cv2.countNonZero(mask) < min_pixels:
        # Versão mais tolerante para vermelho apagado/blur
        mask = build_mask(40, 40)

    # Remove ruído fino e conecta traços
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel_join = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_join, iterations=1)
    return mask

def preprocess_for_ocr(bgr_img, prefer_red_text: bool = False):
    """
    Pré-processamento simples e eficiente para placas com texto numérico:
    - cinza
    - aumento de contraste (CLAHE)
    - blur leve
    - binarização adaptativa
    """
    if prefer_red_text:
        mask = red_text_mask(bgr_img)
        h, w = mask.shape[:2]
        min_pixels = max(40, int(0.0002 * h * w))
        if cv2.countNonZero(mask) >= min_pixels:
            # Inverte para deixar texto escuro em fundo claro (melhor p/ OCR)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            return cv2.bitwise_not(mask)

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Unsharp mask leve para recuperar bordas em imagens um pouco borradas
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptativo costuma funcionar melhor com variação de iluminação
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Em alguns casos o texto fica branco no fundo preto ou vice-versa.
    # Se a placa tiver fundo claro e texto escuro, isso normalmente já ajuda.
    return bin_img

def preprocess_for_ocr_strong(bgr_img, prefer_red_text: bool = False):
    """
    Variante mais agressiva para imagens mais borradas:
    - unsharp mais forte
    - leve upscale
    - binarização adaptativa com janela maior
    """
    if prefer_red_text:
        mask = red_text_mask(bgr_img)
        h, w = mask.shape[:2]
        min_pixels = max(30, int(0.00015 * h * w))
        if cv2.countNonZero(mask) >= min_pixels:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)),
                iterations=1,
            )
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
            return cv2.bitwise_not(mask)

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gray = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 8
    )
    return bin_img

def ensure_uint8(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img
