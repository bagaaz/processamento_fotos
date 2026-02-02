from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    input_dir: Path
    output_dir: Path

    # Caso o tesseract não esteja no PATH, informe o caminho aqui:
    # Windows exemplo: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    tesseract_cmd: str | None = None

    # Placa no meio: recorte central (percentuais da imagem)
    center_crop_w: float = 0.55  # 55% da largura
    center_crop_h: float = 0.35  # 35% da altura

    # Se a inscrição for vermelha em fundo branco, use detecção por cor
    prefer_red_text: bool = True

    # Se falhar, tenta um pré-processamento mais agressivo (útil p/ imagens borradas)
    use_strong_preprocess_on_fail: bool = True

    # OCR: se a placa for só dígitos, isso ajuda muito
    ocr_psm: int = 7  # linha única; alternativas: 6 (bloco uniforme), 8 (uma palavra)
    digits_only: bool = True

    # Regex para “número de inscrição”
    # ajuste conforme seu padrão (tamanho mínimo/máximo, etc.)
    min_digits: int = 6
    max_digits: int = 6

    # Se souber exatamente o tamanho do código, use este valor
    exact_digits: int | None = 6

    # Se não achar no recorte, tenta no frame todo também
    fallback_full_image: bool = True

    # Tempo maximo por foto em segundos (0 desativa)
    photo_timeout_sec: float = 60.0
