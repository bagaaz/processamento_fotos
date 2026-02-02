from __future__ import annotations
import argparse
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from config import AppConfig
from src.image_io import iter_images, read_image
from src.plate_finder import center_crop, try_find_plate_like_region, try_find_red_text_region
from src.preprocess import preprocess_for_ocr, preprocess_for_ocr_strong, deskew_by_red_mask, rotate_bound
from src.ocr_engine import setup_tesseract, ocr_text
from src.inscription_parser import extract_inscription_number
from src.renamer import copy_with_new_name

LOG = logging.getLogger("ocr")

def time_budget(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Tempo limite por foto excedido.")
    return remaining

def setup_logging(output_dir: Path, log_path: Path | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = output_dir / f"processamento-{stamp}.txt"
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)
    LOG.propagate = False
    return log_path

def ocr_with_variants(img, cfg: AppConfig, deadline: float | None):
    psms = [cfg.ocr_psm, 8, 6, 13]
    seen = set()
    candidates = []
    last_txt = ""
    for psm in psms:
        if psm in seen:
            continue
        seen.add(psm)
        timeout_sec = time_budget(deadline)
        txt = ocr_text(img, psm=psm, digits_only=cfg.digits_only, oem=1, timeout_sec=timeout_sec)
        last_txt = txt
        numero = extract_inscription_number(txt, cfg.min_digits, cfg.max_digits, cfg.exact_digits)
        if numero is not None:
            candidates.append(numero)
    return candidates, last_txt

def choose_best_candidate(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    counts = Counter(candidates)
    best, freq = counts.most_common(1)[0]
    # Se houver conflito (múltiplos candidatos diferentes sem repetição),
    # preferimos falhar a aceitar um número possivelmente incorreto.
    if freq == 1 and len(counts) > 1:
        return None
    return best

def quick_ocr(img, cfg: AppConfig, deadline: float | None) -> tuple[str | None, str]:
    timeout_sec = time_budget(deadline)
    txt = ocr_text(img, psm=cfg.ocr_psm, digits_only=cfg.digits_only, oem=1, timeout_sec=timeout_sec)
    numero = extract_inscription_number(txt, cfg.min_digits, cfg.max_digits, cfg.exact_digits)
    return numero, txt

def try_read_roi(roi, cfg: AppConfig, deadline: float | None) -> tuple[str | None, str]:
    # Passo rápido (PSM único)
    pre = preprocess_for_ocr(roi, prefer_red_text=cfg.prefer_red_text)
    numero, txt = quick_ocr(pre, cfg, deadline)
    if numero is not None:
        return numero, txt

    if cfg.use_strong_preprocess_on_fail:
        pre_strong = preprocess_for_ocr_strong(roi, prefer_red_text=cfg.prefer_red_text)
        numero, txt = quick_ocr(pre_strong, cfg, deadline)
        if numero is not None:
            return numero, txt

    # Passo lento: múltiplos PSMs + votação
    all_candidates = []
    candidates_now, txt = ocr_with_variants(pre, cfg, deadline)
    all_candidates.extend(candidates_now)
    if cfg.use_strong_preprocess_on_fail:
        candidates_now, txt = ocr_with_variants(pre_strong, cfg, deadline)
        all_candidates.extend(candidates_now)

    numero = choose_best_candidate(all_candidates)
    return numero, txt

def process_one(img_path: Path, cfg: AppConfig) -> tuple[bool, str]:
    """
    Retorna (ok, mensagem).
    """
    deadline = None
    if cfg.photo_timeout_sec > 0:
        deadline = time.monotonic() + cfg.photo_timeout_sec

    try:
        img = read_image(img_path)

        candidates = []

        # Estratégia 1: localizar texto vermelho em qualquer lugar
        if cfg.prefer_red_text:
            red_roi = try_find_red_text_region(img)
            if red_roi is not None:
                candidates.append(red_roi)

        # Estratégia 2: tentar achar um retângulo tipo placa
        plate_roi = try_find_plate_like_region(img)
        if plate_roi is not None:
            candidates.append(plate_roi)

        # Estratégia 3: fallback para recorte central
        candidates.append(center_crop(img, cfg.center_crop_w, cfg.center_crop_h))

        numero = None
        txt = ""

        for roi in candidates:
            # Primeiro tenta direto (rápido)
            numero, txt = try_read_roi(roi, cfg, deadline)
            if numero is not None:
                break

            # Se falhou, tenta variações mais lentas (deskew/rotações)
            roi_variants = []
            if cfg.prefer_red_text:
                deskewed, angle = deskew_by_red_mask(roi)
                if angle != 0.0:
                    roi_variants.append(deskewed)

            h, w = roi.shape[:2]
            if h > w * 1.15:
                roi_variants.append(rotate_bound(roi, 90))
                roi_variants.append(rotate_bound(roi, -90))

            for rv in roi_variants:
                numero, txt = try_read_roi(rv, cfg, deadline)
                if numero is not None:
                    break
            if numero is not None:
                break

        # Fallback: tenta OCR na imagem toda se não achou
        if (numero is None) and cfg.fallback_full_image:
            numero, txt2 = try_read_roi(img, cfg, deadline)
            if numero is not None:
                txt = txt2

        if numero is None:
            # Salva também os não identificados com prefixo
            fallback_name = f"nao-identificado-{img_path.stem}"
            dst = copy_with_new_name(img_path, cfg.output_dir, fallback_name)
            return (False, f"SEM NÚMERO (OCR='{txt[:40]}...') -> {dst.name}")

        dst = copy_with_new_name(img_path, cfg.output_dir, numero)
        return (True, f"{img_path.name} -> {dst.name}")
    except TimeoutError:
        fallback_name = f"timeout-{img_path.stem}"
        dst = copy_with_new_name(img_path, cfg.output_dir, fallback_name)
        return (False, f"TIMEOUT (> {cfg.photo_timeout_sec:.0f}s) -> {dst.name}")

def process_one_safe(img_path: Path, cfg: AppConfig) -> tuple[bool, str]:
    try:
        return process_one(img_path, cfg)
    except TimeoutError:
        fallback_name = f"timeout-{img_path.stem}"
        dst = copy_with_new_name(img_path, cfg.output_dir, fallback_name)
        return (False, f"TIMEOUT (> {cfg.photo_timeout_sec:.0f}s) -> {dst.name}")
    except Exception as exc:
        fallback_name = f"erro-{img_path.stem}"
        dst = copy_with_new_name(img_path, cfg.output_dir, fallback_name)
        return (False, f"ERRO ({type(exc).__name__}): {exc} -> {dst.name}")

def parse_args() -> tuple[AppConfig, int, Path | None]:
    ap = argparse.ArgumentParser(description="Renomeia fotos pelo número de inscrição via OCR.")
    ap.add_argument("--input", required=True, help="Pasta com as fotos")
    ap.add_argument("--output", required=True, help="Pasta destino")
    ap.add_argument("--tesseract-cmd", default=None, help="Caminho do executável do tesseract (se necessário)")
    ap.add_argument("--min-digits", type=int, default=4)
    ap.add_argument("--max-digits", type=int, default=12)
    ap.add_argument("--psm", type=int, default=7)
    ap.add_argument("--no-digits-only", action="store_true", help="Desativa whitelist de dígitos (se a placa tiver letras)")
    ap.add_argument("--workers", type=int, default=0, help="Número de threads (0 = automático)")
    ap.add_argument("--photo-timeout", type=float, default=60.0, help="Tempo maximo por foto (s). 0 desativa.")
    ap.add_argument("--log", default=None, help="Caminho do arquivo de log (.txt). Padrão: na pasta de saida.")
    args = ap.parse_args()
    photo_timeout = max(0.0, args.photo_timeout)
    cfg = AppConfig(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        tesseract_cmd=args.tesseract_cmd,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        ocr_psm=args.psm,
        digits_only=not args.no_digits_only,
        photo_timeout_sec=photo_timeout,
    )
    log_path = Path(args.log) if args.log else None
    return cfg, args.workers, log_path

def main():
    cfg, workers, log_path = parse_args()
    setup_tesseract(cfg.tesseract_cmd)
    log_file = setup_logging(cfg.output_dir, log_path)
    LOG.info("Inicio: input=%s output=%s", cfg.input_dir, cfg.output_dir)
    LOG.info("Config: timeout=%.1fs", cfg.photo_timeout_sec)
    print(f"Log: {log_file}")

    if not cfg.input_dir.exists():
        LOG.error("Pasta de entrada nao existe: %s", cfg.input_dir)
        raise SystemExit(f"Pasta de entrada não existe: {cfg.input_dir}")

    images = list(iter_images(cfg.input_dir))
    if not images:
        LOG.error("Nenhuma imagem encontrada na pasta de entrada.")
        raise SystemExit("Nenhuma imagem encontrada na pasta de entrada.")

    ok_count = 0
    fail_count = 0

    if workers <= 0:
        workers = min(4, (os.cpu_count() or 2))
    LOG.info("Workers efetivo: %s", workers)

    if workers == 1:
        for p in tqdm(images, desc="Processando"):
            ok, msg = process_one_safe(p, cfg)
            if ok:
                ok_count += 1
                LOG.info(msg)
            else:
                fail_count += 1
                LOG.warning(msg)
            print(msg)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one_safe, p, cfg): p for p in images}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processando"):
                try:
                    ok, msg = fut.result()
                except Exception as exc:
                    ok = False
                    msg = f"ERRO ({type(exc).__name__}): {exc}"
                if ok:
                    ok_count += 1
                    LOG.info(msg)
                else:
                    fail_count += 1
                    LOG.warning(msg)
                print(msg)

    print("\nResumo:")
    print(f"  OK:   {ok_count}")
    print(f"  FALHA:{fail_count}")
    print(f"Saída: {cfg.output_dir}")
    LOG.info("Resumo: OK=%s FALHA=%s", ok_count, fail_count)
    LOG.info("Saida: %s", cfg.output_dir)

if __name__ == "__main__":
    main()
