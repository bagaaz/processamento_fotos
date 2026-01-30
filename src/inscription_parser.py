import re

def extract_inscription_number(
    text: str,
    min_digits: int = 4,
    max_digits: int = 12,
    exact_digits: int | None = None,
) -> str | None:
    """
    Extrai o maior grupo de dígitos plausível dentro de um range.
    Ajuste min/max conforme seu padrão real.
    """
    if not text:
        return None

    candidates = re.findall(r"\d+", text)
    if not candidates:
        return None

    # Quando o OCR separa dígitos por espaços/linhas, junta tudo
    if exact_digits is not None:
        all_digits = "".join(re.findall(r"\d", text))
        if len(all_digits) == exact_digits:
            return all_digits

    # Filtra por tamanho
    if exact_digits is not None:
        candidates = [c for c in candidates if len(c) == exact_digits]
    else:
        candidates = [c for c in candidates if min_digits <= len(c) <= max_digits]
    if not candidates:
        return None

    # Heurística: pega o mais longo (geralmente é o número principal)
    candidates.sort(key=len, reverse=True)
    return candidates[0]
