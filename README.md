### Instalação
pip install -r requirements.txt

Você precisa do Tesseract instalado no sistema:
- Ubuntu/Debian: sudo apt-get install tesseract-ocr
- Windows: instale o Tesseract e, se preciso, passe --tesseract-cmd

### Uso
python main.py --input ./fotos --output ./saida

Para acelerar, use múltiplas threads:
python main.py --input ./fotos --output ./saida --workers 4

Se o tesseract não estiver no PATH (comum no Windows):
python main.py --input ./fotos --output ./saida --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"

Ajustes:
- --psm 7 ou 6 costuma funcionar bem
- --min-digits / --max-digits conforme o padrão do número
- use --no-digits-only se houver letras no número
- Se o código for sempre do mesmo tamanho (ex.: 6), ajuste `exact_digits` em `config.py`.

Observação:
- Se a inscrição for vermelha em fundo branco, o sistema tenta localizar o texto vermelho em qualquer parte da imagem.
  Para desligar isso, ajuste `prefer_red_text=False` em `config.py`.
- Imagens sem número identificado também são copiadas para a saída com prefixo `nao-identificado-`.
- Em imagens mais borradas, há uma segunda tentativa com pré-processamento mais agressivo
  (configure em `use_strong_preprocess_on_fail`).
- OCR também tenta múltiplos modos de segmentação (PSM 7/8/6/13) quando falha no primeiro.
- Se há candidatos conflitantes (mesmo tamanho), o sistema prefere marcar como não identificado
  do que aceitar um número incerto.
- Para ganhar velocidade: desative `use_strong_preprocess_on_fail` e/ou `fallback_full_image`.
