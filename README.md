# Processamento de fotos por OCR

Ferramenta Python que detecta o número de inscrição em fotos com Tesseract e grava os arquivos de saída com o número no nome.

## Instalação

Use Python 3.10 ou posterior, pip e o Tesseract. Instale as dependências Python com `python -m pip install -r requirements.txt`.

Você precisa do Tesseract instalado no sistema:
- Ubuntu/Debian: sudo apt-get install tesseract-ocr
- Windows: instale o Tesseract e, se preciso, passe --tesseract-cmd

## Uso
python main.py --input ./fotos --output ./saida

Para acelerar, use múltiplas threads:
python main.py --input ./fotos --output ./saida --workers 4

Se o tesseract não estiver no PATH (comum no Windows):
python main.py --input ./fotos --output ./saida --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"

Ajustes:
- --psm 7 ou 6 costuma funcionar bem
- --min-digits / --max-digits conforme o padrão do número
- use --no-digits-only se houver letras no número
- --photo-timeout 60 define tempo máximo por foto (segundos). Use 0 para desativar.
- --log /caminho/arquivo.txt para salvar o log em um local específico.
- Se o código for sempre do mesmo tamanho (ex.: 6), ajuste `exact_digits` em `config.py`.

Observação:
- Um log de execução é salvo em `saida/` com nome `processamento-YYYYMMDD-HHMMSS.txt`.
- Se a inscrição for vermelha em fundo branco, o sistema tenta localizar o texto vermelho em qualquer parte da imagem.
  Para desligar isso, ajuste `prefer_red_text=False` em `config.py`.
- Imagens sem número identificado também são copiadas para a saída com prefixo `nao-identificado-`.
- Imagens que estouram o tempo máximo são copiadas para a saída com prefixo `timeout-`.
- Em imagens mais borradas, há uma segunda tentativa com pré-processamento mais agressivo
  (configure em `use_strong_preprocess_on_fail`).
- OCR também tenta múltiplos modos de segmentação (PSM 7/8/6/13) quando falha no primeiro.
- Se há candidatos conflitantes (mesmo tamanho), o sistema prefere marcar como não identificado
  do que aceitar um número incerto.
- Para ganhar velocidade: desative `use_strong_preprocess_on_fail` e/ou `fallback_full_image`.
