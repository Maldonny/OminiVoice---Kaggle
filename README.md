# 🎙️ VMC Voice — Clonagem de Voz & TTS com OmniVoice

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange?logo=gradio)](https://gradio.app/)
[![Kaggle GPU](https://img.shields.io/badge/Platform-Kaggle%20%2F%20Colab%20GPU-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![OmniVoice](https://img.shields.io/badge/Model-OmniVoice%20(k2--fsa)-green)](https://github.com/k2-fsa/OmniVoice)

Estúdio interativo de **Text-to-Speech (TTS)** e **Clonagem de Voz Zero-Shot** de alta performance baseado no [OmniVoice (k2-fsa)](https://github.com/k2-fsa/OmniVoice), desenvolvido para execução acelerada em nuvem (**Kaggle GPU T4** e **Google Colab**) com interface web acessível via navegador por meio do **Gradio**.

---

## 📑 Sumário

- [Visão Geral](#-visão-geral)
- [Principais Funcionalidades](#-principais-funcionalidades)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Guia Rápido de Instalação e Execução](#-guia-rápido-de-instalação-e-execução)
  - [Opção 1: Executando no Kaggle (Recomendado)](#opção-1-executando-no-kaggle-recomendado)
  - [Opção 2: Executando no Google Colab](#opção-2-executando-no-google-colab)
  - [Opção 3: Executando em Ambiente Local com GPU](#opção-3-executando-em-ambiente-local-com-gpu)
- [Como Usar a Interface](#-como-usar-a-interface)
- [Parâmetros e Dicas de Desempenho](#-parâmetros-e-dicas-de-desempenho)
- [Solução de Problemas Comuns (Troubleshooting)](#-solução-de-problemas-comuns-troubleshooting)
- [Licença](#-licença)
- [Agradecimentos e Créditos](#-agradecimentos-e-créditos)

---

## 🌟 Visão Geral

O projeto disponibiliza uma solução pronta para narrar roteiros e clonar timbres de voz com fidelidade e altíssima velocidade (RTF de ~0.025 no OmniVoice).

Ao contrário de modelos convencionais pesados ou complexos de configurar, este ambiente foi lapidado para:
1. Resolver automaticamente as dependências estritas do ecossistema PyTorch no Kaggle/Colab (`torch`, `torchvision`, `torchaudio`, `torchcodec`).
2. Permitir síntese de roteiros longos sem cortes bruscos através de particionamento dinâmico (*chunking*).
3. Gerar **vídeos em Full HD (`1920x1080`, `.mp4`)** com imagem de fundo personalizada e **qualidade máxima de áudio (AAC 320 kbps / 48 kHz Estéreo)** a partir de matriz **WAV PCM 24-bit sem perdas**, além de exportação em **MP3 de alta definição (320 kbps)**.
4. Manter o kernel ativo em segundo plano com um mecanismo de **Heartbeat** anti-inatividade.

---

## 🚀 Principais Funcionalidades

- **Clonagem Zero-Shot Rápida**: Requer apenas entre **3 e 10 segundos** de áudio de referência limpo para clonar qualquer timbre de voz.
- **Multilíngue Nativo**: Suporte oficial a mais de 600 idiomas (incluindo Português `pt`, Inglês `en`, Espanhol `es`, Francês `fr`, Alemão `de`, Italiano `it`, etc.).
- **Particionamento Inteligente (*Chunking*)**: Divide textos longos respeitando pontuações e limites de caracteres (~140 caracteres por trecho), inserindo pausas naturais de silêncio (400 ms) entre cada sentença.
- **Corte Automático de Referência**: Otimiza automaticamente áudios de entrada longos, recortando os primeiros 10 segundos para máxima fidelidade e menor consumo de VRAM.
- **Exportação de Vídeo em Full HD (`.mp4` 1080p)**: Permite enviar uma **imagem de fundo** e gera um vídeo `.mp4` em resolução **Full HD (`1920x1080`)** com modos de enquadramento (Preencher/Cover, Fundo Desfocado/Blur, Bordas Pretas/Contain ou Esticar).
- **Áudio na Melhor Qualidade Possível**: A síntese é consolidada em **WAV PCM 24-bit sem perdas** e codificada diretamente em **AAC 320 kbps a 48 kHz Estéreo** no vídeo `.mp4` e em **MP3 320 kbps (48 kHz)** no arquivo de áudio.
- **Túnel Seguro Gradio**: Gera uma URL pública temporária (`https://xxxx.gradio.live`) sem a necessidade de contas ou túneis externos com limite de requisição.
- **Heartbeat Anti-Timeout**: Thread leve que envia micropulsos para a GPU a cada 4 minutos, prevenindo encerramentos prematuros por inatividade da sessão de nuvem.

---

## 📁 Estrutura do Repositório

```text
├── Code Python/
│   ├── ### INSTALAÇÃO ###.txt       # Comandos com versões exatas de dependências
│   └── clonagem_omini.py           # Aplicação principal (inferência + Gradio)
├── Notebook Pronto/
│   └── clone-de-voz-omni-voice.ipynb # Notebook Jupyter configurado para o Kaggle
├── LICENSE                         # Licença Apache 2.0
└── README.md                       # Documentação do projeto
```

---

## ⚙️ Guia Rápido de Instalação e Execução

### Opção 1: Executando no Kaggle (Recomendado)

1. Crie um novo Notebook no [Kaggle](https://www.kaggle.com/) ou importe o arquivo [`Notebook Pronto/clone-de-voz-omni-voice.ipynb`](Notebook%20Pronto/clone-de-voz-omni-voice.ipynb).
2. No painel lateral direito (*Notebook Options*), certifique-se de configurar:
   - **Accelerator**: `GPU T4 x2` ou `GPU P100`.
   - **Internet**: `Internet On` (necessário para baixar os pesos do modelo).
3. **Célula 1 — Instalação e Reinicialização do Kernel**:
   Execute o bloco abaixo para instalar as versões compatíveis de GPU e reiniciar o ambiente:
   ```python
   !pip install -q --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   !pip install -q --force-reinstall torchcodec==0.7
   !pip install -q omnivoice gradio pydub soundfile Pillow
   !apt-get -qq install -y ffmpeg
   import os; os._exit(0)
   ```
4. **Célula 2 — Execução da Aplicação**:
   Após o reinício automático da sessão, cole e execute o código de [`Code Python/clonagem_omini.py`](Code%20Python/clonagem_omini.py).
5. O terminal exibirá um link público temporário do Gradio (exemplo: `https://xxxx.gradio.live`). Basta clicar no link para usar o estúdio no seu navegador.

---

### Opção 2: Executando no Google Colab

1. Crie um notebook no [Google Colab](https://colab.research.google.com/).
2. Acesse **Ambiente de execução** > **Alterar tipo de ambiente de execução** e selecione **T4 GPU**.
3. Execute a célula de instalação descrita no passo acima.
4. Execute o script [`clonagem_omini.py`](Code%20Python/clonagem_omini.py).

---

### Opção 3: Executando em Ambiente Local com GPU

Para rodar em máquina local com GPU NVIDIA (CUDA):

```bash
# Clone o repositório
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd "OminiVoice - Kaggle"

# Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Linux/macOS
# ou: venv\Scripts\activate no Windows

# Instale os pacotes compatíveis
pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install --force-reinstall torchcodec==0.7
pip install omnivoice gradio pydub soundfile Pillow

# Certifique-se de ter o ffmpeg instalado no sistema
# Ubuntu/Debian: sudo apt install ffmpeg
# Windows: winget install Gyan.FFmpeg

# Inicie a aplicação
python "Code Python/clonagem_omini.py"
```

---

## 🎛️ Como Usar a Interface

1. **Roteiro (Texto a ser falado)**: Digite ou cole o texto que deseja sintetizar. O sistema aceita desde frases curtas até roteiros completos com múltiplos parágrafos.
2. **Idioma da síntese e Formato de Saída**:
   - Escolha o código do idioma desejado (ex.: `pt` para Português, `en` para Inglês).
   - Escolha se deseja exportar **Vídeo MP4 (Full HD) + Áudio MP3**, **Apenas Vídeo MP4 (Full HD)** ou **Apenas Áudio MP3 (320kbps)**.
3. **Imagem de Fundo do Vídeo (Opcional)**:
   - Envie qualquer imagem para servir de plano de fundo do vídeo `.mp4` em **Full HD (`1920x1080`)** (se não enviar, o sistema gera automaticamente um quadro escuro em 1080p).
   - Selecione o **Enquadramento da Imagem em Full HD**: *Preencher tela (Cover)*, *Ajustar inteira com fundo desfocado (Blur)*, *Ajustar inteira com bordas pretas (Contain)* ou *Esticar imagem*.
4. **Áudio de Referência (Opcional para clonagem)**:
   - Grave diretamente pelo microfone ou faça upload de um arquivo de áudio (`.wav`, `.mp3`).
   - Recomenda-se entre **3 a 10 segundos** de fala clara e sem música de fundo.
5. **Texto da referência (Altamente Recomendado)**:
   - Digite a transcrição exata das palavras ditas no áudio de referência.
   - Isso elimina a necessidade de o Whisper transcrever repetidamente o áudio a cada bloco de texto, tornando a geração muito mais rápida.
6. **Ajustes Avançados**:
   - **Passos de difusão (`num_steps`)**: `32` (qualidade balanceada) ou `16` (geração ultrarrápida).
   - **Velocidade da fala (`speed`)**: Ajuste entre `0.5` e `1.5` (padrão: `1.0`).
7. Clique em **🔥 Gerar Vídeo Full HD (.MP4) / Narração** e acompanhe a barra de progresso em tempo real.

---

## 💡 Parâmetros e Dicas de Desempenho

| Parâmetro | Padrão | Recomendação | Descrição |
|---|---|---|---|
| `num_step` | `32` | `16` a `32` | Define a quantidade de passos de difusão. Valores menores geram mais rápido; valores maiores refinam a clareza acústica. |
| `speed` | `1.0` | `0.85` a `1.15` | Controla o ritmo da fala sem distorcer o tom da voz clonada. |
| Duração da Referência | - | `3s - 10s` | O OmniVoice opera melhor com áudios curtos e limpos. Áudios acima de 10s são cortados automaticamente. |
| Transcrição Manual | - | Sempre preencher | Evita overhead do Whisper a cada *chunk* e contorna possíveis instabilidades do `torchcodec`. |

> **Nota sobre modelos comunitários**: Caso deseje experimentar *checkpoints* alternativos (como modelos afinados para Português Brasileiro), você pode alterar a variável `MODEL_NAME` no início de `clonagem_omini.py`, por exemplo para `"edwixx/omnivoice-brpt-v15"`.

---

## 🛠️ Solução de Problemas Comuns (Troubleshooting)

### 1. `Could not load libtorchcodec` ou erro na transcrição automática
- **Causa**: Versão incompatível do pacote `torchcodec` com a versão do PyTorch instalada.
- **Solução Rápida**: Preencha manualmente o campo **"Texto da referência"** com o que foi dito no áudio. Isso pula completamente a biblioteca de decodificação interna do Whisper.
- **Solução Definitiva**: Reinstale fixando a versão `0.7`:
  ```bash
  pip install -q --force-reinstall torchcodec==0.7
  ```

### 2. `operator torchvision::nms does not exist`
- **Causa**: Conflito entre a versão padrão do `torchvision` presente nas imagens do Kaggle e a versão do `torch`.
- **Solução**: Sempre instale em conjunto as versões fixadas na mesma chamada:
  ```bash
  pip install -q --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  ```

### 3. Sessão desconectando por inatividade
- O script já inclui um serviço de *Heartbeat* que roda a cada 4 minutos. Certifique-se de não interromper o processo no terminal e mantenha a aba do notebook aberta ou em segundo plano.

---

## 📄 Licença

Este projeto é distribuído sob a licença de código aberto **Apache License 2.0**. Consulte o arquivo [LICENSE](LICENSE) para obter mais informações.

O modelo base utilizado [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) também é licenciado sob a licença **Apache-2.0**.

---

## 🤝 Agradecimentos e Créditos

- Ao time do [k2-fsa](https://github.com/k2-fsa) pelo desenvolvimento e disponibilização do modelo [OmniVoice](https://github.com/k2-fsa/OmniVoice).
- À comunidade [Gradio](https://gradio.app/) pela biblioteca intuitiva de interfaces web.
- À plataforma [Kaggle](https://www.kaggle.com/) pelo fornecimento de infraestrutura de GPUs para experimentação.
