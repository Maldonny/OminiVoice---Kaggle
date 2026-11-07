# =====================================================================
# VMC VOICE - Versão Kaggle / Google Colab (OmniVoice + Gradio)
# =====================================================================
# Migrado do Chatterbox pro OmniVoice (k2-fsa): https://github.com/k2-fsa/OmniVoice
# Modelo zero-shot com suporte a 600+ idiomas (inclui português nativamente),
# licença Apache-2.0 e RTF ~0.025 (bem mais rápido que o Chatterbox).
#
# Para rodar no Kaggle/Colab:
# 1. Crie um novo Notebook no Kaggle e ative a aceleração por GPU (T4).
# 2. Cole e rode o seguinte comando em uma célula de instalação para configurar
#    e reiniciar o kernel automaticamente:
#    (Obs: o Kaggle às vezes já vem com uma versão mais nova de torchvision
#     pré-instalada, incompatível com o torch que instalamos aqui. Por isso
#     fixamos as 3 libs -- torch, torchvision e torchaudio -- juntas na mesma
#     combinação oficial, com --force-reinstall pra sobrescrever a que já
#     vinha no ambiente. Sem isso dá erro "operator torchvision::nms does
#     not exist" ao importar o omnivoice.
#     Também fixamos torchcodec==0.7, que é a versão compatível com torch
#     2.8.0 -- a versão 0.8+ do torchcodec exige torch 2.9+ e quebra com
#     "Could not load libtorchcodec" especificamente quando o Whisper tenta
#     transcrever automaticamente o áudio de referência, ex.: quando você
#     não preenche o campo de texto da referência.)
#
#    !pip install -q --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
#    !pip install -q --force-reinstall torchcodec==0.7
#    !pip install -q omnivoice gradio pydub soundfile Pillow
#    !apt-get -qq install -y ffmpeg
#    import os; os._exit(0)
#
# 3. Após o reinício automático da sessão, crie outra célula com este script
#    inteiro e execute.
# 4. Ao rodar, ele exibirá um link público temporário do Gradio (ex: https://xxxx.gradio.live).
#    Clique no link para abrir a interface do VMC Voice direto do seu navegador local!
#
# OBS: Na primeira execução, o OmniVoice baixa ~3.3GB de pesos do Hugging Face
# (k2-fsa/OmniVoice). Se quiser testar o fine-tune comunitário pra PT-BR,
# troque MODEL_NAME abaixo por "edwixx/omnivoice-brpt-v15" (não-oficial,
# qualidade não verificada pela Anthropic/k2-fsa, testar antes de confiar).
# =====================================================================

import os
import time
import threading
import subprocess
import numpy as np
import soundfile as sf
import torch
import gradio as gr
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from omnivoice import OmniVoice
from pydub import AudioSegment

MODEL_NAME = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000  # OmniVoice sempre gera áudio a 24kHz
VIDEO_WIDTH = 1920   # Resolução Full HD (Largura)
VIDEO_HEIGHT = 1080  # Resolução Full HD (Altura)
AUDIO_BITRATE = "320k"          # Qualidade máxima de áudio (320 kbps)
VIDEO_AUDIO_SAMPLE_RATE = "48000"  # Taxa de amostragem padrão de estúdio para vídeo (48 kHz)

# --- Túnel: usando o compartilhamento nativo do Gradio ---
# (ngrok estava retornando 524; a URL do Gradio é temporária mas confiável)
GRADIO_PORT = 7860

# 1. Escolher dispositivo GPU ou CPU
if torch.cuda.is_available():
    device_map = "cuda:0"
    device_label = "CUDA (GPU)"
else:
    device_map = "cpu"
    device_label = "CPU"
print(f"[*] Usando dispositivo: {device_label}")
if device_map == "cpu":
    print("[!] AVISO: GPU não detectada. O processamento será bem mais lento. Verifique se ativou a GPU T4 nas configurações do notebook.")

# 2. Carregar o modelo OmniVoice
print(f"[*] Carregando o modelo {MODEL_NAME}... (pode demorar no primeiro download, ~3.3GB)")
try:
    model = OmniVoice.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        dtype=torch.float16 if device_map != "cpu" else torch.float32,
    )
    print("[✓] Modelo carregado com sucesso!")
except Exception as e:
    print(f"[✗] Erro ao carregar o modelo: {e}")
    raise e

# 2.1 Heartbeat / Ping de saúde
# Mantém o kernel "ocupado" periodicamente pra evitar que o Kaggle/Colab
# encerrem a sessão por inatividade. Faz uma operação leve na GPU
# (não interfere na geração de áudio) e loga um timestamp no console.
def heartbeat(intervalo_segundos=240):
    while True:
        time.sleep(intervalo_segundos)
        try:
            if torch.cuda.is_available():
                _ = (torch.rand(8, 8, device="cuda") @ torch.rand(8, 8, device="cuda"))
                torch.cuda.synchronize()
            print(f"[♥] Heartbeat OK - {time.strftime('%d/%m/%Y %H:%M:%S')} - notebook ativo.")
        except Exception as e:
            print(f"[!] Heartbeat falhou: {e}")

threading.Thread(target=heartbeat, daemon=True).start()
print("[*] Heartbeat iniciado (ping a cada 4 minutos).")

# 3. Função auxiliar para dividir o texto em partes (chunks) consistentes
import re

def split_text_into_chunks(text, max_chars=150):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)

            if len(sentence) > max_chars:
                sub_parts = re.split(r'(?<=[,;:])\s+', sentence)
                temp_chunk = ""
                for part in sub_parts:
                    if len(temp_chunk) + len(part) + 1 <= max_chars:
                        if temp_chunk:
                            temp_chunk += " " + part
                        else:
                            temp_chunk = part
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = part
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Corte automático do áudio de referência: o OmniVoice recomenda 3-10s de
# referência (áudios mais longos deixam a inferência mais lenta e podem
# até piorar a clonagem). Se o áudio enviado for mais longo, corta os
# primeiros 10s automaticamente antes de passar pro modelo.
def preparar_ref_audio(ref_path, max_segundos=10.0):
    if not ref_path:
        return ref_path
    try:
        import soundfile as sf_check
        data, sr = sf_check.read(ref_path, dtype="float32")
        duracao = len(data) / sr
        if duracao > max_segundos:
            print(f"[*] Áudio de referência tem {duracao:.1f}s, cortando para os primeiros {max_segundos:.0f}s...")
            data_cortado = data[: int(max_segundos * sr)]
            ref_trimmed_path = "ref_audio_trimmed.wav"
            sf_check.write(ref_trimmed_path, data_cortado, sr)
            return ref_trimmed_path
    except Exception as e:
        print(f"[!] Não consegui cortar o áudio de referência automaticamente ({e}). Usando o original.")
    return ref_path

# NOTA IMPORTANTE sobre ref_text:
# O README do OmniVoice não documenta uma forma de recuperar o texto que o
# Whisper transcreveu automaticamente. Isso significa que, se você NÃO
# informar um texto de referência manualmente, o modelo vai rodar o Whisper
# de novo a cada chunk (mesmo áudio, retrabalho repetido) — funciona, mas
# desperdiça tempo de processamento à toa. Além disso, essa transcrição
# automática depende do torchcodec pra decodificar áudio internamente;
# se você tomar o erro "Could not load libtorchcodec", a causa mais comum
# é justamente essa (veja instruções de instalação no topo do arquivo).
# RECOMENDAÇÃO: transcreva o áudio de referência uma vez (o próprio campo
# abaixo na interface) e cole o texto manualmente no campo "Texto da
# referência" para pular a retranscrição em cada chunk.

# 3.2 Preparação da Imagem de Fundo em Full HD (1920x1080)
def preparar_imagem_fullhd(imagem_path, modo_ajuste="Preencher tela (Cortar bordas - Cover)"):
    """
    Prepara um quadro estático em resolução exata Full HD (1920x1080) usando Lanczos
    antes de enviar ao FFmpeg, garantindo máxima nitidez e codificação ultrarrápida.
    """
    target_size = (VIDEO_WIDTH, VIDEO_HEIGHT)
    temp_img_path = "fundo_fullhd_temp.png"

    if not imagem_path or not os.path.exists(imagem_path):
        # Se nenhuma imagem for enviada, cria um quadro escuro elegante em Full HD
        frame = Image.new("RGB", target_size, color=(18, 18, 24))
        frame.save(temp_img_path, format="PNG")
        return temp_img_path

    img = Image.open(imagem_path)
    img = ImageOps.exif_transpose(img).convert("RGB")

    if modo_ajuste == "Preencher tela (Cortar bordas - Cover)":
        frame = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
    elif modo_ajuste == "Ajustar inteira com fundo desfocado (Blur)":
        bg = ImageOps.fit(img, target_size, method=Image.Resampling.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=32))
        bg = ImageEnhance.Brightness(bg).enhance(0.6)
        fg = ImageOps.contain(img, target_size, method=Image.Resampling.LANCZOS)
        offset = ((VIDEO_WIDTH - fg.width) // 2, (VIDEO_HEIGHT - fg.height) // 2)
        bg.paste(fg, offset)
        frame = bg
    elif modo_ajuste == "Ajustar inteira com bordas pretas (Contain)":
        frame = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
    else:  # "Esticar imagem (1920x1080 exato)"
        frame = img.resize(target_size, resample=Image.Resampling.LANCZOS)

    frame.save(temp_img_path, format="PNG")
    return temp_img_path


# 3.3 Renderização de Vídeo MP4 Full HD (1920x1080) com Áudio na Melhor Qualidade Possível
def gerar_video_mp4_fullhd(wav_path, imagem_path, output_mp4, duracao_segundos, modo_ajuste):
    """
    Gera um arquivo .mp4 em Full HD (1920x1080, H.264 CRF 18 -tune stillimage)
    combinando a imagem de fundo e o áudio WAV PCM 24-bit sem perdas codificado em
    AAC 320 kbps / 48 kHz Estéreo (melhor qualidade suportada pelo container MP4).
    """
    temp_img_path = preparar_imagem_fullhd(imagem_path, modo_ajuste)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-framerate", "24",
            "-i", temp_img_path,
            "-i", wav_path,
            "-t", f"{duracao_segundos:.3f}",
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT},setsar=1",
            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,
            "-ar", VIDEO_AUDIO_SAMPLE_RATE,
            "-ac", "2",
            "-shortest",
            "-movflags", "+faststart",
            output_mp4,
        ]
        resultado = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if resultado.returncode != 0:
            raise RuntimeError(f"FFmpeg retornou erro ao gerar o vídeo:\n{resultado.stderr[-600:]}")
        return output_mp4
    finally:
        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except OSError:
                pass


# 4. Função de geração com chunking e barra de progresso do Gradio
def sintetizar_audio(
    texto,
    audio_referencia,
    texto_referencia,
    idioma,
    num_steps,
    velocidade,
    imagem_fundo,
    modo_imagem,
    formato_saida,
    progress=gr.Progress()
):
    if not texto.strip():
        return None, None, "Erro: Por favor, digite um roteiro para ser narrado."

    try:
        ref_path = preparar_ref_audio(audio_referencia) if audio_referencia else None
        ref_text = texto_referencia.strip() if texto_referencia and texto_referencia.strip() else None

        if ref_path and not ref_text:
            print("[!] Nenhum texto de referência informado — o Whisper vai transcrever "
                  "automaticamente a CADA chunk, o que deixa a geração mais lenta. "
                  "Preencha o campo de texto da referência pra evitar isso.")

        chunks = split_text_into_chunks(texto, max_chars=140)
        total_chunks = len(chunks)
        print(f"[*] Roteiro dividido em {total_chunks} partes para síntese consistente.")

        chunks_audio = []
        silence_samples = int(SAMPLE_RATE * 0.40)
        silence = np.zeros(silence_samples, dtype=np.float32)

        for i, chunk in enumerate(chunks):
            progresso_txt = f"Parte {i+1} de {total_chunks}... ({chunk[:25]}...)"
            progress((i) / total_chunks, desc=progresso_txt)
            print(f"[*] [{i+1}/{total_chunks}] Sintetizando: {chunk}")

            gen_kwargs = dict(
                text=chunk,
                language_id=idioma,
                num_step=int(num_steps),
                speed=float(velocidade),
            )
            if ref_path:
                gen_kwargs["ref_audio"] = ref_path
                if ref_text:
                    gen_kwargs["ref_text"] = ref_text

            audio_out = model.generate(**gen_kwargs)
            wav = audio_out[0]  # lista de np.ndarray (T,) a 24kHz

            chunks_audio.append(wav)
            chunks_audio.append(silence)

        if chunks_audio:
            chunks_audio.pop()  # remove o último silêncio

        progress(0.90, desc="Concatenando áudio mestre em 24-bit sem perdas...")

        final_wav = np.concatenate(chunks_audio)
        duracao_segundos = len(final_wav) / float(SAMPLE_RATE)

        # Salvar o WAV mestre em PCM 24-bit (sem perdas) para alimentar o vídeo e o MP3 com fidelidade máxima
        temp_wav = "fala_gerada_temp.wav"
        sf.write(temp_wav, final_wav, SAMPLE_RATE, subtype="PCM_24")

        gerar_mp4 = formato_saida in (
            "Vídeo MP4 (Full HD) + Áudio MP3",
            "Apenas Vídeo MP4 (Full HD)",
        )
        gerar_mp3 = formato_saida in (
            "Vídeo MP4 (Full HD) + Áudio MP3",
            "Apenas Áudio MP3 (320kbps)",
        )

        output_mp4 = None
        output_mp3 = None
        detalhes_exportacao = []

        try:
            if gerar_mp4:
                progress(0.94, desc="Renderizando vídeo MP4 em Full HD (1920x1080) com áudio 320kbps...")
                output_mp4 = "video_gerado_fullhd.mp4"
                gerar_video_mp4_fullhd(
                    wav_path=temp_wav,
                    imagem_path=imagem_fundo,
                    output_mp4=output_mp4,
                    duracao_segundos=duracao_segundos,
                    modo_ajuste=modo_imagem,
                )
                print(f"[✓] Vídeo Full HD (1920x1080) gerado com sucesso (AAC {AUDIO_BITRATE}, 48kHz)!")
                detalhes_exportacao.append(f"Vídeo MP4 Full HD (1920x1080, AAC {AUDIO_BITRATE}/48kHz)")

            if gerar_mp3:
                progress(0.98, desc="Exportando áudio MP3 em qualidade máxima (320kbps)...")
                output_mp3 = "fala_gerada.mp3"
                AudioSegment.from_file(temp_wav, format="wav").export(
                    output_mp3,
                    format="mp3",
                    bitrate=AUDIO_BITRATE,
                    parameters=["-ar", VIDEO_AUDIO_SAMPLE_RATE, "-q:a", "0"]
                )
                print(f"[✓] Áudio completo gerado com sucesso (MP3, {AUDIO_BITRATE}, 48kHz)!")
                detalhes_exportacao.append(f"Áudio MP3 ({AUDIO_BITRATE}/48kHz)")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        resumo_formatos = " e ".join(detalhes_exportacao)
        msg_sucesso = (
            f"Sucesso! Roteiro de {total_chunks} partes ({duracao_segundos:.1f}s) sintetizado "
            f"e exportado em: {resumo_formatos}."
        )
        return output_mp4, output_mp3, msg_sucesso

    except Exception as e:
        erro_str = str(e)
        if "torchcodec" in erro_str.lower() or "libtorchcodec" in erro_str.lower():
            erro_msg = (
                "Erro: falha ao transcrever automaticamente o áudio de referência "
                "(problema de compatibilidade do torchcodec). SOLUÇÃO RÁPIDA: preencha "
                "manualmente o campo 'Texto da referência' com a transcrição do áudio "
                "e gere novamente — isso evita o Whisper por completo. Se quiser corrigir "
                "de vez, rode na célula de instalação: "
                "!pip install -q --force-reinstall torchcodec==0.7"
            )
        else:
            erro_msg = f"Erro durante a geração: {erro_str}"
        print(f"[✗] {erro_msg}")
        return None, None, erro_msg

# 5. Construir Interface Gradio (Interface Web)
with gr.Blocks(title="VMC VOICE - OmniVoice / Vídeo Full HD & Voz IA") as demo:
    gr.Markdown("""
    # 🎙️🎬 VMC VOICE - OmniVoice (Clonagem de Voz + Vídeo Full HD 1080p)
    Estúdio de voz e vídeo IA rodando na nuvem com **OmniVoice** (600+ idiomas, licença Apache-2.0).
    Escreva o roteiro, envie a voz de referência para clonar, escolha uma **imagem de fundo** e gere seu vídeo **.MP4 em Full HD (1920x1080)** com a **melhor qualidade de áudio possível (320 kbps / 48 kHz)**.
    """)

    with gr.Row():
        with gr.Column():
            roteiro = gr.Textbox(
                label="Roteiro (Texto a ser falado)",
                placeholder="Cole seu texto aqui...",
                lines=8
            )

            with gr.Row():
                idioma = gr.Dropdown(
                    choices=["pt", "en", "es", "fr", "de", "it", "zh", "ja"],
                    value="pt",
                    label="Idioma da síntese"
                )

                formato_saida = gr.Dropdown(
                    choices=[
                        "Vídeo MP4 (Full HD) + Áudio MP3",
                        "Apenas Vídeo MP4 (Full HD)",
                        "Apenas Áudio MP3 (320kbps)"
                    ],
                    value="Vídeo MP4 (Full HD) + Áudio MP3",
                    label="Formato de Saída"
                )

            num_steps = gr.Slider(
                minimum=8,
                maximum=64,
                value=32,
                step=4,
                label="Passos de difusão (num_step). Padrão: 32. Use 16 para geração mais rápida (leve perda de qualidade)."
            )

            velocidade = gr.Slider(
                minimum=0.5,
                maximum=1.5,
                value=1.0,
                step=0.05,
                label="Velocidade da fala (speed). Padrão: 1.0."
            )

            imagem_fundo = gr.Image(
                label="🖼️ Imagem de Fundo do Vídeo (Opcional — gera vídeo MP4 em Full HD 1920x1080)",
                type="filepath",
                sources=["upload", "clipboard"]
            )

            modo_imagem = gr.Dropdown(
                choices=[
                    "Preencher tela (Cortar bordas - Cover)",
                    "Ajustar inteira com fundo desfocado (Blur)",
                    "Ajustar inteira com bordas pretas (Contain)",
                    "Esticar imagem (1920x1080 exato)"
                ],
                value="Preencher tela (Cortar bordas - Cover)",
                label="Enquadramento da Imagem em Full HD (1920x1080)"
            )

        with gr.Column():
            voz_ref = gr.Audio(
                label="Áudio de Referência (Voz a ser clonada, 3-10s recomendado)",
                type="filepath",
                sources=["microphone", "upload"]
            )

            texto_ref = gr.Textbox(
                label="Texto da referência (opcional, mas recomendado)",
                placeholder="Transcrição exata do áudio de referência acima. "
                            "Se deixar em branco, o Whisper transcreve automaticamente "
                            "a cada chunk gerado (mais lento).",
                lines=2
            )

            btn_gerar = gr.Button("🔥 Gerar Vídeo Full HD (.MP4) / Narração", variant="primary")

            video_saida = gr.Video(
                label="🎬 Vídeo Gerado (MP4 Full HD 1920x1080 • Áudio AAC 320kbps / 48kHz)"
            )
            audio_saida = gr.Audio(
                label="🎵 Áudio Gerado (MP3 320kbps / 48kHz)",
                type="filepath"
            )
            status = gr.Textbox(label="Status / Log de Execução", interactive=False)

    btn_gerar.click(
        fn=sintetizar_audio,
        inputs=[
            roteiro,
            voz_ref,
            texto_ref,
            idioma,
            num_steps,
            velocidade,
            imagem_fundo,
            modo_imagem,
            formato_saida
        ],
        outputs=[video_saida, audio_saida, status]
    )

    gr.Markdown("""
    *Dicas:*
    - **Clonagem de Voz**: O OmniVoice recomenda referências de **3 a 10 segundos** em ambiente silencioso e sem música de fundo. Se o texto de referência não for informado, o modelo transcreve automaticamente via Whisper.
    - **Vídeo em Full HD (.mp4)**: Envie qualquer imagem (horizontal, quadrada ou vertical) e escolha como ela deve se ajustar em **1920x1080**. O áudio do vídeo é codificado diretamente da matriz **WAV 24-bit sem perdas** para **AAC 320 kbps a 48 kHz Estéreo** (qualidade máxima).
    """)

# 6. Iniciar a aplicação e gerar o link público via túnel do próprio Gradio
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=GRADIO_PORT, debug=True)