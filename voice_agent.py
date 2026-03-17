import io
import os

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write as write_wav, read as read_wav


SAMPLE_RATE = 16_000  # Hz
DURATION = 5  # segundos de gravação padrão


def setup_client() -> OpenAI:
    """
    Inicializa o cliente da OpenAI.

    Certifique-se de definir a variável de ambiente OPENAI_API_KEY
    (por exemplo, via arquivo .env ou export no shell).
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não encontrada. "
            "Defina no ambiente ou em um arquivo .env na mesma pasta."
        )
    # A SDK da OpenAI já lê OPENAI_API_KEY do ambiente;
    # apenas instanciamos o cliente.
    return OpenAI()


def record_audio(duration: int = DURATION, sample_rate: int = SAMPLE_RATE):
    """Grava áudio do microfone e retorna (sample_rate, numpy_array)."""
    print(f"\nGravando por {duration} segundos... Fale agora!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("Gravação encerrada.")
    return sample_rate, audio


def audio_to_wav_bytes(sample_rate: int, audio) -> io.BytesIO:
    """Converte o numpy array de áudio em bytes de WAV (em memória)."""
    buf = io.BytesIO()
    write_wav(buf, sample_rate, audio)
    buf.seek(0)
    return buf


def transcribe_audio(client: OpenAI, wav_bytes: io.BytesIO) -> str:
    """Envia áudio para a API de transcrição (Whisper / gpt-4o-transcribe)."""
    print("Enviando áudio para transcrição...")
    audio_file = ("audio.wav", wav_bytes.read())

    # Use o modelo que sua conta suportar: "whisper-1" ou "gpt-4o-transcribe"
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
        language="pt",  # força reconhecimento em português
    )
    text = transcription.text.strip()
    print(f"Transcrição: {text!r}")
    return text


def ask_llm(client: OpenAI, user_text: str) -> str:
    """Manda o texto reconhecido para o modelo de chat."""
    print("Enviando texto para o modelo de linguagem...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um assistente de voz educado que responde em "
                    "português de forma curta e clara."
                ),
            },
            {"role": "user", "content": user_text},
        ],
    )
    answer = response.choices[0].message.content.strip()
    print(f"Resposta do modelo: {answer!r}")
    return answer


def tts_speak(client: OpenAI, text: str, filename: str = "response.wav"):
    """
    Gera áudio da resposta usando TTS da OpenAI e toca com sounddevice.

    Ajuste o nome do modelo (por exemplo, 'gpt-4o-mini-tts' ou outro)
    conforme o que estiver disponível na sua conta.
    """
    print("Gerando áudio da resposta (TTS)...")

    # IMPORTANTE: verifique na sua conta/documentação o modelo TTS correto.
    # A forma recomendada na SDK atual é usar streaming_response + stream_to_file.
    from contextlib import closing

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",  # ajuste conforme modelos TTS disponíveis na sua conta
        voice="alloy",
        input=text,
    ) as response:
        response.stream_to_file(filename)

    print("Reproduzindo resposta...")
    sr, data = read_wav(filename)

    # Converte para float32 no intervalo [-1, 1], se necessário
    if data.dtype != np.float32:
        data = data.astype(np.float32) / max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)

    sd.play(data, sr)
    sd.wait()


def main():
    print("=== Agente de Voz com OpenAI (Demo) ===")
    print("Certifique-se de ter definido OPENAI_API_KEY no ambiente ou em um arquivo .env.")
    print("Pressione ENTER para gravar, ou digite 'q' + ENTER para sair.")

    try:
        client = setup_client()
    except RuntimeError as e:
        print(f"Erro de configuração: {e}")
        return

    while True:
        cmd = input("\nAção (ENTER = gravar, q = sair): ").strip().lower()
        if cmd == "q":
            print("Encerrando agente.")
            break

        # 1) Gravar áudio
        try:
            sample_rate, audio = record_audio()
        except Exception as e:
            print(f"Erro ao gravar áudio: {e}")
            continue

        # 2) Converter para WAV em memória
        wav_buf = audio_to_wav_bytes(sample_rate, audio)

        # 3) Transcrever
        try:
            user_text = transcribe_audio(client, wav_buf)
        except Exception as e:
            print(f"Erro na transcrição: {e}")
            continue

        if not user_text:
            print("Nada foi reconhecido, tente novamente.")
            continue

        # 4) Perguntar ao modelo de linguagem
        try:
            answer = ask_llm(client, user_text)
        except Exception as e:
            print(f"Erro ao chamar o modelo de linguagem: {e}")
            continue

        # 5) Gerar TTS e falar
        try:
            tts_speak(client, answer)
        except Exception as e:
            print(f"Erro ao gerar ou reproduzir TTS: {e}")
            print("Mostrando resposta apenas em texto.")
            print("Resposta:", answer)


if __name__ == "__main__":
    main()

