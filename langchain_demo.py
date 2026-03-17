import io
import os

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write as write_wav, read as read_wav

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

SAMPLE_RATE = 16_000
DURATION = 5  # segundos de gravação


def setup_clients():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY não encontrada. Defina no .env ou no ambiente."
        )

    # Cliente “baixo nível” da OpenAI (para áudio)
    openai_client = OpenAI()

    # LLM via LangChain
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # ajuste se quiser outro
        temperature=0.3,
    )

    # Prompt do LangChain (system + human)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um assistente de voz educado que responde em português de forma curta e clara.",
            ),
            ("human", "{pergunta}"),
        ]
    )

    # chain = prompt -> llm
    chain = prompt | llm
    return openai_client, chain


def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
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


def audio_to_wav_bytes(sample_rate, audio):
    buf = io.BytesIO()
    write_wav(buf, sample_rate, audio)
    buf.seek(0)
    return buf


def transcribe_audio(openai_client: OpenAI, wav_bytes: io.BytesIO) -> str:
    print("Enviando áudio para transcrição...")
    audio_file = ("audio.wav", wav_bytes.read())

    transcription = openai_client.audio.transcriptions.create(
        model="gpt-4o-transcribe",  # ou "whisper-1"
        file=audio_file,
        language="pt",
    )
    text = transcription.text.strip()
    print(f"Transcrição: {text!r}")
    return text


def run_langchain(chain, texto: str) -> str:
    print("Chamando LangChain (LLM)...")
    resposta = chain.invoke({"pergunta": texto})
    answer = resposta.content.strip()
    print(f"Resposta do LangChain: {answer!r}")
    return answer


def tts_speak(openai_client: OpenAI, text: str, filename: str = "lc_response.wav"):
    print("Gerando áudio da resposta (TTS via OpenAI)...")

    # Ajuste o modelo para o TTS disponível na sua conta se necessário:
    with openai_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",  # ou "tts-1", "tts-1-hd", etc.
        voice="alloy",
        input=text,
    ) as response:
        response.stream_to_file(filename)

    print("Reproduzindo resposta...")
    sr, data = read_wav(filename)

    if data.dtype != np.float32:
        data = data.astype(np.float32) / max(
            abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max
        )

    sd.play(data, sr)
    sd.wait()


def main():
    print("=== Demo de Áudio + LangChain + OpenAI ===")
    print("Pressione ENTER para gravar, ou digite 'q' + ENTER para sair.")

    try:
        openai_client, chain = setup_clients()
    except RuntimeError as e:
        print(f"Erro de configuração: {e}")
        return

    while True:
        cmd = input("\nAção (ENTER = gravar, q = sair): ").strip().lower()
        if cmd == "q":
            print("Encerrando.")
            break

        # 1) Gravar áudio
        try:
            sr, audio = record_audio()
        except Exception as e:
            print(f"Erro ao gravar áudio: {e}")
            continue

        # 2) Converter para WAV em memória
        wav_buf = audio_to_wav_bytes(sr, audio)

        # 3) Transcrever
        try:
            texto = transcribe_audio(openai_client, wav_buf)
        except Exception as e:
            print(f"Erro na transcrição: {e}")
            continue

        if not texto:
            print("Nada reconhecido, tente novamente.")
            continue

        # 4) Passar o texto para a chain do LangChain
        try:
            answer = run_langchain(chain, texto)
        except Exception as e:
            print(f"Erro ao chamar LangChain: {e}")
            continue

        # 5) TTS da resposta
        try:
            tts_speak(openai_client, answer)
        except Exception as e:
            print(f"Erro no TTS: {e}")
            print("Resposta apenas em texto:", answer)


if __name__ == "__main__":
    main()