import os

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
stt_model = get_stt_model()
tts_model = get_tts_model()

def echo(audio):
    prompt = stt_model.stt(audio)
    response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "Você é um assistente de voz que SEMPRE responde em português do Brasil, de forma natural e curta."
        },
        {"role": "user", "content": prompt},
    ],
    max_tokens=200,
)       
    answer = response.choices[0].message.content
    for audio_chunk in tts_model.stream_tts_sync(answer):
        yield audio_chunk

stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    stream.ui.launch()