## Agentes de Voz com OpenAI, LangChain e FastRTC

Este repositório contém **vários exemplos de agentes de voz** em Python, pensados para **demonstração em aula**:

- `voice_agent.py`: agente de voz simples (terminal) usando **OpenAI direto**.
- `langchain_demo.py`: chat em texto usando **LangChain + OpenAI**.
- `langchain_audio_agent.py`: agente de voz usando **áudio + LangChain + OpenAI** (opcional, se você criar esse arquivo).
- `fastrtc_openai_agent.py`: agente de voz em **tempo real via WebRTC** usando **FastRTC + OpenAI** (UI web Gradio).

Todos os exemplos usam **OpenAI** para LLM e, em alguns casos, para TTS (text-to-speech).

---

## 1. Pré-requisitos gerais

- Python 3.10+ (recomendado 3.10 ou 3.11)
- Sistema com microfone (para os exemplos de voz)
- Conta e chave de API da OpenAI

---

## 2. Setup do ambiente

### 2.1. Criar e ativar um ambiente virtual (opcional, mas recomendado)

No diretório do projeto:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows (PowerShell/CMD)
```

### 2.2. Instalar dependências

```bash
pip install -r requirements.txt
```

Se for usar o exemplo com FastRTC (recomendado instalar extras de STT/TTS):

```bash
pip install "fastrtc[stt,vad,tts]"
```

### 2.3. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto com:

```env
OPENAI_API_KEY=SUA_CHAVE_DA_OPENAI_AQUI
```

---

## 3. `voice_agent.py` – Agente de voz simples (terminal + OpenAI)

### O que ele faz

- Grava **5 segundos** de áudio do microfone.
- Envia o áudio para a API da OpenAI para **transcrição** (STT).
- Manda o texto transcrito para um modelo de chat (`gpt-4o-mini`).
- Gera uma resposta em texto.
- Converte a resposta em **áudio** via TTS da OpenAI e reproduz no alto-falante.

Fluxo didático para mostrar em aula:

> Microfone → STT → LLM → TTS → Caixas de som

### Como rodar

No diretório do projeto, com o venv ativado e `.env` configurado:

```bash
python voice_agent.py
```

No terminal:

- Pressione **ENTER** para iniciar uma gravação (5 segundos).
- Fale a sua pergunta.
- O script mostra:
  - Texto transcrito.
  - Resposta do modelo.
- Em seguida, reproduz a resposta em áudio.

---

## 4. `langchain_demo.py` – Demo de chat em texto com LangChain + OpenAI

### O que ele faz

- Cria um **LLM** com `ChatOpenAI` (LangChain).
- Define um **prompt** com:
  - `system`: instruções (responder em português, curto, claro).
  - `human`: placeholder `{pergunta}`.
- Monta uma **chain** (`prompt | llm`).
- Lê perguntas pelo terminal e responde em texto.

É ótimo para explicar:

- O conceito de **PromptTemplate** (`ChatPromptTemplate`).
- O pipeline `prompt -> modelo` como um “bloco” reutilizável.

### Como rodar

```bash
python langchain_demo.py
```

Uso:

- Digite sua pergunta após o prompt `Você:`.
- O agente responde em texto.
- Digite `sair` para encerrar.

---

## 5. `langchain_audio_agent.py` – Áudio + LangChain + OpenAI (opcional)

> OBS: Este arquivo é opcional; ele segue a mesma ideia do `voice_agent.py`, mas usando LangChain no “cérebro”.  
> Se você já o criou seguindo as instruções anteriores, aqui está o resumo de uso; se não, pode pular para a próxima seção.

### O que ele faz

- Grava áudio do microfone.
- Usa OpenAI para **transcrever** o áudio.
- Passa o texto transcrito para uma **chain do LangChain** (`ChatOpenAI` + `ChatPromptTemplate`).
- Recebe a resposta em texto.
- Converte a resposta em áudio via TTS da OpenAI e reproduz.

A ideia didática:

- Reforçar que o **bloco de raciocínio** (LangChain chain) é independente da entrada (voz ou texto).
- Mostrar como trocar “apenas” o frontend (voz vs. terminal).

### Como rodar (se existir no projeto)

```bash
python langchain_audio_agent.py
```

O comportamento é análogo ao `voice_agent.py`, mas o LLM é chamado via LangChain, não diretamente pela SDK.

---

## 6. `fastrtc_openai_agent.py` – Agente de voz em tempo real com FastRTC + OpenAI

### O que ele faz

Este exemplo usa a biblioteca **FastRTC**:

- Cria um **stream de áudio WebRTC** com UI web (Gradio).
- Usa **VAD (detecção de voz)** para detectar pausas de fala (classe `ReplyOnPause`).
- Para cada “turno de fala”:
  - Usa um modelo de STT do próprio FastRTC (`get_stt_model()`).
  - Envia o texto para a OpenAI (`gpt-4o-mini`) com um prompt em português.
  - Usa um modelo de TTS do FastRTC (`get_tts_model()`) para falar a resposta.

Fluxo:

> Navegador (microfone via WebRTC) → FastRTC (VAD + STT) → OpenAI (LLM) → FastRTC (TTS) → Navegador (áudio de volta)

Tudo isso com uma interface de botão de microfone fornecida pelo próprio FastRTC/Gradio.

### Estrutura resumida do código

- `get_stt_model()` e `get_tts_model()` carregam modelos de STT/TTS do FastRTC.
- `ReplyOnPause(echo)`:
  - Recebe áudio contínuo.
  - Detecta pausas.
  - Quando há uma pausa, chama a função `echo`.
- `echo(audio)`:
  - Chama o STT do FastRTC para converter áudio em texto.
  - Envia o texto para a OpenAI (`gpt-4o-mini`) com um prompt em português.
  - Itera sobre chunks de áudio gerados pelo TTS do FastRTC (`stream_tts_sync`) e dá `yield` para o Stream.
- `Stream(..., modality="audio", mode="send-receive")`:
  - Cria o stream de áudio bidirecional (envia e recebe áudio).
  - Gera a UI WebRTC + Gradio.

### Como rodar

1. Certificar-se de ter os extras instalados:

```bash
pip install "fastrtc[stt,vad,tts]"
```

2. Rodar o script:

```bash
python fastrtc_openai_agent.py
```

3. O FastRTC/Gradio vai abrir uma URL local (ex.: `http://127.0.0.1:7860`):

- Acesse no navegador.
- Clique no botão de microfone e fale.
- Ao terminar de falar (pausa), o VAD detecta o fim do turno:
  - O áudio é transcrito.
  - A transcrição é enviada para a OpenAI.
  - A resposta (em português) é sintetizada em áudio e reproduzida no próprio browser.

### Pontos didáticos interessantes

- **VAD e turn-taking**: `ReplyOnPause` mostra como detectar pausas naturalmente, sem precisar apertar “enviar”.
- **Separação de responsabilidades**:
  - FastRTC cuida de **WebRTC + UI + VAD + STT + TTS**.
  - OpenAI cuida de **entendimento e geração de texto**.
- Comparar esse exemplo com `voice_agent.py`:
  - `voice_agent.py` faz tudo no terminal (sem WebRTC/UI).
  - `fastrtc_openai_agent.py` faz uma experiência mais próxima de um **assistente de voz em navegador**.

---

## 7. Sugestão de ordem para mostrar em aula

1. **`langchain_demo.py` (texto)**  
   - Conceito de LLM, prompt, LangChain, chain.

2. **`voice_agent.py` (voz no terminal)**  
   - Pipeline STT → LLM → TTS sem WebRTC.
   - Mostrar limitações (sem UI, interação por turnos fixos).

3. **`fastrtc_openai_agent.py` (voz em tempo real via navegador)**  
   - Introduzir FastRTC e WebRTC.
   - Mostrar VAD e UI mais realista de “assistente de voz”.

4. (Opcional) **`langchain_audio_agent.py`**  
   - Mostrar como o mesmo “miolo” de LangChain é plugável em diferentes frontends (terminal vs. voz).

---

## 8. Problemas comuns e como resolver

### 8.1. PortAudio não encontrado

Erro:

```text
OSError: PortAudio library not found
```

No Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev libportaudio2 libportaudiocpp0
pip install --force-reinstall sounddevice
```

### 8.2. Modelos de áudio da OpenAI/TTS

Se algum modelo der erro (“modelo não encontrado”), ajuste o nome do modelo em:

- `voice_agent.py`
- `langchain_audio_agent.py`

Para algum dos modelos de TTS/STT disponíveis na sua conta (`gpt-4o-transcribe`, `whisper-1`, `tts-1`, etc.).

### 8.3. FastRTC pedindo extras de STT/TTS

Erro vindo do `get_stt_model` ou `get_tts_model`:

```text
ImportError: Install fastrtc[stt] for speech-to-text and stopword detection support.
```

Ou algo similar para TTS.

Instale:

```bash
pip install "fastrtc[stt,vad,tts]"
```

---

## 9. Extensões possíveis

- Adaptar o LLM para outros provedores (SambaNova, Groq, etc.) trocando apenas o cliente e o `model=...`.
- Integrar **LangChain** dentro do `echo` do FastRTC (em vez de chamar OpenAI direto), reaproveitando a `chain` do `langchain_demo.py`.
- Criar uma interface Streamlit/Gradio adicional para visualizar também o texto transcrito e as respostas, lado a lado com o áudio.