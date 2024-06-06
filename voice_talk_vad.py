# Dependencies
import pyaudio, wave, numpy, collections, faster_whisper, torch.cuda
from openai import OpenAI
from elevenlabs import stream as elevenlabs_stream, Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
from play_audio import play_audio

# Utils
import json
import time
import os
from dotenv import load_dotenv


# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Define clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Load persona file
def load_persona(file_path):
    with open(file_path, 'r') as file:
        persona = json.load(file)
    return persona
persona_file = 'persona.json'
persona = load_persona(persona_file)

# Define initial system prompt for LLM
system_prompt = {
    "role": "system",
    "content": " ".join(persona),
}

model, answer, history = (
    faster_whisper.WhisperModel(
        model_size_or_path="tiny.en",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ),
    "",
    [],
)


def generate(messages):
    global answer
    answer = ""

    # GPT Answer Stream으로 생성되게 하는 코드 -> 나중에 구현해보기
    # for chunk in openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo", messages=messages, stream=True
    # ):
    #     # print(chunk)
    #     if text_chunk := chunk["choices"][0]["delta"].get("content"):
    #         answer += text_chunk
    #         print(text_chunk, end="", flush=True)
    #         yield text_chunk

    # print("messages :", messages)
    response = openai_client.chat.completions.create(
        model="gpt-4o",  # 또는 다른 모델을 사용
        messages=messages,
    )
    response = response.json()
    response = json.loads(response)
    new_answer = response["choices"][0]["message"]["content"]
    print(new_answer)
    return new_answer


def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = numpy.abs(numpy.frombuffer(data, dtype=numpy.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level


while True:
    audio = pyaudio.PyAudio()
    stream = audio.open(
        rate=16000,
        format=pyaudio.paInt16,
        channels=1,
        input=True,
        frames_per_buffer=512,
    )
    audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
    frames, long_term_noise_level, current_noise_level, voice_activity_detected = (
        [],
        0.0,
        0.0,
        False,
    )

    # Play Initiating Audio
    print("\n\nStart speaking. ", end="", flush=True)

    while True:
        data = stream.read(512)
        pegel, long_term_noise_level, current_noise_level = get_levels(
            data, long_term_noise_level, current_noise_level
        )
        audio_buffer.append(data)

        if voice_activity_detected:
            frames.append(data)
            if current_noise_level < ambient_noise_level + 100:
                break  # voice actitivy ends

        if (
            not voice_activity_detected
            and current_noise_level > long_term_noise_level + 300
        ):
            voice_activity_detected = True
            print("I'm all ears.\n")
            ambient_noise_level = long_term_noise_level
            frames.extend(list(audio_buffer))

    stream.stop_stream(), stream.close(), audio.terminate()
    # Start measuring time when the speech has ended
    start_time = time.time()

    # Transcribe recording using whisper
    with wave.open("voice_record.wav", "wb") as wf:
        wf.setparams(
            (1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, "NONE", "NONE")
        )
        wf.writeframes(b"".join(frames))
    user_text = " ".join(
        seg.text for seg in model.transcribe("voice_record.wav", language="en")[0]
    )
    print(f">>>{user_text}\n<<< ", end="", flush=True)
    history.append({"role": "user", "content": user_text})

    # Generate and stream output
    # generator = generate([system_prompt] + history[-10:]) # 최근 10개까지만 입력
    generator = generate([system_prompt] + history)  # 전체 history 다 입력

    audio_stream = elevenlabs_client.generate(
        text=generator,
        model="eleven_monolingual_v1",
        voice=Voice(
            # voice_id="QELRzhyCS20Z5NK8HJoL", # Preacher
            voice_id="1pTeUJPQLqnJpo6bZVeo",  # ESG clip
            settings=VoiceSettings(
                stability=0.71, similarity_boost=1.0, style=0.0, use_speaker_boost=True
            ),
        ),
        stream=True,
    )
    
    # Calculate and print response time
    end_time = time.time()
    duration = end_time - start_time
    print("(Time elapsed: ", duration, "s)")

    # Play buffer audio/words
    audio_path = "audio/funk.mp3"
    play_audio(audio_path)

    # Play TTS result
    elevenlabs_stream(audio_stream)

    history.append({"role": "assistant", "content": answer})
