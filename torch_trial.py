import librosa
from faster_whisper import WhisperModel

print("Загружаю аудио...")
audio, sr = librosa.load("Запись.m4a", sr=16000)

print("Сохраняю как WAV...")
import soundfile as sf
sf.write("Запись.wav", audio, sr)

print("Загружаю модель...")
model = WhisperModel("tiny", device="cuda", compute_type="float16")

print("Транскрибирую...")
segments, info = model.transcribe("Запись.wav", language="ru")

print("\n--- РЕЗУЛЬТАТЫ ---\n")
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")