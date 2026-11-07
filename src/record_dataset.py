
import sounddevice as sd
import soundfile as sf
import os
from datetime import datetime

sample_rate = 44100
duration = 5
device = 1

categories = ["scream", "shouting", "laughing", "normal_speech"]

while True:

    print("\nAvailable categories:")
    print("1. normal_speech")
    print("2. shouting")
    print("3. laughing")
    print("4. scream")
    print("Type 'stop' to exit")

    label = input("\nEnter audio category: ").strip().lower()

    if label == "stop":
        print("Recording session completed!")
        break

    if label not in categories:
        print("Invalid category! Try again.")
        continue

    folder = f"dataset/{label}"
    os.makedirs(folder, exist_ok=True)

    input("Press Enter when ready to record...")

    print("Recording started...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        device=device
    )

    sd.wait()

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = f"{folder}/{label}_{filename}.wav"

    sf.write(filepath, audio, sample_rate)

    print("Recording saved:", filepath)
