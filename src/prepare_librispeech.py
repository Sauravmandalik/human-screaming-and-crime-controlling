
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

SOURCE_DIR = Path("external_datasets/LibriSpeech/dev-clean")
OUTPUT_DIR = Path("dataset/normal_speech")

TARGET_RATE = 22050
CLIP_SECONDS = 2
CLIP_LENGTH = TARGET_RATE * CLIP_SECONDS
CLIPS_PER_SPEAKER = 3

rng = random.Random(42)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_saved = 0
total_skipped = 0
speakers_used = 0

speaker_dirs = sorted(
    folder for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
)

print("Preparing LibriSpeech normal speech clips...")

for speaker_dir in speaker_dirs:

    chapter_dirs = sorted(
        folder for folder in speaker_dir.iterdir()
        if folder.is_dir()
    )

    if not chapter_dirs:
        continue

    selected_chapters = rng.sample(
        chapter_dirs,
        min(CLIPS_PER_SPEAKER, len(chapter_dirs))
    )

    speaker_saved = 0

    for chapter_dir in selected_chapters:

        recordings = sorted(chapter_dir.glob("*.flac"))

        if not recordings:
            total_skipped += 1
            continue

        source_file = rng.choice(recordings)

        output_name = (
            f"librispeech_{speaker_dir.name}_"
            f"{chapter_dir.name}_{source_file.stem}.wav"
        )

        output_path = OUTPUT_DIR / output_name

        if output_path.exists():
            print("Already exists:", output_name)
            continue

        try:
            audio, sample_rate = sf.read(
                source_file,
                dtype="float32"
            )

            # Convert stereo audio to mono.
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)

            # Convert audio to 22050 Hz.
            if sample_rate != TARGET_RATE:
                common = gcd(sample_rate, TARGET_RATE)

                audio = resample_poly(
                    audio,
                    TARGET_RATE // common,
                    sample_rate // common
                )

            # Skip recordings shorter than 2 seconds.
            if len(audio) < CLIP_LENGTH:
                total_skipped += 1
                continue

            # Select a random 2-second section.
            max_start = len(audio) - CLIP_LENGTH
            start = rng.randint(0, max_start)

            clip = audio[start:start + CLIP_LENGTH]

            sf.write(
                output_path,
                clip,
                TARGET_RATE
            )

            total_saved += 1
            speaker_saved += 1

        except Exception as error:
            total_skipped += 1
            print("Error:", source_file.name, error)

    if speaker_saved > 0:
        speakers_used += 1

    print(
        f"Speaker {speaker_dir.name}: "
        f"{speaker_saved} clips saved"
    )

print("\nPreparation completed!")
print("Speakers represented:", speakers_used)
print("New clips saved:", total_saved)
print("Files skipped:", total_skipped)
print("Output folder:", OUTPUT_DIR)
