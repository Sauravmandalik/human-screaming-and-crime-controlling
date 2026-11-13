\# Human Scream Detection System



A Python-based machine learning project that analyzes audio recordings to identify possible human screams. The system uses MFCC (Mel-Frequency Cepstral Coefficients) features and a Support Vector Machine (SVM) classifier to distinguish scream audio from non-scream audio.



The project also includes real-time microphone-based detection.



> \*\*Project status:\*\* Research prototype. This system is not designed or validated for emergency or safety-critical use.



\## Features



\- Audio preprocessing and feature extraction.

\- Stereo-to-mono audio conversion.

\- Audio resampling to 22,050 Hz.

\- MFCC-based feature extraction.

\- Mean and standard deviation feature calculation.

\- SVM-based binary classification.

\- Group-aware train/test splitting for LibriSpeech speakers.

\- Model evaluation with accuracy, confusion matrix and classification report.

\- False-positive and false-negative inspection.

\- Real-time microphone-based audio detection.

\- Prediction smoothing using recent detection results.



\## Technologies Used



\- Python

\- NumPy

\- SciPy

\- SoundFile

\- python\_speech\_features

\- Scikit-learn

\- Joblib

\- SoundDevice



\## Project Structure



```text

hsac/

│

├── dataset/

│   ├── scream/

│   ├── non\_scream/

│   ├── normal\_speech/

│   ├── laughing/

│   └── shouting/

│

├── data/

├── models/

├── test\_audio/

│   ├── scream/

│   └── non\_scream/

│

├── src/

│   ├── extract\_feature\_v2.py

│   ├── train\_model.py

│   ├── evaluate\_model.py

│   ├── predict\_audio.py

│   ├── realtime\_detection.py

│   ├── record\_audio.py

│   ├── record\_dataset.py

│   ├── analyze\_false\_positives.py

│   ├── analyze\_frequency.py

│   ├── compare\_frequency.py

│   ├── diagnose\_audio.py

│   ├── extract\_feature\_v3.py

│   ├── inspect\_dataset.py

│   ├── inspect\_windows.py

│   └── prepare\_librispeech.py

│

├── requirements.txt

├── .gitignore

└── README.md

```



\## Installation



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd hsac

```



Replace `YOUR\_GITHUB\_REPOSITORY\_URL` with your actual GitHub repository URL.



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```bash

python -m pip install -r requirements.txt

```



\## Dataset Preparation



The feature extraction script expects audio files in these folders:



```text

dataset/

├── scream/

├── non\_scream/

├── normal\_speech/

├── laughing/

└── shouting/

```



Supported audio formats include WAV and FLAC.



Place your audio files in the appropriate folders before extracting features.



The dataset is not included in this repository. You must provide your own audio dataset.



\## Model Training



\### Step 1: Extract audio features



```bash

python src/extract\_feature\_v2.py

```



This script:

\- Reads audio files from the dataset folders.

\- Converts stereo audio to mono.

\- Resamples audio to 22,050 Hz.

\- Extracts 20 MFCC coefficients.

\- Calculates mean and standard deviation.

\- Saves the extracted features and labels.



Generated files:



```text

data/

├── features\_v2.npy

├── labels\_v2.npy

└── filenames\_v2.npy

```



\### Step 2: Train the model



```bash

python src/train\_model.py

```



The training script:

\- Loads the extracted features.

\- Uses a StandardScaler and SVM classifier.

\- Performs a five-fold group-aware split and evaluates the first fold.

\- Displays classification metrics and error examples.

\- Saves the trained model.



Generated model:



```text

models/scream\_detector.pkl

```



\## Model Evaluation



Prepare separate test audio files in:



```text

test\_audio/

├── scream/

└── non\_scream/

```



Then run:



```bash

python src/evaluate\_model.py

```



The evaluation script uses the trained model to evaluate test audio and generates a report. Check the script for its exact input and output requirements.



\## Real-Time Detection



After training the model, connect a working microphone and run:



```bash

python src/realtime\_detection.py

```



The system:

\- Captures audio from the microphone.

\- Processes audio in two-second windows.

\- Extracts MFCC features.

\- Predicts scream or non-scream.

\- Uses recent predictions to smooth the final detection result.



Press `Ctrl + C` to stop detection.



\## Limitations



\- The project is an experimental machine learning prototype.

\- Dataset labels may contain incorrect or ambiguous audio samples.

\- Loud sounds, crying, laughter, sirens and other noises may cause false detections.

\- Detection accuracy depends on dataset quality and recording conditions.

\- Real-time predictions are not guaranteed to identify every scream.

\- The current real-time detection script prints results; it does not send emergency notifications.

\- This system must not be relied upon as a substitute for emergency services or personal safety systems.



\## Future Improvements



\- Improve dataset quality and label verification.

\- Expand testing using unseen speakers and recording environments.

\- Reduce false-positive and false-negative predictions.

\- Explore additional audio features and classification models.

\- Develop an emergency notification feature.

\- Improve the real-time detection interface.



\## Author



\*\*Saurav Mandalik\*\*



Machine Learning and Python Project

