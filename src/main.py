
import tkinter as tk
from tkinter import ttk
import threading
import queue
from pathlib import Path
from collections import deque

import numpy as np
import sounddevice as sd
import joblib
from python_speech_features import mfcc


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "scream_detector.pkl"

# Audio settings
SAMPLE_RATE = 22050
WINDOW_SECONDS = 2
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS


class ScreamDetectionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Human Scream Detection System")
        self.root.geometry("650x470")
        self.root.configure(bg="#101827")
        self.root.resizable(False, False)

        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.running = False
        self.history = deque(maxlen=3)
        self.model = None

        self.create_interface()
        self.load_model()

        self.root.after(100, self.process_events)
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def create_interface(self):

        tk.Label(
            self.root,
            text="HUMAN SCREAM DETECTION",
            font=("Arial", 21, "bold"),
            fg="white",
            bg="#101827"
        ).pack(pady=(25, 5))

        tk.Label(
            self.root,
            text="Real-Time Audio Monitoring System",
            font=("Arial", 11),
            fg="#94a3b8",
            bg="#101827"
        ).pack(pady=5)

        self.status_label = tk.Label(
            self.root,
            text="SYSTEM READY",
            font=("Arial", 22, "bold"),
            fg="#38bdf8",
            bg="#101827"
        )
        self.status_label.pack(pady=35)

        self.score_label = tk.Label(
            self.root,
            text="Decision Score: --",
            font=("Arial", 13),
            fg="white",
            bg="#101827"
        )
        self.score_label.pack(pady=10)

        self.connection_label = tk.Label(
            self.root,
            text="Microphone: Not Running",
            font=("Arial", 11),
            fg="#94a3b8",
            bg="#101827"
        )
        self.connection_label.pack(pady=10)

        button_frame = tk.Frame(
            self.root,
            bg="#101827"
        )
        button_frame.pack(pady=30)

        self.start_button = tk.Button(
            button_frame,
            text="START DETECTION",
            font=("Arial", 12, "bold"),
            bg="#16a34a",
            fg="white",
            width=18,
            height=2,
            command=self.start_detection
        )
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = tk.Button(
            button_frame,
            text="STOP DETECTION",
            font=("Arial", 12, "bold"),
            bg="#dc2626",
            fg="white",
            width=18,
            height=2,
            state="disabled",
            command=self.stop_detection
        )
        self.stop_button.grid(row=0, column=1, padx=10)

        tk.Label(
            self.root,
            text="Experimental prototype | Not for emergency safety use",
            font=("Arial", 9),
            fg="#64748b",
            bg="#101827"
        ).pack(side="bottom", pady=15)

    def load_model(self):

        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found: {MODEL_PATH}"
                )

            self.model = joblib.load(MODEL_PATH)

            self.status_label.config(
                text="SYSTEM READY",
                fg="#22c55e"
            )

        except Exception as error:
            self.status_label.config(
                text="MODEL ERROR",
                fg="#ef4444"
            )
            self.start_button.config(state="disabled")
            self.connection_label.config(
                text=str(error)
            )

    def start_detection(self):

        if self.model is None or self.running:
            return

        self.stop_event.clear()
        self.history.clear()
        self.running = True

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        self.status_label.config(
            text="ANALYSING...",
            fg="#38bdf8"
        )

        self.connection_label.config(
            text="Microphone: Starting..."
        )

        self.worker = threading.Thread(
            target=self.detection_loop,
            daemon=True
        )
        self.worker.start()

    def detection_loop(self):

        try:
            while not self.stop_event.is_set():

                audio = sd.rec(
                    WINDOW_SIZE,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32"
                )

                sd.wait()

                if self.stop_event.is_set():
                    break

                audio = audio.flatten()

                features_mfcc = mfcc(
                    audio.astype(np.float64),
                    samplerate=SAMPLE_RATE,
                    numcep=20,
                    nfft=1024
                )

                features = np.concatenate([
                    np.mean(features_mfcc, axis=0),
                    np.std(features_mfcc, axis=0)
                ]).reshape(1, -1)

                prediction = self.model.predict(features)[0]
                score = self.model.decision_function(features)[0]

                self.history.append(prediction == 1)

                if len(self.history) < 3:
                    final_result = "ANALYSING"
                elif sum(self.history) >= 2:
                    final_result = "SCREAM DETECTED"
                else:
                    final_result = "NO SCREAM"

                self.events.put((
                    "result",
                    final_result,
                    float(score)
                ))

        except Exception as error:
            self.events.put(("error", str(error)))

        finally:
            self.events.put(("stopped",))

    def stop_detection(self):

        self.stop_event.set()

        try:
            sd.stop()
        except Exception:
            pass

        self.stop_button.config(state="disabled")
        self.status_label.config(
            text="STOPPING...",
            fg="#f59e0b"
        )

    def process_events(self):

        try:
            while True:
                event = self.events.get_nowait()

                if event[0] == "result":

                    result = event[1]
                    score = event[2]

                    self.status_label.config(
                        text=result,
                        fg=(
                            "#ef4444"
                            if result == "SCREAM DETECTED"
                            else "#22c55e"
                            if result == "NO SCREAM"
                            else "#38bdf8"
                        )
                    )

                    self.score_label.config(
                        text=f"Decision Score: {score:.3f}"
                    )

                    self.connection_label.config(
                        text="Microphone: Running"
                    )

                elif event[0] == "error":

                    self.status_label.config(
                        text="DETECTION ERROR",
                        fg="#ef4444"
                    )

                    self.connection_label.config(
                        text=event[1][:75]
                    )

                elif event[0] == "stopped":

                    self.running = False

                    self.start_button.config(
                        state="normal"
                    )

                    self.stop_button.config(
                        state="disabled"
                    )

                    self.connection_label.config(
                        text="Microphone: Not Running"
                    )

                    if self.status_label.cget("text") == "STOPPING...":
                        self.status_label.config(
                            text="STOPPED",
                            fg="#94a3b8"
                        )

        except queue.Empty:
            pass

        if self.root.winfo_exists():
            self.root.after(100, self.process_events)

    def close_app(self):

        self.stop_event.set()

        try:
            sd.stop()
        except Exception:
            pass

        self.root.destroy()


if __name__ == "__main__":

    print("Starting Scream Detection GUI...")

    root = tk.Tk()
    print("Window created.")

    app = ScreamDetectionApp(root)
    print("Application initialized.")

    root.mainloop()