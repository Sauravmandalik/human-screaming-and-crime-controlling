import tkinter as tk
from tkinter import ttk
import threading
import queue
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
import joblib
from python_speech_features import mfcc


# ---------------- SETTINGS ----------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "scream_detector.pkl"

SAMPLE_RATE = 22050
WINDOW_SECONDS = 2
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS

# 1024 samples = about 46 ms at 22050 Hz
CHUNK_SIZE = 1024

# Keep the visualization lightweight.
VISUAL_SECONDS = 0.75
VISUAL_MAX_SAMPLES = int(SAMPLE_RATE * VISUAL_SECONDS)
GUI_REFRESH_MS = 100

BG = "#0b1220"
PANEL = "#142238"
PANEL_DARK = "#0f1b2d"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
BLUE = "#38bdf8"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"


class ScreamDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Scream Detection System")
        self.root.geometry("980x760")
        self.root.minsize(820, 650)
        self.root.configure(bg=BG)

        self.model = None
        self.running = False
        self.closing = False
        self.stop_event = threading.Event()

        # Communication between background threads and the GUI.
        self.events = queue.Queue()
        self.prediction_queue = queue.Queue(maxsize=1)

        # Shared audio used only by the visualizer.
        self.audio_lock = threading.Lock()
        self.visual_audio = deque(maxlen=VISUAL_MAX_SAMPLES)

        # The existing model uses a 2-second feature window.
        self.prediction_history = deque(maxlen=3)

        self.capture_thread = None
        self.prediction_thread = None

        self.create_interface()
        self.load_model()

        self.root.after(100, self.process_events)
        self.root.after(GUI_REFRESH_MS, self.refresh_visuals)
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    # ---------------- GUI ----------------

    def create_interface(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=24, pady=(16, 8))

        tk.Label(
            header,
            text="HUMAN SCREAM DETECTION",
            font=("Segoe UI", 21, "bold"),
            fg=TEXT,
            bg=BG
        ).pack(anchor="w")

        tk.Label(
            header,
            text="REAL-TIME AUDIO MONITORING SYSTEM",
            font=("Segoe UI", 9, "bold"),
            fg=MUTED,
            bg=BG
        ).pack(anchor="w", pady=(3, 0))

        status_panel = tk.Frame(
            self.root, bg=PANEL, padx=16, pady=12
        )
        status_panel.pack(fill="x", padx=24, pady=6)

        self.status_label = tk.Label(
            status_panel,
            text="SYSTEM READY",
            font=("Segoe UI", 19, "bold"),
            fg=GREEN,
            bg=PANEL
        )
        self.status_label.pack(anchor="w")

        self.score_label = tk.Label(
            status_panel,
            text="Decision score: --",
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=PANEL
        )
        self.score_label.pack(anchor="w", pady=(3, 0))

        self.connection_label = tk.Label(
            status_panel,
            text="Microphone: Not running",
            font=("Segoe UI", 10),
            fg=MUTED,
            bg=PANEL
        )
        self.connection_label.pack(anchor="w", pady=(2, 0))

        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=24, pady=7)

        self.start_button = tk.Button(
            controls,
            text="▶  START DETECTION",
            command=self.start_detection,
            font=("Segoe UI", 10, "bold"),
            bg="#15803d",
            fg="white",
            activebackground=GREEN,
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=10,
            cursor="hand2"
        )
        self.start_button.pack(side="left", padx=(0, 10))

        self.stop_button = tk.Button(
            controls,
            text="■  STOP DETECTION",
            command=self.stop_detection,
            font=("Segoe UI", 10, "bold"),
            bg="#991b1b",
            fg="white",
            activebackground=RED,
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=10,
            cursor="hand2",
            state="disabled"
        )
        self.stop_button.pack(side="left")

        # Audio level
        self.section_title("MICROPHONE INPUT LEVEL")
        level_panel = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        level_panel.pack(fill="x", padx=24, pady=(3, 7))

        self.level_canvas = tk.Canvas(
            level_panel, height=16, bg=PANEL_DARK,
            highlightthickness=0
        )
        self.level_canvas.pack(fill="x")

        self.level_label = tk.Label(
            level_panel,
            text="Level: 0%   |   RMS: 0.0000",
            font=("Segoe UI", 9),
            fg=MUTED,
            bg=PANEL
        )
        self.level_label.pack(anchor="w", pady=(4, 0))

        # Waveform
        self.section_title("LIVE AUDIO WAVEFORM")
        self.wave_canvas = tk.Canvas(
            self.root, height=105, bg=PANEL,
            highlightthickness=0
        )
        self.wave_canvas.pack(fill="x", padx=24, pady=(3, 7))

        # Spectrum
        self.section_title("FREQUENCY SPECTRUM  |  0–5000 Hz")
        self.spectrum_canvas = tk.Canvas(
            self.root, height=125, bg=PANEL,
            highlightthickness=0
        )
        self.spectrum_canvas.pack(fill="x", padx=24, pady=(3, 7))

        # History
        self.section_title("RECENT DETECTION HISTORY")
        history_panel = tk.Frame(self.root, bg=PANEL, padx=8, pady=6)
        history_panel.pack(fill="both", expand=True, padx=24, pady=(3, 7))

        self.history_list = tk.Listbox(
            history_panel,
            height=4,
            bg=PANEL_DARK,
            fg=TEXT,
            font=("Consolas", 9),
            selectbackground="#263b57",
            relief="flat",
            highlightthickness=0,
            borderwidth=0
        )
        self.history_list.pack(fill="both", expand=True)

        tk.Label(
            self.root,
            text="EXPERIMENTAL PROTOTYPE  •  NOT FOR EMERGENCY SAFETY USE",
            font=("Segoe UI", 8),
            fg="#718096",
            bg=BG
        ).pack(pady=(2, 8))

        self.draw_idle_visuals()

    def section_title(self, title):
        tk.Label(
            self.root,
            text=title,
            font=("Segoe UI", 9, "bold"),
            fg=TEXT,
            bg=BG
        ).pack(anchor="w", padx=26, pady=(5, 0))

    def draw_idle_visuals(self):
        self.wave_canvas.create_text(
            400, 52,
            text="Start detection to view live audio",
            fill=MUTED,
            font=("Segoe UI", 10)
        )
        self.spectrum_canvas.create_text(
            400, 52,
            text="Frequency data will appear here",
            fill=MUTED,
            font=("Segoe UI", 10)
        )

    # ---------------- MODEL ----------------

    def load_model(self):
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model file not found:\n{MODEL_PATH}"
                )

            self.model = joblib.load(MODEL_PATH)
            self.status_label.config(
                text="SYSTEM READY",
                fg=GREEN
            )

        except Exception as error:
            self.model = None
            self.status_label.config(text="MODEL ERROR", fg=RED)
            self.connection_label.config(text=str(error)[:120])
            self.start_button.config(state="disabled")

    # ---------------- START / STOP ----------------

    def start_detection(self):
        if self.model is None or self.running:
            return

        self.stop_event.clear()
        self.prediction_history.clear()
        self.history_list.delete(0, tk.END)

        while True:
            try:
                self.prediction_queue.get_nowait()
            except queue.Empty:
                break

        with self.audio_lock:
            self.visual_audio.clear()

        self.running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="LISTENING...", fg=BLUE)
        self.score_label.config(text="Decision score: waiting for first prediction")
        self.connection_label.config(text="Microphone: connecting...")

        self.prediction_thread = threading.Thread(
            target=self.prediction_loop,
            daemon=True
        )
        self.capture_thread = threading.Thread(
            target=self.capture_loop,
            daemon=True
        )

        self.prediction_thread.start()
        self.capture_thread.start()

    def stop_detection(self):
        if not self.running:
            return

        self.stop_event.set()
        self.stop_button.config(state="disabled")
        self.status_label.config(text="STOPPING...", fg=AMBER)
        self.connection_label.config(text="Microphone: stopping...")

    # ---------------- AUDIO CAPTURE ----------------

    def capture_loop(self):
        audio_window = np.zeros(WINDOW_SIZE, dtype=np.float32)
        window_position = 0

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE
            ) as stream:

                self.events.put(("connected",))

                while not self.stop_event.is_set():
                    chunk_data, overflowed = stream.read(CHUNK_SIZE)

                    if self.stop_event.is_set():
                        break

                    chunk = chunk_data[:, 0].copy()

                    # Update the short rolling buffer used by the display.
                    with self.audio_lock:
                        self.visual_audio.extend(chunk.tolist())

                    # Add captured audio to the model's 2-second window.
                    offset = 0
                    while offset < len(chunk):
                        amount = min(
                            WINDOW_SIZE - window_position,
                            len(chunk) - offset
                        )

                        audio_window[
                            window_position:window_position + amount
                        ] = chunk[offset:offset + amount]

                        window_position += amount
                        offset += amount

                        if window_position >= WINDOW_SIZE:
                            self.enqueue_prediction(audio_window.copy())
                            window_position = 0

        except Exception as error:
            if not self.stop_event.is_set():
                self.events.put(("error", str(error)))
                self.stop_event.set()

        finally:
            self.events.put(("capture_stopped",))

    def enqueue_prediction(self, audio):
        # Keep only the newest pending window if inference is slower
        # than audio capture. This avoids an ever-growing backlog.
        try:
            self.prediction_queue.put_nowait(audio)
        except queue.Full:
            try:
                self.prediction_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.prediction_queue.put_nowait(audio)
            except queue.Full:
                pass

    # ---------------- MODEL INFERENCE ----------------

    def prediction_loop(self):
        while not self.stop_event.is_set():
            try:
                audio = self.prediction_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.predict_audio(audio)
            except Exception as error:
                self.events.put(("error", str(error)))
                self.stop_event.set()
                break

    def predict_audio(self, audio):
        mfcc_values = mfcc(
            audio.astype(np.float64),
            samplerate=SAMPLE_RATE,
            numcep=20,
            nfft=1024
        )

        features = np.concatenate([
            np.mean(mfcc_values, axis=0),
            np.std(mfcc_values, axis=0)
        ]).reshape(1, -1)

        prediction = int(self.model.predict(features)[0])
        score = float(self.model.decision_function(features)[0])

        self.prediction_history.append(prediction == 1)

        if len(self.prediction_history) < 3:
            final_result = "ANALYSING"
        elif sum(self.prediction_history) >= 2:
            final_result = "SCREAM DETECTED"
        else:
            final_result = "NO SCREAM"

        self.events.put((
            "result",
            final_result,
            score,
            "SCREAM" if prediction == 1 else "NON-SCREAM"
        ))

    # ---------------- VISUALIZATION ----------------

    def refresh_visuals(self):
        if self.closing:
            return

        if self.running:
            with self.audio_lock:
                audio = np.asarray(
                    self.visual_audio,
                    dtype=np.float32
                )

            if len(audio) > 0:
                self.draw_waveform(audio)
                self.draw_spectrum(audio)
                self.draw_level(audio)

        self.root.after(GUI_REFRESH_MS, self.refresh_visuals)

    def draw_waveform(self, audio):
        canvas = self.wave_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 80)
        middle = height / 2

        canvas.create_line(
            0, middle, width, middle,
            fill="#30445e"
        )

        count = min(300, len(audio))
        if count < 2:
            return

        indices = np.linspace(
            0, len(audio) - 1, count, dtype=int
        )
        samples = audio[indices]

        # Automatic vertical scaling, with a sensible minimum range.
        amplitude = max(float(np.max(np.abs(samples))), 0.025)
        scale = (height * 0.40) / amplitude

        coords = []
        for index, sample in enumerate(samples):
            x = index * (width - 2) / (count - 1) + 1
            y = middle - float(sample) * scale
            coords.extend((x, y))

        canvas.create_line(
            *coords,
            fill=BLUE,
            width=2,
            smooth=False
        )

    def draw_spectrum(self, audio):
        canvas = self.spectrum_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 100)

        # Zero-pad short visual buffers so the spectrum can still render.
        fft_size = 4096
        samples = np.zeros(fft_size, dtype=np.float32)
        take = min(len(audio), fft_size)
        if take:
            samples[-take:] = audio[-take:]

        samples *= np.hanning(fft_size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(samples))
        frequencies = np.fft.rfftfreq(
            fft_size, d=1.0 / SAMPLE_RATE
        )

        spectrum = spectrum[frequencies <= 5000]
        if len(spectrum) == 0:
            return

        bar_count = 48
        groups = np.array_split(spectrum, bar_count)
        values = np.array([
            np.max(group) if len(group) else 0.0
            for group in groups
        ])

        peak = float(np.max(values))
        if peak > 0:
            values = values / peak

        chart_bottom = height - 22
        chart_top = 8
        max_bar_height = chart_bottom - chart_top
        gap = 3
        bar_width = (width - 24) / bar_count

        for index, value in enumerate(values):
            bar_height = float(value) * max_bar_height
            x1 = 12 + index * bar_width
            x2 = x1 + max(1, bar_width - gap)
            y1 = chart_bottom - bar_height

            canvas.create_rectangle(
                x1, y1, x2, chart_bottom,
                fill=GREEN,
                outline=""
            )

        for hz in range(0, 5001, 1000):
            x = 12 + (hz / 5000) * (width - 24)
            canvas.create_text(
                x, height - 10,
                text=str(hz),
                fill=MUTED,
                font=("Segoe UI", 8)
            )

    def draw_level(self, audio):
        rms = float(np.sqrt(np.mean(audio * audio)) + 1e-10)
        dbfs = 20 * np.log10(rms)

        percent = int(np.clip((dbfs + 60) / 60 * 100, 0, 100))

        canvas = self.level_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 16)
        filled = width * percent / 100

        canvas.create_rectangle(
            0, 0, width, height,
            fill=PANEL_DARK,
            outline=""
        )
        canvas.create_rectangle(
            0, 0, filled, height,
            fill=GREEN,
            outline=""
        )

        self.level_label.config(
            text=f"Level: {percent}%   |   RMS: {rms:.4f}   |   {dbfs:.1f} dBFS"
        )

    # ---------------- GUI EVENTS ----------------

    def add_history(self, prediction, result):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_list.insert(
            0, f"{timestamp} | {prediction} | {result}"
        )

        while self.history_list.size() > 8:
            self.history_list.delete(tk.END)

    def process_events(self):
        if self.closing:
            return

        try:
            while True:
                event = self.events.get_nowait()

                if event[0] == "connected":
                    self.connection_label.config(
                        text="Microphone: Running"
                    )

                elif event[0] == "result":
                    result, score, prediction = event[1:]

                    color = (
                        RED if result == "SCREAM DETECTED"
                        else GREEN if result == "NO SCREAM"
                        else BLUE
                    )

                    self.status_label.config(
                        text=result,
                        fg=color
                    )
                    self.score_label.config(
                        text=f"Decision score: {score:.3f}"
                    )
                    self.add_history(prediction, result)

                elif event[0] == "error":
                    self.status_label.config(
                        text="DETECTION ERROR",
                        fg=RED
                    )
                    self.connection_label.config(
                        text=str(event[1])[:120]
                    )

                elif event[0] == "capture_stopped":
                    self.running = False
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.connection_label.config(
                        text="Microphone: Not running"
                    )

                    if self.status_label.cget("text") == "STOPPING...":
                        self.status_label.config(
                            text="STOPPED",
                            fg=MUTED
                        )

        except queue.Empty:
            pass

        self.root.after(100, self.process_events)

    # ---------------- CLOSE ----------------

    def close_app(self):
        self.closing = True
        self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ScreamDetectionApp(root)
    root.mainloop()
