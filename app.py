from __future__ import annotations

import os
import sys
import queue
import subprocess
import threading
import time
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from clipper_engine import (
    APP_VERSION, DEFAULT_OLLAMA_MODEL, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR,
    JobCancelled, get_ollama_status, process_job, format_eta
)
from paths import LOG_DIR

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ClipperApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"ClipForge · Local Gaming Clipper v{APP_VERSION}")
        self.geometry("1060x820")
        self.minsize(900, 720)
        self.configure(fg_color="#11131b")

        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.running = False
        self.start_time: float | None = None
        self.last_overall = 0.0
        self.current_output: Path | None = None
        self.log_file = LOG_DIR / "app.log"
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self.after(150, self._poll_events)
        self.after(500, self._refresh_status)
        self.after(1000, self._eta_tick)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="#191c28", corner_radius=16)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 10))
        header.grid_columnconfigure(1, weight=1)
        logo = ctk.CTkLabel(header, text="🎬", font=ctk.CTkFont(size=30))
        logo.grid(row=0, column=0, rowspan=2, padx=(18, 12), pady=13)
        ctk.CTkLabel(header, text="CLIPFORGE", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#f4f5fb").grid(row=0, column=1, sticky="sw", pady=(12, 0))
        ctk.CTkLabel(header, text=f"LOCAL GAMING CLIPPING STUDIO  ·  WINDOWS / macOS / LINUX  ·  NO SUBSCRIPTIONS  ·  v{APP_VERSION}",
                     font=ctk.CTkFont(size=10, weight="bold"), text_color="#9299ad").grid(
                         row=1, column=1, sticky="nw", pady=(1, 12))
        self.ollama_badge = ctk.CTkLabel(header, text="Checking local AI…", corner_radius=9,
                                         fg_color="#34384a", text_color="#e0e4f2",
                                         padx=12, pady=7)
        self.ollama_badge.grid(row=0, column=2, rowspan=2, padx=18)

        body = ctk.CTkScrollableFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        body.grid_columnconfigure(0, weight=1)

        # Input panel
        input_card = self._card(body, "1  ·  SOURCE VIDEO")
        input_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        input_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(input_card, text="YouTube URL (video you own or are authorised to use)",
                     text_color="#aeb5c8").grid(row=1, column=0, sticky="w", padx=16, pady=(8, 5))
        self.url_var = ctk.StringVar()
        url_row = ctk.CTkFrame(input_card, fg_color="transparent")
        url_row.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        url_row.grid_columnconfigure(0, weight=1)
        self.url_entry = ctk.CTkEntry(url_row, textvariable=self.url_var, height=40,
                                      placeholder_text="https://www.youtube.com/watch?v=…")
        self.url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.paste_url_button = ctk.CTkButton(url_row, text="Paste URL", width=105,
                                               height=40, command=self._paste_url)
        self.paste_url_button.grid(row=0, column=1)

        auth_row = ctk.CTkFrame(input_card, fg_color="#202536", corner_radius=10)
        auth_row.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 10))
        auth_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(auth_row, text="YOUTUBE LOGIN BROWSER", text_color="#c7cce0",
                     anchor="w", font=ctk.CTkFont(size=11, weight="bold")).grid(
                         row=0, column=0, sticky="w", padx=(12, 10), pady=(9, 2))
        self.cookies_browser_var = ctk.StringVar(value="Firefox")
        self.cookies_browser_menu = ctk.CTkComboBox(
            auth_row, variable=self.cookies_browser_var,
            values=["Chrome", "Edge", "Brave", "Firefox", "Vivaldi", "Opera", "None (anonymous)"],
            height=34, width=180
        )
        self.cookies_browser_menu.grid(row=0, column=1, sticky="w", pady=(7, 2))
        ctk.CTkLabel(auth_row,
                     text="Use the browser already signed into YouTube. Fully exit that browser before downloading.",
                     text_color="#aeb5c8", anchor="w", font=ctk.CTkFont(size=10)).grid(
                         row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 9))

        ctk.CTkLabel(input_card, text="OR choose a local video file", text_color="#aeb5c8").grid(
            row=4, column=0, sticky="w", padx=16, pady=(3, 5))
        file_row = ctk.CTkFrame(input_card, fg_color="transparent")
        file_row.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 14))
        file_row.grid_columnconfigure(0, weight=1)
        self.file_var = ctk.StringVar()
        self.file_entry = ctk.CTkEntry(file_row, textvariable=self.file_var, height=36,
                                       placeholder_text="No local file selected")
        self.file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.browse_button = ctk.CTkButton(file_row, text="Browse…", width=110,
                                            command=self._browse_video)
        self.browse_button.grid(row=0, column=1)
        self.url_entry.bind("<KeyRelease>", lambda _e: self._clear_local_file())

        # Settings
        settings = self._card(body, "2  ·  CLIP SETTINGS")
        settings.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for col in range(4):
            settings.grid_columnconfigure(col, weight=1)
        self.game_var = ctk.StringVar(value="Auto / mixed games")
        self.count_var = ctk.StringVar(value="10")
        self.length_var = ctk.StringVar(value="35")
        self.stride_var = ctk.StringVar(value="15")
        self.whisper_var = ctk.StringVar(value="small")
        self.ollama_var = ctk.StringVar(value=DEFAULT_OLLAMA_MODEL)
        self.device_var = ctk.StringVar(value="CPU (recommended for first run)")
        self.language_mode_var = ctk.StringVar(value="Mixed (English + Bengali + Hindi)")
        self.render_layout_var = ctk.StringVar(value="Fit + blurred background")
        self.ollama_enabled = ctk.BooleanVar(value=True)
        self.nvenc_enabled = ctk.BooleanVar(value=True)

        self._field(settings, "Game profile", self.game_var,
                    ["Auto / mixed games", "Minecraft", "GTA V", "FPS / Tactical", "Battle Royale", "Other"], 0)
        self._field(settings, "Number of clips", self.count_var, ["5", "10", "15", "20"], 1)
        self._field(settings, "Clip length (sec)", self.length_var, ["30", "35", "40", "45", "60"], 2)
        self._field(settings, "Scan interval (sec)", self.stride_var, ["10", "12", "15", "20"], 3)
        self._field(settings, "Whisper model", self.whisper_var, ["tiny", "base", "small", "medium"], 0, row=3)
        self._field(settings, "Transcription device", self.device_var,
                    ["CPU (recommended for first run)", "CUDA (advanced; requires compatible CUDA libraries)"], 1, row=3)
        self._field(settings, "Ollama model", self.ollama_var, None, 2, row=3)
        self.ollama_check = ctk.CTkCheckBox(settings, text="Use local Ollama visual AI ranking",
                                            variable=self.ollama_enabled, onvalue=True, offvalue=False)
        self.ollama_check.grid(row=4, column=0, columnspan=2, sticky="w", padx=16, pady=(10, 7))
        self.nvenc_check = ctk.CTkCheckBox(settings, text="Use NVIDIA NVENC when available",
                                           variable=self.nvenc_enabled, onvalue=True, offvalue=False)
        self.nvenc_check.grid(row=4, column=2, columnspan=2, sticky="w", padx=16, pady=(10, 7))
        self._field(settings, "Speech language mode", self.language_mode_var,
                    ["Mixed (English + Bengali + Hindi)", "English", "Bengali", "Hindi"], 0, row=5)
        self._field(settings, "Vertical layout", self.render_layout_var,
                    ["Fit + blurred background", "Fill (center crop)"], 1, row=5)
        ctk.CTkLabel(
            settings,
            text="Mixed mode detects language per speech segment. Visual AI checks sampled frames to demote menus and starting screens.",
            text_color="#aeb5c8", anchor="w", justify="left", wraplength=520
        ).grid(row=5, column=2, columnspan=2, sticky="w", padx=16, pady=(27, 8))
        self.burn_captions_var = ctk.BooleanVar(value=True)
        self.normalize_audio_var = ctk.BooleanVar(value=True)
        self.burn_captions_check = ctk.CTkCheckBox(
            settings, text="Burn in animated word-highlight captions when speech is usable",
            variable=self.burn_captions_var
        )
        self.burn_captions_check.grid(row=6, column=0, columnspan=2, sticky="w", padx=16, pady=(8, 10))
        self.normalize_audio_check = ctk.CTkCheckBox(
            settings, text="Normalize game/commentary audio for social video",
            variable=self.normalize_audio_var
        )
        self.normalize_audio_check.grid(row=6, column=2, columnspan=2, sticky="w", padx=16, pady=(8, 10))

        # Output
        output = self._card(body, "3  ·  OUTPUT LOCATION")
        output.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        output.grid_columnconfigure(0, weight=1)
        self.output_var = ctk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.output_entry = ctk.CTkEntry(output, textvariable=self.output_var, height=36)
        self.output_entry.grid(row=1, column=0, sticky="ew", padx=(16, 8), pady=(9, 14))
        self.output_browse = ctk.CTkButton(output, text="Choose folder…", width=130,
                                            command=self._browse_output)
        self.output_browse.grid(row=1, column=1, padx=(0, 16), pady=(9, 14))

        # Process and status
        process = self._card(body, "4  ·  PROCESSING")
        process.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        process.grid_columnconfigure(0, weight=1)
        top_status = ctk.CTkFrame(process, fg_color="transparent")
        top_status.grid(row=1, column=0, sticky="ew", padx=16, pady=(10, 3))
        top_status.grid_columnconfigure(0, weight=1)
        self.stage_label = ctk.CTkLabel(top_status, text="Ready when you are", anchor="w",
                                        font=ctk.CTkFont(size=13, weight="bold"))
        self.stage_label.grid(row=0, column=0, sticky="w")
        self.eta_label = ctk.CTkLabel(top_status, text="ETA: —", text_color="#aeb5c8")
        self.eta_label.grid(row=0, column=1, sticky="e")
        self.overall_bar = ctk.CTkProgressBar(process, height=12, progress_color="#7657e8")
        self.overall_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 4))
        self.overall_bar.set(0)
        progress_row = ctk.CTkFrame(process, fg_color="transparent")
        progress_row.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 5))
        progress_row.grid_columnconfigure(0, weight=1)
        self.message_label = ctk.CTkLabel(progress_row, text="Paste a YouTube URL or select a local video.",
                                          text_color="#aeb5c8", anchor="w")
        self.message_label.grid(row=0, column=0, sticky="w")
        self.percent_label = ctk.CTkLabel(progress_row, text="0%", text_color="#c9c3ff")
        self.percent_label.grid(row=0, column=1, sticky="e")
        self.stage_bar = ctk.CTkProgressBar(process, height=6, progress_color="#31b6a1")
        self.stage_bar.grid(row=4, column=0, sticky="ew", padx=16, pady=(3, 14))
        self.stage_bar.set(0)

        buttons = ctk.CTkFrame(process, fg_color="transparent")
        buttons.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 15))
        self.start_button = ctk.CTkButton(buttons, text="▶  START CLIPPING", height=42,
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          fg_color="#7154df", hover_color="#5e42c5",
                                          command=self._start_job)
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.cancel_button = ctk.CTkButton(buttons, text="Cancel job", width=110, height=42,
                                           fg_color="#343849", hover_color="#44495d",
                                           state="disabled", command=self._cancel_job)
        self.cancel_button.pack(side="left", padx=(0, 8))
        self.open_output_button = ctk.CTkButton(buttons, text="Open output", width=120, height=42,
                                                command=self._open_output)
        self.open_output_button.pack(side="left")

        # Logs panel
        logs = self._card(body, "LIVE LOG")
        logs.grid(row=4, column=0, sticky="ew", pady=(0, 16))
        logs.grid_columnconfigure(0, weight=1)
        self.log_box = ctk.CTkTextbox(logs, height=145, fg_color="#0c0e14",
                                      text_color="#c9cedd", font=ctk.CTkFont(family="Consolas", size=11))
        self.log_box.grid(row=1, column=0, sticky="ew", padx=14, pady=(8, 14))
        self.log_box.configure(state="disabled")
        self._append_log("ClipForge started. All AI analysis is configured to run locally.")

        footer = ctk.CTkLabel(self, text="Local-first processing · No paid API keys · Review clips before publishing",
                              text_color="#747d94", font=ctk.CTkFont(size=10))
        footer.grid(row=2, column=0, pady=(0, 10))

    def _card(self, parent, title):
        card = ctk.CTkFrame(parent, fg_color="#191c28", corner_radius=14)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#aeb5c8").grid(row=0, column=0, columnspan=4,
                                                   sticky="w", padx=16, pady=(13, 0))
        return card

    def _field(self, parent, label, variable, values, col, row=1):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=col, sticky="ew", padx=16, pady=(8, 2))
        ctk.CTkLabel(wrap, text=label, text_color="#aeb5c8", anchor="w").pack(fill="x", pady=(0, 5))
        if values:
            widget = ctk.CTkComboBox(wrap, variable=variable, values=values, height=34)
        else:
            widget = ctk.CTkEntry(wrap, textvariable=variable, height=34)
        widget.pack(fill="x")
        return widget

    def _paste_url(self):
        try:
            pasted = self.clipboard_get().strip()
        except Exception:
            messagebox.showwarning("Clipboard empty", "Copy the YouTube video URL first, then click Paste URL.")
            return
        if not pasted:
            messagebox.showwarning("Clipboard empty", "Copy the YouTube video URL first, then click Paste URL.")
            return
        # Replace the field instead of appending to existing text.
        self.url_var.set(pasted)
        self.file_var.set("")
        self.url_entry.icursor("end")

    @staticmethod
    def _clean_youtube_url(value: str) -> tuple[str, bool]:
        raw = value.strip().strip('"\'<>')
        # Recover from accidental repeated paste, e.g. URL + URL with no space.
        starts = [m.start() for m in __import__("re").finditer(r"https?://", raw, flags=__import__("re").I)]
        duplicated = len(starts) > 1
        if duplicated:
            raw = raw[:starts[1]].strip()
        return raw, duplicated

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Choose a gaming video",
            filetypes=[("Video files", "*.mp4 *.mkv *.webm *.mov *.avi"), ("All files", "*.*")]
        )
        if path:
            self.file_var.set(path)
            self.url_var.set("")

    def _clear_local_file(self):
        if self.url_var.get().strip():
            self.file_var.set("")

    def _browse_output(self):
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_var.set(path)

    def _append_log(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _refresh_status(self):
        online, message = get_ollama_status(self.ollama_var.get().strip() or DEFAULT_OLLAMA_MODEL)
        color = "#214d3d" if online and "available" in message else "#4a3d25" if online else "#4a2d35"
        self.ollama_badge.configure(text=message, fg_color=color)
        self.after(10000, self._refresh_status)

    def _start_job(self):
        url = self.url_var.get().strip()
        local = self.file_var.get().strip()
        duplicated_url = False
        if url:
            url, duplicated_url = self._clean_youtube_url(url)
            self.url_var.set(url)
            if duplicated_url:
                self._append_log("Detected multiple pasted URLs in the field; using the first URL only.")
            if not url.lower().startswith(("https://www.youtube.com/", "https://youtube.com/",
                                           "https://youtu.be/", "http://www.youtube.com/",
                                           "http://youtube.com/", "http://youtu.be/")):
                messagebox.showwarning("Invalid YouTube URL", "Paste a YouTube video URL (youtube.com or youtu.be).")
                return
        if not url and not local:
            messagebox.showwarning("Missing video", "Paste a YouTube URL or browse for a local video file.")
            return
        source = url or local
        try:
            count = int(self.count_var.get())
            length = int(self.length_var.get())
            stride = int(self.stride_var.get())
            if count < 1 or length < 10 or stride < 5:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid settings", "Use a positive clip count, clip length >=10 sec and scan interval >=5 sec.")
            return
        output_text = self.output_var.get().strip()
        if not output_text:
            messagebox.showwarning("Missing output folder", "Choose an output folder first.")
            return
        output_root = Path(output_text).expanduser()
        whisper_device = "cuda" if self.device_var.get().startswith("CUDA") else "cpu"
        if whisper_device == "cuda":
            if not messagebox.askyesno("Advanced GPU transcription", "CUDA transcription requires compatible CUDA/cuDNN libraries for CTranslate2. Continue? If it fails, switch back to CPU."):
                return
        args = {
            "source": source,
            "output_root": output_root,
            "count": count,
            "clip_length": length,
            "stride": stride,
            "whisper_model": self.whisper_var.get(),
            "whisper_device": whisper_device,
            "ollama_model": self.ollama_var.get().strip() or DEFAULT_OLLAMA_MODEL,
            "use_ollama": bool(self.ollama_enabled.get()),
            "use_nvenc": bool(self.nvenc_enabled.get()),
            "game_profile": self.game_var.get(),
            "language_mode": self.language_mode_var.get(),
            "render_layout": self.render_layout_var.get(),
            "burn_captions": bool(self.burn_captions_var.get()),
            "normalize_audio": bool(self.normalize_audio_var.get()),
            "cookies_browser": (None if self.cookies_browser_var.get().startswith("None")
                                else self.cookies_browser_var.get().strip().lower()),
        }
        self.cancel_event.clear()
        self.running = True
        self.start_time = time.monotonic()
        self.last_overall = 0
        self.overall_bar.set(0)
        self.stage_bar.set(0)
        self.percent_label.configure(text="0%")
        self.stage_label.configure(text="Starting job…")
        self.message_label.configure(text="Preparing local processing pipeline…")
        self.eta_label.configure(text="ETA: calculating…")
        self.start_button.configure(state="disabled", text="PROCESSING…")
        self.cancel_button.configure(state="normal")
        for widget in (self.url_entry, self.paste_url_button, self.file_entry, self.browse_button,
                       self.cookies_browser_menu, self.output_entry, self.output_browse):
            widget.configure(state="disabled")
        self._append_log(f"Starting job: {source}")
        self.worker = threading.Thread(target=self._worker_run, args=(args,), daemon=False)
        self.worker.start()

    def _worker_run(self, args):
        def report(stage, stage_pct, overall_pct, message):
            self.events.put(("progress", stage, stage_pct, overall_pct, message))

        def log(message):
            self.events.put(("log", str(message)))

        try:
            result = process_job(
                args["source"], args["output_root"], DEFAULT_INPUT_DIR,
                count=args["count"], clip_length=args["clip_length"], stride=args["stride"],
                whisper_model=args["whisper_model"], whisper_device=args["whisper_device"],
                ollama_model=args["ollama_model"], use_ollama=args["use_ollama"],
                use_nvenc=args["use_nvenc"], game_profile=args["game_profile"],
                cookies_browser=args["cookies_browser"],
                language_mode=args["language_mode"], render_layout=args["render_layout"],
                burn_captions=args["burn_captions"], normalize_audio=args["normalize_audio"],
                progress=report, log=log, cancel_event=self.cancel_event,
            )
            self.events.put(("done", str(result)))
        except JobCancelled as exc:
            self.events.put(("cancelled", str(exc)))
        except Exception as exc:
            details = traceback.format_exc()
            self.events.put(("error", str(exc), details))

    def _cancel_job(self):
        if self.running:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled", text="Stopping…")
            self._append_log("Cancellation requested. The current operation will stop at its next safe checkpoint.")

    def _poll_events(self):
        try:
            while True:
                item = self.events.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log(item[1])
                elif kind == "progress":
                    _, stage, stage_pct, overall_pct, message = item
                    self.stage_label.configure(text=stage)
                    self.stage_bar.set(stage_pct / 100)
                    self.overall_bar.set(overall_pct / 100)
                    self.last_overall = max(self.last_overall, overall_pct)
                    self.percent_label.configure(text=f"{overall_pct:.0f}%")
                    self.message_label.configure(text=message)
                    self._update_eta(overall_pct)
                elif kind == "done":
                    self.current_output = Path(item[1])
                    self._append_log(f"SUCCESS · Output: {item[1]}")
                    self.stage_label.configure(text="Completed successfully")
                    self.message_label.configure(text="Your clips and subtitle files are ready.")
                    self.eta_label.configure(text="Finished")
                    self.overall_bar.set(1)
                    self.stage_bar.set(1)
                    self.percent_label.configure(text="100%")
                    self._finish_job()
                    messagebox.showinfo("Clipping complete", f"Your clips were saved here:\n\n{item[1]}")
                elif kind == "cancelled":
                    self._append_log(item[1])
                    self.stage_label.configure(text="Cancelled")
                    self.message_label.configure(text="The job was cancelled at a safe checkpoint.")
                    self.eta_label.configure(text="—")
                    self._finish_job()
                elif kind == "error":
                    self._append_log("ERROR · " + item[1])
                    self._append_log(item[2])
                    self.stage_label.configure(text="Job failed")
                    self.message_label.configure(text=item[1])
                    self.eta_label.configure(text="—")
                    self._finish_job()
                    messagebox.showerror("Clipping error", item[1] + "\n\nFull details are in logs\\app.log and the Live Log panel.")
        except queue.Empty:
            pass
        self.after(150, self._poll_events)

    def _update_eta(self, pct: float):
        if not self.start_time or pct < 2 or pct >= 100:
            return
        elapsed = time.monotonic() - self.start_time
        remaining = elapsed * (100 - pct) / pct
        self.eta_label.configure(text=f"ETA: ~{format_eta(remaining)}")

    def _eta_tick(self):
        if self.running and self.last_overall >= 2:
            self._update_eta(self.last_overall)
        self.after(1000, self._eta_tick)

    def _finish_job(self):
        self.running = False
        self.start_button.configure(state="normal", text="▶  START CLIPPING")
        self.cancel_button.configure(state="disabled", text="Cancel job")
        for widget in (self.url_entry, self.paste_url_button, self.file_entry, self.browse_button,
                       self.cookies_browser_menu, self.output_entry, self.output_browse):
            widget.configure(state="normal")

    def _open_output(self):
        path = self.current_output if self.current_output and self.current_output.exists() else Path(self.output_var.get())
        path.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)], start_new_session=True)
            else:
                subprocess.Popen(["xdg-open", str(path)], start_new_session=True)
        except Exception as exc:
            messagebox.showerror("Could not open folder", str(exc))

    def _on_close(self):
        if self.running:
            messagebox.showwarning(
                "Clipping job is still running",
                "Keep ClipForge open until the job finishes. Use Cancel job and wait for the safe-stop message if you need to stop it."
            )
            return
        self.destroy()


def main():
    app = ClipperApp()
    app.mainloop()


if __name__ == "__main__":
    main()
