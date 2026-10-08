"""
P15 — Desktop Voice Interface for P-MAI.
Native Windows/Desktop GUI built with Tkinter, integrating with PMAIVoiceClient.
Provides real-time state visualization, streaming speech controls,
barge-in interruption, stop vs cancel separation, and conversation history.
"""
import asyncio
import json
import math
import struct
import threading
import time
import tkinter as tk
from tkinter import font as tkfont, messagebox
from typing import Optional

from client.voice.audio_device import AudioDevice
from client.voice.client import PMAIVoiceClient
from client.voice.config import VoiceClientConfig
from client.voice.state_machine import ClientUIState


class PMAIVoiceDesktopApp:
    """
    Tkinter desktop application for P-MAI voice interaction.
    """

    def __init__(self, root: tk.Tk, config: Optional[VoiceClientConfig] = None):
        self.root = root
        self.root.title("P-MAI Voice Assistant")
        self.root.geometry("540x680")
        self.root.minsize(480, 580)
        self.root.configure(bg="#0b0f19")

        self.config = config or VoiceClientConfig()
        self.client = PMAIVoiceClient(config=self.config)

        # Background asyncio thread & state
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.bg_thread: Optional[threading.Thread] = None
        self.is_running = True
        self.is_recording = False

        # Visualizer animation
        self.audio_level = 0.0

        # Register client callbacks
        self.client.on_state_change = self._on_client_state_change
        self.client.on_partial_transcript = self._on_client_partial
        self.client.on_final_transcript = self._on_client_final
        self.client.on_turn_completed = self._on_client_turn_completed
        self.client.on_error = self._on_client_error

        self._build_ui()
        self._start_background_loop()
        self._animate_visualizer()

    def _build_ui(self) -> None:
        title_font = tkfont.Font(family="Segoe UI", size=15, weight="bold")
        sub_font = tkfont.Font(family="Segoe UI", size=9)
        state_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        body_font = tkfont.Font(family="Segoe UI", size=10)
        btn_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")

        # 1. Header Frame
        header = tk.Frame(self.root, bg="#121826", pady=12, padx=16, highlightthickness=1, highlightbackground="#1f293d")
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(header, text="P-MAI Voice Assistant", fg="#f3f4f6", bg="#121826", font=title_font)
        title_lbl.pack(anchor=tk.W)

        sub_lbl = tk.Label(header, text="Autonomous Real-Time Conversational Agent", fg="#9ca3af", bg="#121826", font=sub_font)
        sub_lbl.pack(anchor=tk.W)

        # 2. State & Hero Frame
        hero = tk.Frame(self.root, bg="#0b0f19", pady=14, padx=16)
        hero.pack(fill=tk.X)

        self.state_lbl = tk.Label(hero, text="● READY", fg="#10b981", bg="#0b0f19", font=state_font)
        self.state_lbl.pack(pady=4)

        # Visualizer Canvas
        self.canvas = tk.Canvas(hero, height=44, bg="#111827", highlightthickness=1, highlightbackground="#1f2937")
        self.canvas.pack(fill=tk.X, pady=6)

        # Live Partial Transcript Label
        self.partial_lbl = tk.Label(
            hero,
            text="",
            fg="#a5f3fc",
            bg="#111827",
            font=body_font,
            wraplength=480,
            justify=tk.LEFT,
            pady=6,
            padx=10,
        )
        self.partial_lbl.pack(fill=tk.X, pady=4)

        # 3. Conversation History
        history_frame = tk.Frame(self.root, bg="#0b0f19", padx=16, pady=4)
        history_frame.pack(fill=tk.BOTH, expand=True)

        self.history_text = tk.Text(
            history_frame,
            bg="#111827",
            fg="#f3f4f6",
            font=body_font,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bd=0,
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground="#1f293d",
        )
        self.history_text.pack(fill=tk.BOTH, expand=True)

        self.history_text.tag_config("user", foreground="#93c5fd", font=tkfont.Font(family="Segoe UI", size=10, weight="bold"))
        self.history_text.tag_config("mai", foreground="#c084fc", font=tkfont.Font(family="Segoe UI", size=10, weight="bold"))
        self.history_text.tag_config("msg", foreground="#f3f4f6")

        # Initial greeting in history
        self._append_history("MAI", "Ready for voice interaction. Click Speak or press Spacebar.")

        # 4. Control Buttons Frame
        controls = tk.Frame(self.root, bg="#121826", pady=14, padx=16, highlightthickness=1, highlightbackground="#1f293d")
        controls.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_speak = tk.Button(
            controls,
            text="🎙 Speak",
            bg="#2563eb",
            fg="#ffffff",
            font=btn_font,
            relief=tk.FLAT,
            padx=14,
            pady=8,
            cursor="hand2",
            command=self.toggle_recording,
        )
        self.btn_speak.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        self.btn_stop = tk.Button(
            controls,
            text="⏹ Stop",
            bg="#b45309",
            fg="#ffffff",
            font=btn_font,
            relief=tk.FLAT,
            padx=12,
            pady=8,
            cursor="hand2",
            command=self.on_stop_speaking,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        self.btn_cancel = tk.Button(
            controls,
            text="❌ Cancel Task",
            bg="#991b1b",
            fg="#ffffff",
            font=btn_font,
            relief=tk.FLAT,
            padx=12,
            pady=8,
            cursor="hand2",
            command=self.on_cancel_task,
        )
        self.btn_cancel.pack(side=tk.LEFT, padx=4)

        # 5. Status / Connection Bar
        status_bar = tk.Frame(self.root, bg="#080c14", pady=6, padx=16)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.conn_lbl = tk.Label(status_bar, text="🟢 Connecting...", fg="#3b82f6", bg="#080c14", font=sub_font)
        self.conn_lbl.pack(side=tk.LEFT)

        self.perf_lbl = tk.Label(status_bar, text="Ready", fg="#9ca3af", bg="#080c14", font=sub_font)
        self.perf_lbl.pack(side=tk.RIGHT)

        # Keyboard shortcuts
        self.root.bind("<space>", lambda e: self.toggle_recording())
        self.root.bind("<Escape>", lambda e: self.on_stop_speaking())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _start_background_loop(self) -> None:
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._connect_client())
            self.loop.run_forever()

        self.bg_thread = threading.Thread(target=run_loop, daemon=True)
        self.bg_thread.start()

    async def _connect_client(self) -> None:
        try:
            await self.client.connect()
            self.root.after(0, lambda: self._update_conn_status(True, "Connected"))
            # Spawn the continuous event receiver loop
            asyncio.create_task(self._listen_for_server_events())
        except Exception as exc:
            self.root.after(0, lambda: self._update_conn_status(False, f"Error: {exc}"))

    async def _listen_for_server_events(self) -> None:
        """Continuously receive server events from the WebSocket."""
        while self.is_running and self.client.is_connected and self.client.ws:
            try:
                raw_msg = await self.client.ws.recv()
                event = json.loads(raw_msg)
                self.client.handle_server_event(event)
            except Exception:
                if not self.is_running or not self.client.is_connected:
                    break
                await asyncio.sleep(0.1)

    # Thread-safe UI updates
    def _on_client_state_change(self, state: ClientUIState, label: str) -> None:
        colors = {
            ClientUIState.DISCONNECTED: ("#6b7280", "● DISCONNECTED"),
            ClientUIState.CONNECTING: ("#3b82f6", "● CONNECTING..."),
            ClientUIState.IDLE: ("#10b981", "● READY"),
            ClientUIState.LISTENING: ("#f59e0b", "● LISTENING..."),
            ClientUIState.TRANSCRIBING: ("#06b6d4", "● UNDERSTANDING..."),
            ClientUIState.PROCESSING: ("#3b82f6", "● THINKING..."),
            ClientUIState.SPEAKING: ("#8b5cf6", "● SPEAKING..."),
            ClientUIState.INTERRUPTED: ("#ec4899", "● INTERRUPTED"),
            ClientUIState.ERROR: ("#ef4444", f"● {label.upper()}"),
            ClientUIState.CLOSING: ("#6b7280", "● CLOSING..."),
            ClientUIState.CLOSED: ("#6b7280", "● CLOSED"),
        }
        color, text = colors.get(state, ("#9ca3af", f"● {state.value}"))
        self.root.after(0, lambda: self._update_state_ui(text, color))

    def _update_state_ui(self, text: str, color: str) -> None:
        self.state_lbl.config(text=text, fg=color)
        if self.is_recording:
            self.btn_speak.config(text="🔴 Stop Speaking", bg="#dc2626")
        else:
            self.btn_speak.config(text="🎙 Speak", bg="#2563eb")

    def _update_conn_status(self, connected: bool, message: str) -> None:
        fg = "#10b981" if connected else "#ef4444"
        icon = "🟢" if connected else "🔴"
        self.conn_lbl.config(text=f"{icon} {message}", fg=fg)

    def _on_client_partial(self, partial_text: str) -> None:
        self.root.after(0, lambda: self.partial_lbl.config(text=partial_text))

    def _on_client_final(self, final_text: str) -> None:
        self.root.after(0, lambda: self._append_history("User", final_text))
        self.root.after(0, lambda: self.partial_lbl.config(text=""))

    def _on_client_turn_completed(self, user_msg: str, mai_msg: str, metrics: dict) -> None:
        self.root.after(0, lambda: self._append_history("MAI", mai_msg))
        mai_ms = metrics.get("mai_latency_ms", metrics.get("total_turn_turnaround_ms", 0))
        self.root.after(0, lambda: self.perf_lbl.config(text=f"Turnaround: {mai_ms}ms"))

    def _on_client_error(self, err_msg: str) -> None:
        self.root.after(0, lambda: self._append_history("SYSTEM", f"Error: {err_msg}"))

    def _append_history(self, speaker: str, text: str) -> None:
        self.history_text.config(state=tk.NORMAL)
        if speaker == "User":
            self.history_text.insert(tk.END, f"\nUser:\n", "user")
        elif speaker == "MAI":
            self.history_text.insert(tk.END, f"\nMAI:\n", "mai")
        else:
            self.history_text.insert(tk.END, f"\n{speaker}:\n", "user")
        self.history_text.insert(tk.END, f"{text}\n", "msg")
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)

    def toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.is_recording:
            return
        self.is_recording = True
        self.btn_speak.config(text="🔴 Stop Speaking", bg="#dc2626")

        async def _record_loop():
            try:
                await self.client.start_turn()
                self.client.device.open_input_stream()

                # Stream audio continuously from microphone while user is speaking
                while self.is_recording and self.client.is_connected:
                    chunk = await asyncio.to_thread(self.client.device.read_microphone_chunk)
                    if not self.is_recording:
                        break
                    await self.client.send_audio_chunk(chunk)

                    # Compute audio energy for visualizer
                    if len(chunk) >= 2:
                        samples = struct.unpack(f"<{len(chunk)//2}h", chunk)
                        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                        self.audio_level = min(1.0, rms / 10000.0)

                    await asyncio.sleep(0.01)

            except Exception as e:
                self.root.after(0, lambda: self._on_client_error(str(e)))
                self.is_recording = False
                self.root.after(0, lambda: self.btn_speak.config(text="🎙 Speak", bg="#2563eb"))

        if self.loop:
            asyncio.run_coroutine_threadsafe(_record_loop(), self.loop)

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        self.is_recording = False
        self.audio_level = 0.0
        self.btn_speak.config(text="🎙 Speak", bg="#2563eb")

        async def _finish_turn():
            try:
                await self.client.finish_turn()
            except Exception as e:
                self.root.after(0, lambda: self._on_client_error(str(e)))

        if self.loop:
            asyncio.run_coroutine_threadsafe(_finish_turn(), self.loop)

    def on_stop_speaking(self) -> None:
        if self.is_recording:
            self.stop_recording()
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.client.stop_speaking(), self.loop)

    def on_cancel_task(self) -> None:
        if self.is_recording:
            self.stop_recording()
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.client.cancel_task(), self.loop)

    def _animate_visualizer(self) -> None:
        """Animate visualizer canvas with dynamic waveform bars."""
        if not self.is_running:
            return

        try:
            self.canvas.delete("all")
            width = self.canvas.winfo_width() or 500
            height = self.canvas.winfo_height() or 44
            mid_y = height // 2

            if self.is_recording:
                # Active microphone energy bars
                num_bars = 24
                bar_width = (width - 40) / num_bars
                for i in range(num_bars):
                    amp = (self.audio_level * 0.8 + 0.1) * math.sin(time.time() * 8 + i * 0.5)
                    bar_h = max(3, abs(amp) * (height - 10))
                    x0 = 20 + i * bar_width + 2
                    x1 = x0 + bar_width - 4
                    y0 = mid_y - bar_h // 2
                    y1 = mid_y + bar_h // 2
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="#06b6d4", outline="")
            elif self.client.device.is_playing:
                # Active speaker output bars
                num_bars = 24
                bar_width = (width - 40) / num_bars
                for i in range(num_bars):
                    amp = 0.6 * math.sin(time.time() * 12 + i * 0.7)
                    bar_h = max(4, abs(amp) * (height - 10))
                    x0 = 20 + i * bar_width + 2
                    x1 = x0 + bar_width - 4
                    y0 = mid_y - bar_h // 2
                    y1 = mid_y + bar_h // 2
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="#8b5cf6", outline="")
            else:
                # Ambient center line
                self.canvas.create_line(10, mid_y, width - 10, mid_y, fill="#374151", width=2)
        except Exception:
            pass

        self.root.after(50, self._animate_visualizer)

    def on_closing(self) -> None:
        self.is_running = False
        self.is_recording = False
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.client.close(), self.loop)
        self.root.destroy()


def launch_desktop_gui(config: Optional[VoiceClientConfig] = None) -> None:
    """Entrypoint to run the desktop voice GUI application."""
    root = tk.Tk()
    app = PMAIVoiceDesktopApp(root, config=config)
    root.mainloop()


if __name__ == "__main__":
    launch_desktop_gui()
