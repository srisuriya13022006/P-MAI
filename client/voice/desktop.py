"""
P15 — Desktop Voice Client Launcher.
Supports launching either the native Windows Desktop GUI or opening the Web voice interface.
Usage:
    python -m client.voice.desktop          # Launches native Tkinter GUI
    python -m client.voice.desktop --web    # Launches browser Web Voice Client
"""
import argparse
import sys
import webbrowser

from client.voice.config import VoiceClientConfig
from client.voice.desktop_gui import launch_desktop_gui


def main():
    parser = argparse.ArgumentParser(description="P-MAI Desktop Voice Client")
    parser.add_argument("--web", action="store_true", help="Open Web voice interface in default browser")
    parser.add_argument("--url", default="http://localhost:8000/voice/ui", help="URL of web voice interface")
    parser.add_argument("--ws-url", default="ws://localhost:8000/voice/realtime", help="WebSocket voice endpoint")
    parser.add_argument("--conversation-id", default="desktop-session", help="Conversation ID")
    parser.add_argument("--user-id", default="default", help="User ID")

    args = parser.parse_args()

    if args.web:
        print(f"Opening P-MAI Web Voice Client at {args.url}...")
        webbrowser.open(args.url)
        sys.exit(0)

    config = VoiceClientConfig(
        backend_http_url=args.url.split("/voice")[0] if "/voice" in args.url else "http://localhost:8000",
        backend_ws_url=args.ws_url,
        conversation_id=args.conversation_id,
        user_id=args.user_id,
    )
    print("Launching P-MAI Native Desktop Voice Assistant...")
    launch_desktop_gui(config)


if __name__ == "__main__":
    main()
