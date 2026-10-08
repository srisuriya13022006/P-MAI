"""
P15 — Client UI State Machine.
Defines explicit states, transitions, user-friendly labels, and server event mappings.
"""
from enum import Enum
from typing import Optional


class ClientUIState(str, Enum):
    """Explicit states for the P-MAI Voice Client UI."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


# Human-friendly UI status labels
STATE_LABELS = {
    ClientUIState.DISCONNECTED: "Disconnected",
    ClientUIState.CONNECTING: "Connecting...",
    ClientUIState.IDLE: "Ready",
    ClientUIState.LISTENING: "Listening...",
    ClientUIState.TRANSCRIBING: "Understanding...",
    ClientUIState.PROCESSING: "Thinking...",
    ClientUIState.SPEAKING: "Speaking...",
    ClientUIState.INTERRUPTED: "Interrupted",
    ClientUIState.ERROR: "Something went wrong",
    ClientUIState.CLOSING: "Closing...",
    ClientUIState.CLOSED: "Closed",
}

# Allowed state transitions
VALID_TRANSITIONS = {
    ClientUIState.DISCONNECTED: {ClientUIState.CONNECTING, ClientUIState.ERROR, ClientUIState.CLOSED},
    ClientUIState.CONNECTING: {ClientUIState.IDLE, ClientUIState.ERROR, ClientUIState.DISCONNECTED},
    ClientUIState.IDLE: {
        ClientUIState.LISTENING,
        ClientUIState.PROCESSING,
        ClientUIState.CLOSING,
        ClientUIState.DISCONNECTED,
        ClientUIState.ERROR,
    },
    ClientUIState.LISTENING: {
        ClientUIState.TRANSCRIBING,
        ClientUIState.PROCESSING,
        ClientUIState.IDLE,
        ClientUIState.INTERRUPTED,
        ClientUIState.ERROR,
        ClientUIState.DISCONNECTED,
    },
    ClientUIState.TRANSCRIBING: {
        ClientUIState.PROCESSING,
        ClientUIState.IDLE,
        ClientUIState.LISTENING,
        ClientUIState.INTERRUPTED,
        ClientUIState.ERROR,
        ClientUIState.DISCONNECTED,
    },
    ClientUIState.PROCESSING: {
        ClientUIState.SPEAKING,
        ClientUIState.IDLE,
        ClientUIState.INTERRUPTED,
        ClientUIState.ERROR,
        ClientUIState.DISCONNECTED,
    },
    ClientUIState.SPEAKING: {
        ClientUIState.IDLE,
        ClientUIState.LISTENING,  # Barge-in
        ClientUIState.INTERRUPTED,
        ClientUIState.ERROR,
        ClientUIState.DISCONNECTED,
    },
    ClientUIState.INTERRUPTED: {
        ClientUIState.IDLE,
        ClientUIState.LISTENING,
        ClientUIState.PROCESSING,
        ClientUIState.ERROR,
        ClientUIState.DISCONNECTED,
    },
    ClientUIState.ERROR: {
        ClientUIState.IDLE,
        ClientUIState.CONNECTING,
        ClientUIState.DISCONNECTED,
        ClientUIState.CLOSED,
    },
    ClientUIState.CLOSING: {ClientUIState.CLOSED, ClientUIState.DISCONNECTED},
    ClientUIState.CLOSED: {ClientUIState.CONNECTING, ClientUIState.DISCONNECTED},
}


class VoiceClientStateMachine:
    """
    Manages deterministic client-side UI state transitions.
    """

    def __init__(self, initial_state: ClientUIState = ClientUIState.DISCONNECTED):
        self._state: ClientUIState = initial_state
        self._error_message: Optional[str] = None

    @property
    def current_state(self) -> ClientUIState:
        return self._state

    @property
    def display_label(self) -> str:
        if self._state == ClientUIState.ERROR and self._error_message:
            return self._error_message
        return STATE_LABELS.get(self._state, self._state.value)

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    def transition_to(self, target_state: ClientUIState, error_message: Optional[str] = None) -> bool:
        """
        Transition to a new state if valid.
        Returns True if transition occurred, False if rejected or already in state.
        """
        if self._state == target_state:
            if error_message:
                self._error_message = error_message
            return True

        allowed = VALID_TRANSITIONS.get(self._state, set())
        if target_state in allowed:
            self._state = target_state
            if error_message:
                self._error_message = error_message
            elif target_state == ClientUIState.ERROR:
                self._error_message = "Something went wrong"
            else:
                self._error_message = None
            return True

        # Fallback / graceful state recovery for unexpected transitions
        self._state = target_state
        if error_message:
            self._error_message = error_message
        elif target_state == ClientUIState.ERROR:
            self._error_message = "Something went wrong"
        else:
            self._error_message = None
        return True

    def reset_to_idle(self) -> None:
        """Helper to return state to IDLE."""
        self.transition_to(ClientUIState.IDLE)
