/**
 * P15 — P-MAI Real-Time Voice Web Client Controller.
 *
 * Handles:
 * - microphone capture (16kHz Mono PCM)
 * - realtime WebSocket lifecycle
 * - streaming response audio buffering
 * - safe WAV / raw PCM playback
 * - visualizer rendering
 * - barge-in
 * - stop vs cancel
 *
 * IMPORTANT:
 * Voice is only a presentation / I/O layer.
 * The backend remains authoritative for:
 * routing, planning, tools, memory, task state,
 * clarification, recovery, and grounding.
 */

// ============================================================
// State Machine
// ============================================================

const STATES = {
  DISCONNECTED: { label: "Disconnected", color: "#6b7280" },
  CONNECTING: { label: "Connecting...", color: "#3b82f6" },
  IDLE: { label: "Ready", color: "#10b981" },
  LISTENING: { label: "Listening...", color: "#f59e0b" },
  TRANSCRIBING: { label: "Understanding...", color: "#06b6d4" },
  PROCESSING: { label: "Thinking...", color: "#3b82f6" },
  SPEAKING: { label: "Speaking...", color: "#8b5cf6" },
  INTERRUPTED: { label: "Interrupted", color: "#ec4899" },
  ERROR: { label: "Something went wrong", color: "#ef4444" },
  CLOSED: { label: "Closed", color: "#6b7280" },
};


// ============================================================
// Client
// ============================================================

class PMAIWebVoiceClient {
  constructor() {
    // ----------------------------------------------------------
    // WebSocket
    // ----------------------------------------------------------

    this.ws = null;
    this.currentState = "DISCONNECTED";

    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 3;
    this.reconnectTimer = null;

    // ----------------------------------------------------------
    // Recording
    // ----------------------------------------------------------

    this.isRecording = false;
    this.micStream = null;
    this.processorNode = null;
    this.sourceNode = null;

    // ----------------------------------------------------------
    // Audio Context
    // ----------------------------------------------------------

    this.audioCtx = null;
    this.analyserNode = null;

    // ----------------------------------------------------------
    // Playback state
    // ----------------------------------------------------------

    this.isPlayingAudio = false;
    this.currentAudioSource = null;

    /*
     * IMPORTANT:
     *
     * These are RAW BYTE CHUNKS received from RESPONSE_CHUNK.
     * We DO NOT decode them individually.
     *
     * We assemble all chunks belonging to one response and
     * decode/play the complete audio payload safely.
     */
    this.audioQueue = [];
    this.responseAudioChunks = this.audioQueue;
    this.responseAudioBytes = 0;

    this.responseAudioContentType = null;
    this.responseHasAudio = false;

    // Used to invalidate stale asynchronous decode operations.
    this.playbackGeneration = 0;

    // ----------------------------------------------------------
    // Performance metrics
    // ----------------------------------------------------------

    this.turnStart = 0;
    this.timeAudioEnd = 0;
    this.responseStartTime = 0;
    this.responseEndTime = 0;

    // ----------------------------------------------------------
    // UI elements
    // ----------------------------------------------------------

    this.elStateBadge = document.getElementById("state-badge");
    this.elStateText = document.getElementById("state-text");
    this.elPulseIndicator = document.getElementById("pulse-indicator");

    this.elConnText = document.getElementById("conn-text");
    this.elConnDot = document.getElementById("conn-dot");

    this.elVisualizer = document.getElementById("visualizer");
    this.elLiveTranscript = document.getElementById("live-transcript");
    this.elConversationFeed =
      document.getElementById("conversation-feed");
    this.elErrorBanner = document.getElementById("error-banner");

    this.elBtnSpeak = document.getElementById("btn-speak");
    this.elBtnStop = document.getElementById("btn-stop");
    this.elBtnCancel = document.getElementById("btn-cancel");

    this.elMetricsBar = document.getElementById("metrics-bar");

    // ----------------------------------------------------------
    // Canvas
    // ----------------------------------------------------------

    this.canvasCtx = null;

    if (this.elVisualizer) {
      this.canvasCtx = this.elVisualizer.getContext("2d");
    }

    // ----------------------------------------------------------
    // Initialize
    // ----------------------------------------------------------

    this.setupVisualizerLoop();
    this.attachEventListeners();
    this.initWebSocket();
  }


  // ============================================================
  // State Handling
  // ============================================================

  setState(stateName, customMsg = null) {
    this.currentState = stateName;

    const info = STATES[stateName] || STATES.IDLE;
    const label = customMsg || info.label;

    if (this.elStateText) {
      this.elStateText.textContent = label;
    }

    if (this.elPulseIndicator) {
      this.elPulseIndicator.style.background = info.color;
      this.elPulseIndicator.style.boxShadow =
        `0 0 10px ${info.color}`;
    }

    if (this.elBtnSpeak) {
      this.elBtnSpeak.classList.toggle(
        "recording",
        this.isRecording
      );

      this.elBtnSpeak.innerHTML = this.isRecording
        ? "🔴 Stop Speaking"
        : "🎙 Speak";
    }

    if (this.elErrorBanner) {
      if (stateName === "ERROR") {
        this.elErrorBanner.textContent = label;
        this.elErrorBanner.style.display = "block";
      } else {
        this.elErrorBanner.style.display = "none";
      }
    }
  }


  setConnectionStatus(connected, text) {
    if (this.elConnDot) {
      this.elConnDot.style.background = connected
        ? "#10b981"
        : "#ef4444";

      this.elConnDot.style.boxShadow =
        `0 0 8px ${
          connected ? "#10b981" : "#ef4444"
        }`;
    }

    if (this.elConnText) {
      this.elConnText.textContent = text;
    }
  }


  // ============================================================
  // WebSocket
  // ============================================================

  initWebSocket() {
    this.setState("CONNECTING");
    this.setConnectionStatus(false, "Connecting...");

    const protocol =
      window.location.protocol === "https:" ? "wss:" : "ws:";

    const host =
      window.location.host || "localhost:8000";

    const wsUrl =
      `${protocol}//${host}/voice/realtime` +
      `?conversation_id=web-client&user_id=web-user`;

    try {
      this.ws = new WebSocket(wsUrl);
    } catch (err) {
      this.handleDisconnect(err);
      return;
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;

      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }

      this.setConnectionStatus(true, "Connected");
      this.setState("IDLE");
    };


    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleServerEvent(msg);
      } catch (err) {
        this.setState(
          "ERROR",
          "Malformed server response"
        );
      }
    };


    this.ws.onerror = () => {
      this.setState(
        "ERROR",
        "WebSocket connection error"
      );
    };


    this.ws.onclose = () => {
      this.handleDisconnect();
    };
  }


  handleDisconnect(err = null) {
    this.setConnectionStatus(false, "Disconnected");

    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;

      this.setState(
        "CONNECTING",
        `Reconnecting (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`
      );

      this.reconnectTimer = setTimeout(() => {
        this.initWebSocket();
      }, 2000);
    } else {
      this.setState(
        "DISCONNECTED",
        "Connection lost. Please refresh or reconnect."
      );
    }
  }


  // ============================================================
  // Server Event Handling
  // ============================================================

  handleServerEvent(event) {
    if (!event || typeof event !== "object") {
      this.setState("ERROR", "Invalid server event");
      return;
    }

    const type = event.type;

    switch (type) {
      // --------------------------------------------------------
      // Speech events
      // --------------------------------------------------------

      case "SPEECH_START":
        this.setState("LISTENING");
        break;


      case "SPEECH_END":
        this.setState("TRANSCRIBING");
        break;


      // --------------------------------------------------------
      // STT
      // --------------------------------------------------------

      case "TRANSCRIPT_PARTIAL":
        if (this.elLiveTranscript) {
          this.elLiveTranscript.textContent =
            event.transcript || "";
        }
        break;


      case "TRANSCRIPT_FINAL":
        if (this.elLiveTranscript) {
          this.elLiveTranscript.textContent = "";
        }

        if (event.transcript) {
          this.appendMessage(
            "User",
            event.transcript
          );
        }

        this.setState("PROCESSING");
        break;


      // --------------------------------------------------------
      // Response start
      // --------------------------------------------------------

      case "RESPONSE_START":
        this.beginResponseAudio(event);
        this.setState("PROCESSING");
        break;


      // --------------------------------------------------------
      // IMPORTANT:
      //
      // Do NOT decode each RESPONSE_CHUNK separately.
      // Just buffer the raw bytes.
      // --------------------------------------------------------

      case "RESPONSE_CHUNK":
        this.setState("SPEAKING");

        if (event.audio_base64) {
          this.bufferResponseAudioChunk(
            event.audio_base64,
            event
          );
        }
        break;


      // --------------------------------------------------------
      // Response end
      // --------------------------------------------------------

      case "RESPONSE_END":
        this.responseEndTime = performance.now();

        if (event.text_chunk) {
          this.appendMessage(
            "MAI",
            event.text_chunk
          );
        }

        if (event.text_response) {
          this.appendMessage(
            "MAI",
            event.text_response
          );
        }

        if (event.metadata) {
          const maiMs =
            Number(event.metadata.mai_latency_ms) || 0;

          const ttsMs =
            Number(event.metadata.tts_latency_ms) || 0;

          if (this.elMetricsBar) {
            this.elMetricsBar.textContent =
              `MAI: ${maiMs}ms | TTS: ${ttsMs}ms`;
          }
        }

        /*
         * Playback begins ONLY after the full response
         * audio stream is available.
         *
         * This avoids decodeAudioData() failures caused by
         * arbitrary WAV/PCM fragments.
         */
        if (this.responseHasAudio) {
          this.playBufferedResponseAudio();
        } else {
          this.setState("IDLE");
        }

        break;


      // --------------------------------------------------------
      // Backend interruption
      // --------------------------------------------------------

      case "INTERRUPT":
        this.stopPlayback();

        this.setState("INTERRUPTED");

        window.setTimeout(() => {
          if (
            this.currentState === "INTERRUPTED"
          ) {
            this.setState("IDLE");
          }
        }, 800);

        break;


      // --------------------------------------------------------
      // Error
      // --------------------------------------------------------

      case "ERROR":
        this.stopPlayback();

        this.setState(
          "ERROR",
          `${event.error_code || "ERROR"}: ${
            event.error_message || "Server error"
          }`
        );
        break;


      // --------------------------------------------------------
      // Close
      // --------------------------------------------------------

      case "CLOSE":
        this.stopPlayback();
        this.setState("CLOSED");
        break;


      default:
        /*
         * Ignore unknown future events safely.
         */
        break;
    }
  }


  // ============================================================
  // Conversation UI
  // ============================================================

  appendMessage(speaker, text) {
    if (!this.elConversationFeed || !text) {
      return;
    }

    const card = document.createElement("div");

    card.className =
      `message-card ${
        speaker === "User"
          ? "user-message"
          : "mai-message"
      }`;

    const label = document.createElement("div");

    label.className =
      speaker === "User"
        ? "user-label"
        : "mai-label";

    label.textContent = speaker;

    const content =
      document.createElement("div");

    content.textContent = text;

    card.appendChild(label);
    card.appendChild(content);

    this.elConversationFeed.appendChild(card);

    // Keep history bounded to 20 messages.
    while (
      this.elConversationFeed.children.length > 20
    ) {
      this.elConversationFeed.removeChild(
        this.elConversationFeed.firstChild
      );
    }

    this.elConversationFeed.scrollTop =
      this.elConversationFeed.scrollHeight;
  }


  // ============================================================
  // Audio Context
  // ============================================================

  async ensureAudioContext() {
    if (!this.audioCtx) {
      this.audioCtx =
        new (
          window.AudioContext ||
          window.webkitAudioContext
        )();
    }

    /*
     * Chrome may suspend an AudioContext until it is resumed
     * from a user interaction.
     */
    if (this.audioCtx.state === "suspended") {
      try {
        await this.audioCtx.resume();
      } catch (err) {
        console.warn(
          "Unable to resume AudioContext:",
          err
        );
      }
    }

    return this.audioCtx;
  }


  // ============================================================
  // Recording
  // ============================================================

  async startRecording() {
    /*
     * Barge-in:
     *
     * If MAI is currently speaking, stop local playback first.
     * This does NOT cancel the task.
     */
    if (
      this.currentState === "SPEAKING" ||
      this.isPlayingAudio
    ) {
      this.stopPlayback();
    }

    /*
     * Clear stale response audio before starting a new turn.
     */
    this.resetResponseAudio();

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      this.setState(
        "ERROR",
        "Microphone not supported on this browser."
      );
      return;
    }

    try {
      this.micStream =
        await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
    } catch (err) {
      this.setState(
        "ERROR",
        "Microphone permission denied or device unavailable."
      );
      return;
    }

    try {
      await this.ensureAudioContext();
    } catch (err) {
      this.setState(
        "ERROR",
        "Web Audio is unavailable in this browser."
      );

      this.cleanupMicrophone();
      return;
    }

    try {
      this.sourceNode =
        this.audioCtx.createMediaStreamSource(
          this.micStream
        );

      this.analyserNode =
        this.audioCtx.createAnalyser();

      this.analyserNode.fftSize = 256;

      this.sourceNode.connect(
        this.analyserNode
      );

      /*
       * The current P15 microphone implementation uses
       * ScriptProcessorNode for compatibility.
       *
       * NOTE:
       * Chrome may show a deprecation warning.
       * This is unrelated to the TTS decoding bug.
       */
      const bufferSize = 4096;

      this.processorNode =
        this.audioCtx.createScriptProcessor(
          bufferSize,
          1,
          1
        );

      this.processorNode.onaudioprocess = (e) => {
        if (!this.isRecording) {
          return;
        }

        try {
          const inputData =
            e.inputBuffer.getChannelData(0);

          const pcm16 =
            this.downsampleAndConvert(
              inputData,
              this.audioCtx.sampleRate,
              16000
            );

          this.sendAudioChunk(pcm16);
        } catch (err) {
          console.warn(
            "Microphone audio processing failed:",
            err
          );
        }
      };

      this.sourceNode.connect(
        this.processorNode
      );

      /*
       * Keep the processing node alive.
       */
      this.processorNode.connect(
        this.audioCtx.destination
      );

      this.isRecording = true;
      this.turnStart = performance.now();

      this.setState("LISTENING");

      if (
        this.ws &&
        this.ws.readyState === WebSocket.OPEN
      ) {
        this.ws.send(
          JSON.stringify({
            type: "AUDIO_START",
          })
        );
      }
    } catch (err) {
      this.cleanupMicrophone();

      this.setState(
        "ERROR",
        "Unable to initialize microphone audio."
      );
    }
  }


  stopRecording() {
    if (!this.isRecording) {
      return;
    }

    this.isRecording = false;
    this.timeAudioEnd = performance.now();

    this.cleanupMicrophone();

    this.setState("TRANSCRIBING");

    if (
      this.ws &&
      this.ws.readyState === WebSocket.OPEN
    ) {
      this.ws.send(
        JSON.stringify({
          type: "AUDIO_END",
        })
      );
    }
  }


  cleanupMicrophone() {
    if (this.processorNode) {
      try {
        this.processorNode.disconnect();
      } catch (err) {
        // Ignore cleanup failure.
      }

      this.processorNode = null;
    }

    if (this.sourceNode) {
      try {
        this.sourceNode.disconnect();
      } catch (err) {
        // Ignore cleanup failure.
      }

      this.sourceNode = null;
    }

    if (this.micStream) {
      this.micStream
        .getTracks()
        .forEach((track) => {
          try {
            track.stop();
          } catch (err) {
            // Ignore cleanup failure.
          }
        });

      this.micStream = null;
    }
  }


  toggleRecording() {
    if (this.isRecording) {
      this.stopRecording();
    } else {
      this.startRecording();
    }
  }


  // ============================================================
  // Sending Audio
  // ============================================================

  sendAudioChunk(pcm16Bytes) {
    if (
      !this.ws ||
      this.ws.readyState !== WebSocket.OPEN
    ) {
      return;
    }

    if (
      !pcm16Bytes ||
      pcm16Bytes.byteLength === 0
    ) {
      return;
    }

    const base64Audio =
      this.arrayBufferToBase64(
        pcm16Bytes.buffer
      );

    /*
     * Preserve the existing backend contract.
     *
     * NOTE:
     * These bytes are raw PCM, while the historical client
     * labels them audio/wav. Do not change this contract here
     * unless the backend is also updated to expect a PCM MIME
     * type.
     */
    this.ws.send(
      JSON.stringify({
        type: "AUDIO_CHUNK",
        audio_base64: base64Audio,
        content_type: "audio/wav",
      })
    );
  }


  // ============================================================
  // PCM Conversion
  // ============================================================

  downsampleAndConvert(
    buffer,
    sampleRate,
    outSampleRate
  ) {
    if (
      outSampleRate === sampleRate
    ) {
      return this.floatTo16BitPCM(buffer);
    }

    const sampleRateRatio =
      sampleRate / outSampleRate;

    const newLength =
      Math.round(
        buffer.length / sampleRateRatio
      );

    const result =
      new Int16Array(newLength);

    let offsetResult = 0;
    let offsetBuffer = 0;

    while (
      offsetResult < result.length
    ) {
      const nextOffsetBuffer =
        Math.round(
          (offsetResult + 1) *
          sampleRateRatio
        );

      let accum = 0;
      let count = 0;

      for (
        let i = offsetBuffer;
        i < nextOffsetBuffer &&
        i < buffer.length;
        i++
      ) {
        accum += buffer[i];
        count++;
      }

      const s =
        Math.max(
          -1,
          Math.min(
            1,
            count > 0
              ? accum / count
              : 0
          )
        );

      result[offsetResult] =
        s < 0
          ? s * 0x8000
          : s * 0x7fff;

      offsetResult++;
      offsetBuffer =
        nextOffsetBuffer;
    }

    return result;
  }


  floatTo16BitPCM(input) {
    const output =
      new Int16Array(input.length);

    for (
      let i = 0;
      i < input.length;
      i++
    ) {
      const s =
        Math.max(
          -1,
          Math.min(1, input[i])
        );

      output[i] =
        s < 0
          ? s * 0x8000
          : s * 0x7fff;
    }

    return output;
  }


  // ============================================================
  // Base64 Helpers
  // ============================================================

  arrayBufferToBase64(buffer) {
    let binary = "";

    const bytes =
      new Uint8Array(buffer);

    const len =
      bytes.byteLength;

    for (
      let i = 0;
      i < len;
      i++
    ) {
      binary += String.fromCharCode(
        bytes[i]
      );
    }

    return window.btoa(binary);
  }


  base64ToArrayBuffer(base64) {
    const binaryString =
      window.atob(base64);

    const len =
      binaryString.length;

    const bytes =
      new Uint8Array(len);

    for (
      let i = 0;
      i < len;
      i++
    ) {
      bytes[i] =
        binaryString.charCodeAt(i);
    }

    return bytes.buffer;
  }


  // ============================================================
  // RESPONSE AUDIO BUFFERING
  // ============================================================

  beginResponseAudio(event = null) {
    /*
     * Invalidate any previous asynchronous decoder.
     */
    this.playbackGeneration++;

    this.resetResponseAudio();

    if (event) {
      this.responseAudioContentType =
        event.audio_content_type ||
        event.content_type ||
        (event.metadata &&
          event.metadata.audio_content_type) ||
        null;
    }

    this.responseStartTime =
      performance.now();
  }


  resetResponseAudio() {
    /*
     * Clear only the pending response bytes.
     * Current actual playback is controlled separately by
     * stopPlayback().
     */
    this.audioQueue.length = 0;
    this.responseAudioBytes = 0;
    this.responseAudioContentType = null;
    this.responseHasAudio = false;
  }


  bufferResponseAudioChunk(
    base64Data,
    event = null
  ) {
    if (
      typeof base64Data !== "string" ||
      base64Data.length === 0
    ) {
      return;
    }

    try {
      const arrayBuf =
        this.base64ToArrayBuffer(
          base64Data
        );

      if (!arrayBuf || arrayBuf.byteLength === 0) {
        return;
      }

      /*
       * Record content type if backend provides it.
       */
      if (event) {
        this.responseAudioContentType =
          event.audio_content_type ||
          event.content_type ||
          this.responseAudioContentType ||
          (event.metadata &&
            event.metadata.audio_content_type) ||
          null;
      }

      /*
       * IMPORTANT:
       * Store the bytes.
       *
       * Do not call decodeAudioData() here.
       */
      this.audioQueue.push(arrayBuf);

      this.responseAudioBytes +=
        arrayBuf.byteLength;

      this.responseHasAudio = true;

      /*
       * Debug info is intentionally lightweight.
       * Remove these logs later if desired.
       */
      console.debug(
        "[P-MAI audio] received chunk:",
        {
          bytes: arrayBuf.byteLength,
          totalBytes:
            this.responseAudioBytes,
          chunks:
            this.audioQueue.length,
          contentType:
            this.responseAudioContentType,
        }
      );
    } catch (err) {
      console.warn(
        "[P-MAI audio] base64 decode failed:",
        err
      );
    }
  }


  // ============================================================
  // Byte Concatenation
  // ============================================================

  concatArrayBuffers(buffers) {
    const totalLength =
      buffers.reduce(
        (sum, buffer) =>
          sum + buffer.byteLength,
        0
      );

    const result =
      new Uint8Array(totalLength);

    let offset = 0;

    for (
      const buffer of buffers
    ) {
      result.set(
        new Uint8Array(buffer),
        offset
      );

      offset +=
        buffer.byteLength;
    }

    return result.buffer;
  }


  // ============================================================
  // Audio Format Detection
  // ============================================================

  isWavBuffer(arrayBuffer) {
    if (
      !arrayBuffer ||
      arrayBuffer.byteLength < 12
    ) {
      return false;
    }

    const bytes =
      new Uint8Array(arrayBuffer);

    /*
     * RIFF
     * WAVE
     */
    return (
      bytes[0] === 0x52 && // R
      bytes[1] === 0x49 && // I
      bytes[2] === 0x46 && // F
      bytes[3] === 0x46 && // F
      bytes[8] === 0x57 && // W
      bytes[9] === 0x41 && // A
      bytes[10] === 0x56 && // V
      bytes[11] === 0x45    // E
    );
  }


  // ============================================================
  // Raw PCM → WAV
  // ============================================================

  createWavFromPcm(
    pcmBuffer,
    sampleRate = 16000,
    channels = 1,
    bitsPerSample = 16
  ) {
    const pcmBytes =
      new Uint8Array(pcmBuffer);

    const bytesPerSample =
      bitsPerSample / 8;

    const blockAlign =
      channels * bytesPerSample;

    const byteRate =
      sampleRate * blockAlign;

    const dataSize =
      pcmBytes.byteLength;

    const buffer =
      new ArrayBuffer(
        44 + dataSize
      );

    const view =
      new DataView(buffer);

    const writeAscii =
      (offset, text) => {
        for (
          let i = 0;
          i < text.length;
          i++
        ) {
          view.setUint8(
            offset + i,
            text.charCodeAt(i)
          );
        }
      };

    // RIFF
    writeAscii(0, "RIFF");

    view.setUint32(
      4,
      36 + dataSize,
      true
    );

    // WAVE
    writeAscii(8, "WAVE");

    // fmt
    writeAscii(12, "fmt ");

    view.setUint32(
      16,
      16,
      true
    );

    // PCM format = 1
    view.setUint16(
      20,
      1,
      true
    );

    view.setUint16(
      22,
      channels,
      true
    );

    view.setUint32(
      24,
      sampleRate,
      true
    );

    view.setUint32(
      28,
      byteRate,
      true
    );

    view.setUint16(
      32,
      blockAlign,
      true
    );

    view.setUint16(
      34,
      bitsPerSample,
      true
    );

    // data
    writeAscii(36, "data");

    view.setUint32(
      40,
      dataSize,
      true
    );

    new Uint8Array(
      buffer,
      44
    ).set(pcmBytes);

    return buffer;
  }


  // ============================================================
  // Complete Buffered Response Playback
  // ============================================================

  async playBufferedResponseAudio() {
    const generation =
      this.playbackGeneration;

    if (
      !this.audioQueue.length ||
      this.responseAudioBytes <= 0
    ) {
      this.setState("IDLE");
      return;
    }

    try {
      /*
       * Build one complete byte buffer.
       */
      const combinedBuffer =
        this.concatArrayBuffers(
          this.audioQueue
        );

      if (
        !combinedBuffer ||
        combinedBuffer.byteLength === 0
      ) {
        this.setState("IDLE");
        return;
      }

      /*
       * Invalidate stale work if the user has already
       * interrupted playback.
       */
      if (
        generation !==
        this.playbackGeneration
      ) {
        return;
      }

      /*
       * IMPORTANT:
       *
       * If the complete response is a valid WAV,
       * decode the complete WAV.
       *
       * Otherwise assume raw PCM and wrap it into WAV.
       */
      let playableBuffer =
        combinedBuffer;

      if (
        !this.isWavBuffer(
          combinedBuffer
        )
      ) {
        playableBuffer =
          this.createWavFromPcm(
            combinedBuffer,
            16000,
            1,
            16
          );

        console.debug(
          "[P-MAI audio] raw PCM detected; wrapped as WAV",
          {
            pcmBytes:
              combinedBuffer.byteLength,
            wavBytes:
              playableBuffer.byteLength,
          }
        );
      } else {
        console.debug(
          "[P-MAI audio] complete WAV detected",
          {
            wavBytes:
              combinedBuffer.byteLength,
          }
        );
      }

      const ctx =
        await this.ensureAudioContext();

      if (
        generation !==
        this.playbackGeneration
      ) {
        return;
      }

      /*
       * Decode ONE complete valid audio payload.
       *
       * This is the key fix for the previous:
       *
       * EncodingError: Unable to decode audio data
       */
      const audioBuffer =
        await ctx.decodeAudioData(
          playableBuffer.slice(0)
        );

      if (
        generation !==
        this.playbackGeneration
      ) {
        return;
      }

      /*
       * Stop any previous source before starting.
       */
      this.stopCurrentAudioSourceOnly();

      const source =
        ctx.createBufferSource();

      source.buffer =
        audioBuffer;

      /*
       * Connect to destination.
       */
      source.connect(
        ctx.destination
      );

      /*
       * Connect to visualizer analyser when available.
       *
       * analyserNode may still be connected to the microphone,
       * which is fine for visualization.
       */
      if (this.analyserNode) {
        try {
          source.connect(
            this.analyserNode
          );
        } catch (err) {
          // Visualizer connection is optional.
        }
      }

      this.currentAudioSource =
        source;

      this.isPlayingAudio =
        true;

      this.setState("SPEAKING");

      source.onended = () => {
        /*
         * Only update state if this source is still current.
         */
        if (
          this.currentAudioSource ===
          source
        ) {
          this.isPlayingAudio = false;
          this.currentAudioSource = null;

          /*
           * Playback completed.
           */
          if (
            this.currentState ===
            "SPEAKING"
          ) {
            this.setState("IDLE");
          }
        }
      };

      source.start(0);

      console.debug(
        "[P-MAI audio] playback started",
        {
          durationSeconds:
            audioBuffer.duration,
          sampleRate:
            audioBuffer.sampleRate,
          channels:
            audioBuffer.numberOfChannels,
        }
      );
    } catch (err) {
      /*
       * Do not silently fail anymore.
       */
      console.error(
        "[P-MAI audio] playback failed:",
        err
      );

      this.isPlayingAudio = false;
      this.currentAudioSource = null;

      this.setState(
        "ERROR",
        "Unable to play MAI audio in this browser."
      );
    } finally {
      /*
       * The byte buffers are no longer needed after playback
       * has been prepared.
       */
      if (
        generation ===
        this.playbackGeneration
      ) {
        this.resetResponseAudio();
      }
    }
  }


  // ============================================================
  // Playback Controls
  // ============================================================

  stopCurrentAudioSourceOnly() {
    if (!this.currentAudioSource) {
      return;
    }

    try {
      this.currentAudioSource.onended =
        null;

      this.currentAudioSource.stop();
    } catch (err) {
      // Already stopped / naturally ended.
    }

    this.currentAudioSource =
      null;

    this.isPlayingAudio =
      false;
  }


  stopPlayback() {
    /*
     * Invalidate pending async decoding.
     */
    this.playbackGeneration++;

    /*
     * Stop currently playing audio.
     */
    this.stopCurrentAudioSourceOnly();

    /*
     * Clear any queued/buffered audio.
     */
    this.resetResponseAudio();
  }


  // ============================================================
  // Stop Speaking
  // ============================================================

  onStopSpeaking() {
    /*
     * This means:
     *
     * STOP AUDIO ONLY
     *
     * It does not cancel P8 task state.
     */
    this.stopPlayback();

    if (
      this.ws &&
      this.ws.readyState === WebSocket.OPEN
    ) {
      this.ws.send(
        JSON.stringify({
          type: "INTERRUPT",
          reason: "stop_speaking_button",
        })
      );
    }

    this.setState("IDLE");
  }


  // ============================================================
  // Cancel Task
  // ============================================================

  onCancelTask() {
    this.stopPlayback();

    /*
     * Preserve the existing backend cancellation mechanism.
     */
    if (
      this.ws &&
      this.ws.readyState === WebSocket.OPEN
    ) {
      this.ws.send(
        JSON.stringify({
          type: "AUDIO_START",
        })
      );

      window.setTimeout(() => {
        if (
          this.ws &&
          this.ws.readyState === WebSocket.OPEN
        ) {
          this.ws.send(
            JSON.stringify({
              type: "AUDIO_END",
            })
          );
        }
      }, 50);
    }

    this.appendMessage(
      "User",
      "Cancel active task"
    );

    this.setState("IDLE");
  }


  // ============================================================
  // Visualizer
  // ============================================================

  setupVisualizerLoop() {
    const draw = () => {
      requestAnimationFrame(draw);

      if (
        !this.canvasCtx ||
        !this.elVisualizer
      ) {
        return;
      }

      const width =
        this.elVisualizer.width;

      const height =
        this.elVisualizer.height;

      this.canvasCtx.clearRect(
        0,
        0,
        width,
        height
      );

      if (
        this.analyserNode &&
        (
          this.isRecording ||
          this.isPlayingAudio
        )
      ) {
        const bufferLength =
          this.analyserNode
            .frequencyBinCount;

        const dataArray =
          new Uint8Array(
            bufferLength
          );

        this.analyserNode
          .getByteFrequencyData(
            dataArray
          );

        const barWidth =
          (width / bufferLength) *
          2.5;

        let x = 0;

        /*
         * Create gradient once per frame.
         */
        const gradient =
          this.canvasCtx.createLinearGradient(
            0,
            height,
            0,
            0
          );

        gradient.addColorStop(
          0,
          "#06b6d4"
        );

        gradient.addColorStop(
          1,
          "#8b5cf6"
        );

        this.canvasCtx.fillStyle =
          gradient;

        for (
          let i = 0;
          i < bufferLength;
          i++
        ) {
          const barHeight =
            (dataArray[i] / 255) *
            height;

          this.canvasCtx.fillRect(
            x,
            height - barHeight,
            barWidth,
            barHeight
          );

          x +=
            barWidth + 1;
        }
      } else {
        // Idle ambient line.
        this.canvasCtx.beginPath();

        this.canvasCtx.moveTo(
          0,
          height / 2
        );

        this.canvasCtx.lineTo(
          width,
          height / 2
        );

        this.canvasCtx.strokeStyle =
          "rgba(255, 255, 255, 0.1)";

        this.canvasCtx.lineWidth = 1;

        this.canvasCtx.stroke();
      }
    };

    draw();
  }


  // ============================================================
  // Event Listeners
  // ============================================================

  attachEventListeners() {
    if (this.elBtnSpeak) {
      this.elBtnSpeak.addEventListener(
        "click",
        () => this.toggleRecording()
      );
    }

    if (this.elBtnStop) {
      this.elBtnStop.addEventListener(
        "click",
        () => this.onStopSpeaking()
      );
    }

    if (this.elBtnCancel) {
      this.elBtnCancel.addEventListener(
        "click",
        () => this.onCancelTask()
      );
    }

    // Keyboard shortcuts.
    window.addEventListener(
      "keydown",
      (e) => {
        /*
         * Space:
         * Start/stop recording.
         */
        if (
          e.code === "Space" &&
          e.target === document.body
        ) {
          e.preventDefault();

          this.toggleRecording();
        }

        /*
         * Escape:
         * Stop speaking only.
         */
        else if (
          e.code === "Escape"
        ) {
          e.preventDefault();

          this.onStopSpeaking();
        }
      }
    );
  }
}


// ============================================================
// Instantiate on page load
// ============================================================

window.addEventListener(
  "DOMContentLoaded",
  () => {
    window.pmaiVoiceClient =
      new PMAIWebVoiceClient();
  }
);