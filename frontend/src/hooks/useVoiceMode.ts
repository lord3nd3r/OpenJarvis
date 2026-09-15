import { useCallback, useEffect, useRef, useState } from 'react';
import { synthesizeSpeech, transcribeAudio } from '../lib/api';
import { matchWakeWord, toSpeakableText } from '../lib/voice-text';

export type VoiceStatus =
  | 'off'
  | 'starting'
  | 'listening'
  | 'attention'
  | 'transcribing'
  | 'thinking'
  | 'speaking'
  | 'error';

interface Options {
  enabled: boolean;
  wakeWord: string;
  busy: boolean;
  onCommand: (text: string) => Promise<void> | void;
}

// Energy-based segmentation. Whisper transcribes each segment; the wake word
// filters out everything not addressed to the assistant (including Whisper's
// habit of hallucinating "Thank you." on silence).
const POLL_MS = 50;
const SILENCE_MS = 900;
const MIN_SPEECH_MS = 350;
const MAX_SEGMENT_MS = 20_000;
const ATTENTION_MS = 8_000;
const NOISE_FLOOR_ALPHA = 0.02;
const SPEECH_FLOOR_MULTIPLIER = 2.5;
const SPEECH_MIN_RMS = 0.012;

const pickMimeType = (): string => {
  if (typeof MediaRecorder === 'undefined') return '';
  for (const t of ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/ogg']) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
};

const extFor = (mime: string) => (mime.includes('ogg') ? 'ogg' : 'webm');

function speakWithBrowser(text: string): Promise<void> {
  return new Promise((resolve) => {
    const synth = window.speechSynthesis;
    if (!synth) return resolve();
    const u = new SpeechSynthesisUtterance(text);
    const voice = synth.getVoices().find((v) => v.lang.toLowerCase().startsWith('en'));
    if (voice) u.voice = voice;
    // Some platforms (Firefox on Linux without speech-dispatcher) accept the
    // utterance but never fire end/error; don't let that keep the mic muted.
    const guard = window.setTimeout(resolve, 1500 + text.length * 80);
    const done = () => {
      window.clearTimeout(guard);
      resolve();
    };
    u.onend = done;
    u.onerror = done;
    synth.cancel();
    synth.speak(u);
  });
}

function playBlob(blob: Blob): Promise<void> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    const done = () => {
      URL.revokeObjectURL(url);
      resolve();
    };
    audio.onended = done;
    audio.onerror = done;
    audio.play().catch(done);
  });
}

export function useVoiceMode({ enabled, wakeWord, busy, onCommand }: Options) {
  const [status, setStatus] = useState<VoiceStatus>('off');
  const [lastHeard, setLastHeard] = useState('');
  const [error, setError] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const pollRef = useRef<number | null>(null);
  const mimeRef = useRef('');
  const activeRef = useRef(false);
  const mutedRef = useRef(false);
  const busyRef = useRef(busy);
  const attentionUntilRef = useRef(0);
  const wakeWordRef = useRef(wakeWord);
  const onCommandRef = useRef(onCommand);
  const noiseFloorRef = useRef(0.005);
  const segmentRef = useRef({ hadSpeech: false, speechStart: 0, lastVoice: 0, start: 0 });

  busyRef.current = busy;
  wakeWordRef.current = wakeWord;
  onCommandRef.current = onCommand;

  const resetSegment = () => {
    segmentRef.current = { hadSpeech: false, speechStart: 0, lastVoice: 0, start: Date.now() };
  };

  const startRecorder = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || !activeRef.current) return;
    const rec = mimeRef.current
      ? new MediaRecorder(stream, { mimeType: mimeRef.current })
      : new MediaRecorder(stream);
    chunksRef.current = [];
    rec.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    rec.start();
    recorderRef.current = rec;
    resetSegment();
  }, []);

  // Cut the current segment; returns its audio and whether it held speech.
  const cutSegment = useCallback((): Promise<{ blob: Blob; speechMs: number } | null> => {
    return new Promise((resolve) => {
      const rec = recorderRef.current;
      const seg = segmentRef.current;
      if (!rec || rec.state !== 'recording') return resolve(null);
      const speechMs = seg.hadSpeech ? seg.lastVoice - seg.speechStart : 0;
      rec.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || mimeRef.current });
        chunksRef.current = [];
        resolve({ blob, speechMs });
      };
      rec.stop();
    });
  }, []);

  const speak = useCallback(async (markdown: string) => {
    const text = toSpeakableText(markdown);
    if (!text) return;
    mutedRef.current = true;
    setStatus('speaking');
    try {
      let blob: Blob | null = null;
      try {
        blob = await synthesizeSpeech(text);
      } catch {
        blob = null;
      }
      if (blob) await playBlob(blob);
      else await speakWithBrowser(text);
    } finally {
      mutedRef.current = false;
      if (activeRef.current) {
        // Drop whatever the mic picked up while we were talking.
        await cutSegment();
        startRecorder();
        setStatus('listening');
      }
    }
  }, [cutSegment, startRecorder]);

  const handleTranscript = useCallback(async (text: string) => {
    const heard = text.trim();
    if (!heard) return;
    setLastHeard(heard);
    const { matched, command } = matchWakeWord(heard, wakeWordRef.current);
    const attentive = Date.now() < attentionUntilRef.current;

    let toSend = '';
    if (matched && command) toSend = command;
    else if (matched) {
      attentionUntilRef.current = Date.now() + ATTENTION_MS;
      setStatus('attention');
      await speak('Yeah?');
      attentionUntilRef.current = Date.now() + ATTENTION_MS;
      setStatus('attention');
      return;
    } else if (attentive) toSend = heard;

    if (!toSend) return;
    attentionUntilRef.current = 0;
    setStatus('thinking');
    try {
      await onCommandRef.current(toSend);
    } finally {
      if (activeRef.current && !mutedRef.current) setStatus('listening');
    }
  }, [speak]);

  const poll = useCallback(async () => {
    const analyser = analyserRef.current;
    if (!analyser || !activeRef.current) return;
    const buf = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);

    const now = Date.now();
    const seg = segmentRef.current;
    const threshold = Math.max(SPEECH_MIN_RMS, noiseFloorRef.current * SPEECH_FLOOR_MULTIPLIER);
    const speaking = rms > threshold;

    if (!speaking) {
      noiseFloorRef.current = noiseFloorRef.current * (1 - NOISE_FLOOR_ALPHA) + rms * NOISE_FLOOR_ALPHA;
    }

    if (mutedRef.current || busyRef.current) {
      // Keep the recorder rolling but never treat this audio as input.
      if (now - seg.start > MAX_SEGMENT_MS) {
        await cutSegment();
        startRecorder();
      }
      return;
    }

    if (speaking) {
      if (!seg.hadSpeech) {
        seg.hadSpeech = true;
        seg.speechStart = now;
      }
      seg.lastVoice = now;
    }

    const silentLongEnough = seg.hadSpeech && now - seg.lastVoice > SILENCE_MS;
    const tooLong = now - seg.start > MAX_SEGMENT_MS;
    if (silentLongEnough || tooLong) {
      const cut = await cutSegment();
      startRecorder();
      if (!cut || cut.speechMs < MIN_SPEECH_MS) return;
      setStatus('transcribing');
      try {
        const result = await transcribeAudio(cut.blob, `voice.${extFor(mimeRef.current)}`);
        await handleTranscript(result.text);
      } catch (e: any) {
        setError(e?.message || 'Transcription failed');
      } finally {
        if (activeRef.current && !mutedRef.current) {
          setStatus((s) => (s === 'transcribing' ? 'listening' : s));
        }
      }
    }
  }, [cutSegment, startRecorder, handleTranscript]);

  const stop = useCallback(() => {
    activeRef.current = false;
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    const rec = recorderRef.current;
    if (rec && rec.state !== 'inactive') {
      rec.onstop = null;
      rec.stop();
    }
    recorderRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    ctxRef.current?.close().catch(() => {});
    ctxRef.current = null;
    analyserRef.current = null;
    window.speechSynthesis?.cancel();
    attentionUntilRef.current = 0;
    setStatus('off');
  }, []);

  const start = useCallback(async () => {
    if (activeRef.current) return;
    setError(null);
    setStatus('starting');
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone not supported in this browser');
      setStatus('error');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      streamRef.current = stream;
      const ctx = new AudioContext();
      ctxRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;
      mimeRef.current = pickMimeType();
      activeRef.current = true;
      startRecorder();
      let inFlight = false;
      pollRef.current = window.setInterval(() => {
        if (inFlight) return;
        inFlight = true;
        void poll().finally(() => {
          inFlight = false;
        });
      }, POLL_MS);
      setStatus('listening');
    } catch {
      setError('Microphone access denied');
      setStatus('error');
      stop();
    }
  }, [poll, startRecorder, stop]);

  useEffect(() => {
    if (enabled) void start();
    else stop();
    return stop;
  }, [enabled, start, stop]);

  return { status, lastHeard, error, speak };
}
