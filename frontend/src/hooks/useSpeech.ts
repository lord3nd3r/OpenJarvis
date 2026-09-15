import { useState, useCallback, useRef, useEffect } from 'react';
import { transcribeAudio, fetchSpeechHealth } from '../lib/api';

export type SpeechState = 'idle' | 'recording' | 'transcribing';

const isBrowserSpeechSupported = (): boolean => {
  return typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);
};

export function useSpeech() {
  const [state, setState] = useState<SpeechState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [available, setAvailable] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const recognitionRef = useRef<any>(null);
  const browserResultRef = useRef<string>('');

  // Check if speech backend or browser speech recognition is available on mount
  useEffect(() => {
    fetchSpeechHealth()
      .then((health) => setAvailable(health.available || isBrowserSpeechSupported()))
      .catch(() => setAvailable(isBrowserSpeechSupported()));
  }, []);

  const startRecording = useCallback(async (): Promise<void> => {
    setError(null);
    browserResultRef.current = '';

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone not supported in this browser');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;

      const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRec) {
        try {
          const rec = new SpeechRec();
          rec.continuous = true;
          rec.interimResults = false;
          rec.lang = 'en-US';
          rec.onresult = (e: any) => {
            let text = '';
            for (let i = e.resultIndex; i < e.results.length; ++i) {
              if (e.results[i].isFinal) text += e.results[i][0].transcript;
            }
            if (text.trim()) browserResultRef.current = text.trim();
          };
          rec.start();
          recognitionRef.current = rec;
        } catch {}
      }

      setState('recording');
    } catch (err) {
      setError('Microphone access denied');
      setState('idle');
    }
  }, []);

  const stopRecording = useCallback(async (): Promise<string> => {
    return new Promise((resolve, reject) => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {}
      }

      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state !== 'recording') {
        reject(new Error('Not recording'));
        return;
      }

      recorder.onstop = async () => {
        setState('transcribing');

        // Stop all audio tracks
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;

        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        chunksRef.current = [];

        try {
          const result = await transcribeAudio(blob);
          setState('idle');
          resolve(result.text);
        } catch (err) {
          setState('idle');
          if (browserResultRef.current) {
            resolve(browserResultRef.current);
          } else {
            const msg = err instanceof Error ? err.message : 'Transcription failed';
            setError(msg);
            reject(err);
          }
        }
      };

      recorder.stop();
    });
  }, []);

  return {
    state,
    error,
    available,
    startRecording,
    stopRecording,
    isRecording: state === 'recording',
    isTranscribing: state === 'transcribing',
  };
}
