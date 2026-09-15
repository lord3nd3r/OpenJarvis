// Helpers for hands-free voice mode: wake-word matching on Whisper
// transcripts and turning markdown replies into something worth reading aloud.

const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const normalize = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s']/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();

export interface WakeWordMatch {
  matched: boolean;
  command: string;
}

// Whisper often hears "Jarvis" as these; treat them as the wake word too.
const DEFAULT_ALIASES: Record<string, string[]> = {
  jarvis: ['jarvis', 'jarvas', 'jarves', 'jervis', 'travis'],
};

export function matchWakeWord(transcript: string, wakeWord: string): WakeWordMatch {
  const text = normalize(transcript);
  const wake = normalize(wakeWord);
  if (!text || !wake) return { matched: false, command: '' };

  const forms = [wake, ...(DEFAULT_ALIASES[wake] ?? [])];
  const pattern = new RegExp(`\\b(?:${forms.map(escapeRegExp).join('|')})\\b`, 'u');
  const m = pattern.exec(text);
  if (!m) return { matched: false, command: '' };

  const command = text.slice(m.index + m[0].length).trim();
  return { matched: true, command };
}

const MAX_SPOKEN_CHARS = 2000;

export function toSpeakableText(markdown: string): string {
  let t = markdown;
  t = t.replace(/```[\s\S]*?```/g, ' ');
  t = t.replace(/`([^`]*)`/g, '$1');
  t = t.replace(/!\[[^\]]*\]\([^)]*\)/g, ' ');
  t = t.replace(/\[\[\d+\]\]\([^)]*\)/g, ' ');
  t = t.replace(/\[([^\]]*)\]\([^)]*\)/g, '$1');
  t = t.replace(/<[^>]+>/g, ' ');
  t = t.replace(/^\s{0,3}#{1,6}\s+/gm, '');
  t = t.replace(/^\s*[-*+]\s+/gm, '');
  t = t.replace(/^\s*\d+\.\s+/gm, '');
  t = t.replace(/^\s*\|.*\|\s*$/gm, ' ');
  t = t.replace(/^\s*[-=_*]{3,}\s*$/gm, ' ');
  t = t.replace(/[*_~]{1,3}([^*_~]+)[*_~]{1,3}/g, '$1');
  t = t.replace(/https?:\/\/\S+/g, ' ');
  t = t.replace(/[ \t]+/g, ' ').replace(/\s*\n\s*/g, '\n').trim();
  if (t.length > MAX_SPOKEN_CHARS) {
    const cut = t.slice(0, MAX_SPOKEN_CHARS);
    t = cut.slice(0, Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('\n'), 0) || cut.length);
  }
  return t;
}
