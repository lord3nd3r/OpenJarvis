import { describe, expect, it } from 'vitest';
import { matchWakeWord, toSpeakableText } from './voice-text';

describe('matchWakeWord', () => {
  it('matches the wake word at the start and returns the command', () => {
    expect(matchWakeWord('Yo Jarvis, what time is it?', 'jarvis')).toEqual({
      matched: true,
      command: "what time is it",
    });
  });

  it('matches the wake word mid-sentence', () => {
    expect(matchWakeWord('hey there jarvis turn it up', 'jarvis').command).toBe(
      'turn it up',
    );
  });

  it('returns an empty command when only the wake word was said', () => {
    expect(matchWakeWord('Jarvis.', 'jarvis')).toEqual({ matched: true, command: '' });
  });

  it('accepts common mis-hearings of jarvis', () => {
    expect(matchWakeWord('Travis, sup', 'jarvis').matched).toBe(true);
  });

  it('ignores speech without the wake word', () => {
    expect(matchWakeWord('Thank you.', 'jarvis')).toEqual({ matched: false, command: '' });
  });

  it('does not match partial words', () => {
    expect(matchWakeWord('the jarvisation', 'jarvis').matched).toBe(false);
  });

  it('supports a custom wake word', () => {
    expect(matchWakeWord('ok computer play music', 'computer').command).toBe('play music');
  });
});

describe('toSpeakableText', () => {
  it('drops citations, links, code, and markdown decorations', () => {
    const md = [
      '# Today',
      'It is **sunny** [[1]](https://x.ai) in _Denver_ ([source](https://a.b)).',
      '```py',
      'print(1)',
      '```',
      '- one',
      '- two',
    ].join('\n');
    expect(toSpeakableText(md)).toBe('Today\nIt is sunny in Denver (source).\none\ntwo');
  });

  it('caps very long replies', () => {
    const long = Array(400).fill('This is a sentence.').join(' ');
    expect(toSpeakableText(long).length).toBeLessThanOrEqual(2000);
  });
});
