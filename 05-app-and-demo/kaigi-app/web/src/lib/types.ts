export type Direction = 'auto' | 'ja2en' | 'en2ja'
export type Lang = 'ja' | 'en'

/** Lifecycle of one capture→translate→speak cycle. */
export type Phase = 'idle' | 'recording' | 'transcribing' | 'translating' | 'speaking'

export interface Utterance {
  id: string
  src: Lang
  tgt: Lang
  srcText: string
  tgtText: string
  /** ms since epoch when the utterance was finalized */
  ts: number
  /** end-to-end latency in ms (sim or measured) */
  latencyMs: number
}

export interface BackendStatus {
  mode: 'sim' | 'live'
  device: string
  ready: boolean
}

export const DIRECTION_LABELS: Record<Direction, string> = {
  auto: 'Auto',
  ja2en: 'JA → EN',
  en2ja: 'EN → JA',
}

export const LANG_LABEL: Record<Lang, string> = {
  ja: '日本語',
  en: 'English',
}

export const LANG_SHORT: Record<Lang, string> = {
  ja: 'JA',
  en: 'EN',
}
