import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { SAMPLE_MINUTES, SAMPLE_TURNS } from './samples'
import type { BackendStatus, Direction, Lang, Phase, Utterance } from './types'
import { useMicrophone } from './useMicrophone'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

let _id = 0
const uid = () => `u${++_id}`

/** Split text into stream-able tokens (word-wise for EN, char-wise for JA). */
function tokenize(text: string, lang: Lang): string[] {
  if (lang === 'ja') return Array.from(text)
  return text.split(/(\s+)/).filter(Boolean)
}

export interface Partial {
  src: Lang
  tgt: Lang
  srcText: string
  tgtText: string
}

export function useAssistant() {
  const [direction, setDirection] = useState<Direction>('auto')
  const [phase, setPhase] = useState<Phase>('idle')
  const [utterances, setUtterances] = useState<Utterance[]>([])
  const [partial, setPartial] = useState<Partial | null>(null)
  const [minutes, setMinutes] = useState<string>('')
  const [minutesLoading, setMinutesLoading] = useState(false)
  const [status, setStatus] = useState<BackendStatus>({ mode: 'sim', device: 'simulated', ready: true })

  const mic = useMicrophone()
  const runIdRef = useRef(0)
  const turnRef = useRef(0)
  const startedAtRef = useRef(0)
  const playCtxRef = useRef<AudioContext | null>(null)
  const playAtRef = useRef(0)

  // Probe the FastAPI bridge; upgrade to live mode if it is up.
  useEffect(() => {
    let alive = true
    const ctrl = new AbortController()
    const t = setTimeout(() => ctrl.abort(), 1200)
    fetch('/api/status', { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((j: { device?: string; ready?: boolean }) => {
        if (alive) setStatus({ mode: 'live', device: j.device ?? 'cuda', ready: !!j.ready })
      })
      .catch(() => {})
      .finally(() => clearTimeout(t))
    return () => {
      alive = false
      ctrl.abort()
    }
  }, [])

  const nextTurn = useCallback(() => {
    const pool =
      direction === 'auto'
        ? SAMPLE_TURNS
        : SAMPLE_TURNS.filter((t) => (direction === 'ja2en' ? t.src === 'ja' : t.src === 'en'))
    const list = pool.length ? pool : SAMPLE_TURNS
    const turn = list[turnRef.current % list.length]
    turnRef.current++
    return turn
  }, [direction])

  /** Simulated capture→translate→speak cycle. */
  const runSim = useCallback(async () => {
    const myRun = ++runIdRef.current
    const turn = nextTurn()
    const src = turn.src
    const tgt: Lang = src === 'ja' ? 'en' : 'ja'

    setPartial({ src, tgt, srcText: '', tgtText: '' })

    // 1) Transcribe (source streams in)
    setPhase('transcribing')
    let acc = ''
    for (const tok of tokenize(turn.srcText, src)) {
      if (runIdRef.current !== myRun) return
      acc += tok
      setPartial((p) => (p ? { ...p, srcText: acc } : p))
      await sleep(src === 'ja' ? 34 : 26)
    }
    await sleep(160)

    // 2) Translate (target streams in)
    if (runIdRef.current !== myRun) return
    setPhase('translating')
    let tacc = ''
    for (const tok of tokenize(turn.tgtText, tgt)) {
      if (runIdRef.current !== myRun) return
      tacc += tok
      setPartial((p) => (p ? { ...p, tgtText: tacc } : p))
      await sleep(tgt === 'ja' ? 34 : 26)
    }

    // 3) Speak (spoken output plays)
    if (runIdRef.current !== myRun) return
    setPhase('speaking')
    const speakMs = Math.min(2600, 700 + turn.tgtText.length * 45)
    await sleep(speakMs)
    if (runIdRef.current !== myRun) return

    // 4) Finalize — realistic on-device latency (matches the ~0.9 s measured figure)
    const latencyMs = 820 + Math.round(Math.random() * 180)
    setUtterances((prev) => [
      ...prev,
      { id: uid(), src, tgt, srcText: turn.srcText, tgtText: turn.tgtText, ts: Date.now(), latencyMs },
    ])
    setPartial(null)
    setPhase('idle')
  }, [nextTurn])

  /** Play one base64 int16 PCM chunk from the live backend, gapless. */
  const playChunk = useCallback((b64: string, sr: number) => {
    const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    const ctx = playCtxRef.current ?? new Ctx()
    playCtxRef.current = ctx
    const bin = atob(b64)
    const i16 = new Int16Array(bin.length / 2)
    for (let i = 0; i < i16.length; i++) i16[i] = (bin.charCodeAt(i * 2) | (bin.charCodeAt(i * 2 + 1) << 8)) << 16 >> 16
    const buf = ctx.createBuffer(1, i16.length, sr)
    const ch = buf.getChannelData(0)
    for (let i = 0; i < i16.length; i++) ch[i] = i16[i] / 32768
    const node = ctx.createBufferSource()
    node.buffer = buf
    node.connect(ctx.destination)
    const now = ctx.currentTime
    const at = Math.max(now, playAtRef.current)
    node.start(at)
    playAtRef.current = at + buf.duration
  }, [])

  /** Live cascade via the FastAPI bridge; falls back to sim on any failure. */
  const runLive = useCallback(
    (blob: Blob) =>
      new Promise<void>((resolve) => {
        const myRun = ++runIdRef.current
        const t0 = startedAtRef.current || performance.now()
        playAtRef.current = 0
        let src: Lang = 'ja'
        let tgt: Lang = 'en'
        let srcText = ''
        let tgtText = ''
        setPhase('transcribing')
        setPartial(null)

        const proto = location.protocol === 'https:' ? 'wss' : 'ws'
        const ws = new WebSocket(`${proto}://${location.host}/ws/translate`)
        ws.binaryType = 'arraybuffer'

        const finalize = (latencyMs: number) => {
          if (runIdRef.current !== myRun) return
          if (srcText) {
            setUtterances((prev) => [
              ...prev,
              { id: uid(), src, tgt, srcText, tgtText, ts: Date.now(), latencyMs },
            ])
          }
          setPartial(null)
          setPhase('idle')
        }

        ws.onopen = async () => {
          ws.send(JSON.stringify({ direction }))
          ws.send(await blob.arrayBuffer())
        }
        ws.onmessage = (ev) => {
          if (runIdRef.current !== myRun) {
            ws.close()
            return
          }
          const m = JSON.parse(ev.data as string)
          switch (m.type) {
            case 'detected':
              src = m.src
              tgt = m.tgt
              setPartial({ src, tgt, srcText: '', tgtText: '' })
              break
            case 'transcript':
              srcText = m.text
              setPartial((p) => ({ src, tgt, srcText, tgtText: p?.tgtText ?? '' }))
              break
            case 'translation':
              tgtText = m.text
              setPhase('translating')
              setPartial({ src, tgt, srcText, tgtText })
              break
            case 'audio':
              setPhase('speaking')
              playChunk(m.data, m.sr)
              break
            case 'done':
              finalize(m.latencyMs ?? Math.round(performance.now() - t0))
              ws.close()
              resolve()
              break
            case 'error':
              finalize(Math.round(performance.now() - t0))
              ws.close()
              resolve()
              break
          }
        }
        ws.onerror = () => {
          // Backend unreachable mid-stream → graceful fallback.
          if (runIdRef.current === myRun) runSim().then(resolve)
          else resolve()
        }
      }),
    [direction, playChunk, runSim],
  )

  const startRecording = useCallback(async () => {
    if (phase !== 'idle') return
    startedAtRef.current = performance.now()
    setPhase('recording')
    await mic.start()
  }, [mic, phase])

  const stopRecording = useCallback(async () => {
    if (phase !== 'recording') return
    const wav = await mic.stop()
    // Live backend (real Assistant) when connected + we captured audio; else sim.
    if (status.mode === 'live' && wav) {
      await runLive(wav)
    } else {
      await runSim()
    }
  }, [mic, phase, runLive, runSim, status.mode])

  const cancel = useCallback(async () => {
    runIdRef.current++
    await mic.stop()
    setPartial(null)
    setPhase('idle')
  }, [mic])

  const generateMinutes = useCallback(async () => {
    if (!utterances.length || minutesLoading) return
    setMinutesLoading(true)
    if (status.mode === 'live') {
      try {
        const r = await fetch('/api/minutes', { method: 'POST' })
        const j = (await r.json()) as { markdown?: string }
        setMinutes(j.markdown || SAMPLE_MINUTES)
      } catch {
        setMinutes(SAMPLE_MINUTES)
      }
    } else {
      await sleep(1900)
      setMinutes(SAMPLE_MINUTES)
    }
    setMinutesLoading(false)
  }, [utterances.length, minutesLoading, status.mode])

  const reset = useCallback(async () => {
    runIdRef.current++
    await mic.stop()
    if (status.mode === 'live') void fetch('/api/reset', { method: 'POST' }).catch(() => {})
    turnRef.current = 0
    setUtterances([])
    setPartial(null)
    setMinutes('')
    setPhase('idle')
  }, [mic, status.mode])

  const stats = useMemo(() => {
    const n = utterances.length
    const avg = n ? Math.round(utterances.reduce((s, u) => s + u.latencyMs, 0) / n) : 0
    const words = utterances.reduce((s, u) => s + u.srcText.split(/\s+/).length, 0)
    return { count: n, avgLatency: avg, words }
  }, [utterances])

  return {
    direction,
    setDirection,
    phase,
    utterances,
    partial,
    minutes,
    minutesLoading,
    status,
    stats,
    levelRef: mic.levelRef,
    micAvailable: mic.available,
    startRecording,
    stopRecording,
    cancel,
    generateMinutes,
    reset,
  }
}

export type AssistantApi = ReturnType<typeof useAssistant>
