import { useCallback, useRef, useState } from 'react'

/**
 * Microphone capture + live amplitude metering.
 *
 * Degrades gracefully: if the browser denies mic access (or there is no device,
 * e.g. a headless Playwright run), start() still resolves so the simulated flow
 * can proceed — `level` simply stays at 0 and the waveform falls back to a
 * synthetic animation.
 */
export function useMicrophone() {
  const [available, setAvailable] = useState(false)
  const levelRef = useRef(0)
  const streamRef = useRef<MediaStream | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const rafRef = useRef<number | null>(null)
  const dataRef = useRef<Uint8Array | null>(null)
  // raw PCM capture for the live backend (encoded to WAV on stop)
  const procRef = useRef<ScriptProcessorNode | null>(null)
  const pcmRef = useRef<Float32Array[]>([])
  const sampleRateRef = useRef(48000)

  const tick = useCallback(() => {
    const analyser = analyserRef.current
    const buf = dataRef.current
    if (analyser && buf) {
      analyser.getByteTimeDomainData(buf as Uint8Array<ArrayBuffer>)
      let sum = 0
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128
        sum += v * v
      }
      // RMS, lightly boosted for visual presence
      levelRef.current = Math.min(1, Math.sqrt(sum / buf.length) * 2.6)
    }
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  const start = useCallback(async () => {
    pcmRef.current = []
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new Ctx()
      ctxRef.current = ctx
      if (ctx.state === 'suspended') await ctx.resume() // required for ScriptProcessor to run
      sampleRateRef.current = ctx.sampleRate
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 1024
      source.connect(analyser)
      analyserRef.current = analyser
      dataRef.current = new Uint8Array(analyser.fftSize)

      // Raw PCM tap for the live backend (WAV is built on stop).
      try {
        const proc = ctx.createScriptProcessor(4096, 1, 1)
        proc.onaudioprocess = (e) => pcmRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)))
        source.connect(proc)
        proc.connect(ctx.destination)
        procRef.current = proc
      } catch {
        /* ScriptProcessor unsupported — live capture disabled, sim still works */
      }

      setAvailable(true)
      rafRef.current = requestAnimationFrame(tick)
    } catch {
      // No mic / denied → degraded mode, simulated flow still runs.
      setAvailable(false)
    }
  }, [tick])

  const stop = useCallback(async (): Promise<Blob | null> => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    levelRef.current = 0

    const wav = pcmRef.current.length ? encodeWav(pcmRef.current, sampleRateRef.current) : null

    procRef.current?.disconnect()
    streamRef.current?.getTracks().forEach((t) => t.stop())
    ctxRef.current?.close().catch(() => {})
    procRef.current = null
    streamRef.current = null
    ctxRef.current = null
    analyserRef.current = null
    pcmRef.current = []
    return wav
  }, [])

  return { available, levelRef, start, stop }
}

/** Concatenate captured Float32 chunks into a 16-bit PCM mono WAV blob. */
function encodeWav(chunks: Float32Array[], sampleRate: number): Blob {
  let length = 0
  for (const c of chunks) length += c.length
  const pcm = new Float32Array(length)
  let off = 0
  for (const c of chunks) {
    pcm.set(c, off)
    off += c.length
  }

  const buffer = new ArrayBuffer(44 + pcm.length * 2)
  const view = new DataView(buffer)
  const writeStr = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i))
  }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + pcm.length * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(36, 'data')
  view.setUint32(40, pcm.length * 2, true)
  let p = 44
  for (let i = 0; i < pcm.length; i++, p += 2) {
    const s = Math.max(-1, Math.min(1, pcm[i]))
    view.setInt16(p, s * 0x7fff, true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}
