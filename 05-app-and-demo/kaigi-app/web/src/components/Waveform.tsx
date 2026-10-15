import { useEffect, useRef, type RefObject } from 'react'
import { useReducedMotion } from '../lib/useReducedMotion'

interface WaveformProps {
  active: boolean
  /** live amplitude 0..1 from the mic (0 when unavailable → synthetic motion) */
  levelRef: RefObject<number>
  className?: string
}

const BAR_COUNT = 48

/** Mirrored bar waveform. Driven by real mic amplitude when present, otherwise
 *  a smooth synthetic wave so the demo looks alive without a microphone. */
export function Waveform({ active, levelRef, className }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const reduced = useReducedMotion()
  const activeRef = useRef(active)
  activeRef.current = active

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf = 0
    let t = 0
    const heights = new Array(BAR_COUNT).fill(0)

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const { width, height } = canvas.getBoundingClientRect()
      canvas.width = Math.max(1, Math.floor(width * dpr))
      canvas.height = Math.max(1, Math.floor(height * dpr))
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)

    const draw = () => {
      const { width, height } = canvas.getBoundingClientRect()
      ctx.clearRect(0, 0, width, height)
      const mid = height / 2
      const gap = 3
      const barW = (width - gap * (BAR_COUNT - 1)) / BAR_COUNT
      // Skip while the element has no usable width (transient layout/resize).
      if (barW <= 0 || height <= 0) {
        if (!reduced) raf = requestAnimationFrame(draw)
        return
      }
      const live = Math.max(0, Math.min(1, levelRef.current || 0))

      t += 0.08
      for (let i = 0; i < BAR_COUNT; i++) {
        let target: number
        if (!activeRef.current) {
          target = 0.04
        } else if (live > 0.01) {
          // real mic: amplitude with a per-bar shape so it reads as a waveform
          const shape = 0.4 + 0.6 * Math.abs(Math.sin(i * 0.5 + t))
          target = Math.min(1, live * shape * 1.4) + 0.04
        } else {
          // synthetic: layered sines centered in the middle of the strip
          const center = 1 - Math.abs(i - BAR_COUNT / 2) / (BAR_COUNT / 2)
          const s = (Math.sin(i * 0.6 + t) + Math.sin(i * 0.27 - t * 1.3)) / 2
          target = (0.25 + 0.55 * Math.abs(s)) * (0.45 + 0.55 * center)
        }
        // ease toward target for fluidity
        heights[i] += (target - heights[i]) * (reduced ? 1 : 0.25)
        const h = Math.max(2, heights[i] * (height * 0.46))
        const x = i * (barW + gap)

        const grad = ctx.createLinearGradient(0, mid - h, 0, mid + h)
        grad.addColorStop(0, 'rgba(255, 255, 255, 0.95)')
        grad.addColorStop(0.5, 'rgba(209, 213, 219, 0.9)')
        grad.addColorStop(1, 'rgba(148, 153, 163, 0.85)')
        ctx.fillStyle = grad
        roundRect(ctx, x, mid - h, barW, h * 2, barW / 2)
        ctx.fill()
      }

      if (!reduced) raf = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [levelRef, reduced])

  return <canvas ref={canvasRef} className={className} aria-hidden="true" />
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  if (w <= 0 || h <= 0) return
  const rr = Math.max(0, Math.min(r, w / 2, h / 2))
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}
