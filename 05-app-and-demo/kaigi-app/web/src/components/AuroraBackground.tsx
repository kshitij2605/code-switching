import { Threads } from './Threads'

// Whitish-grey / silver threads on black (Apple-minimal). Stable reference so the
// WebGL effect initializes once, not on every render.
const THREAD_COLOR: [number, number, number] = [0.9, 0.92, 0.97]

// Full-screen threads, masked to fade out toward the top so they read as
// concentrated at the bottom of the screen while staying clearly visible.
const BOTTOM_FADE = 'linear-gradient(to bottom, transparent 0%, transparent 28%, rgba(0,0,0,0.5) 58%, #000 100%)'

/** Page backdrop — React Bits "Threads" line field in silver on black, weighted
 *  to the bottom of the screen. Sits behind all content (pointer-events: none). */
export function AuroraBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
      <Threads
        className="absolute inset-0 h-full w-full"
        style={{ WebkitMaskImage: BOTTOM_FADE, maskImage: BOTTOM_FADE }}
        color={THREAD_COLOR}
        amplitude={1}
        distance={1}
        enableMouseInteraction
      />
    </div>
  )
}
