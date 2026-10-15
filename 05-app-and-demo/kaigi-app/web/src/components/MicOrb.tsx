import { Loader2, Mic, Square } from 'lucide-react'
import { cn } from '../lib/cn'
import type { Phase } from '../lib/types'

interface MicOrbProps {
  phase: Phase
  onStart: () => void
  onStop: () => void
}

export function MicOrb({ phase, onStart, onStop }: MicOrbProps) {
  const recording = phase === 'recording'
  const busy = phase === 'transcribing' || phase === 'translating' || phase === 'speaking'

  const handle = () => {
    if (recording) onStop()
    else if (phase === 'idle') onStart()
  }

  return (
    <div className="relative grid place-items-center">
      {/* pulse rings while recording */}
      {recording && (
        <>
          <span className="pointer-events-none absolute h-20 w-20 rounded-full bg-rose-500/30 [animation:var(--animate-pulse-ring)]" />
          <span
            className="pointer-events-none absolute h-20 w-20 rounded-full bg-rose-500/20 [animation:var(--animate-pulse-ring)]"
            style={{ animationDelay: '0.8s' }}
          />
        </>
      )}

      <button
        type="button"
        onClick={handle}
        disabled={busy}
        aria-label={recording ? 'Stop recording' : 'Start recording'}
        aria-pressed={recording}
        className={cn(
          'relative grid h-20 w-20 place-items-center rounded-full text-white transition-all duration-300',
          'focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-brand-400',
          'disabled:cursor-not-allowed',
          recording
            ? 'scale-105 bg-gradient-to-br from-rose-500 to-rose-600 shadow-[0_0_40px_-4px_rgba(244,63,94,0.6)]'
            : busy
              ? 'bg-gradient-to-br from-slate-600 to-slate-700'
              : 'bg-gradient-to-br from-brand-300 to-brand-500 shadow-glow hover:scale-105 active:scale-95',
        )}
      >
        {recording ? (
          <Square className="h-7 w-7 fill-current" />
        ) : busy ? (
          <Loader2 className="h-8 w-8 animate-spin" />
        ) : (
          <Mic className="h-8 w-8" strokeWidth={2} />
        )}
      </button>
    </div>
  )
}
