import { ArrowRight, RotateCcw } from 'lucide-react'
import { cn } from '../lib/cn'
import { LANG_SHORT, type Phase } from '../lib/types'
import type { AssistantApi } from '../lib/useAssistant'
import { DirectionControl } from './DirectionControl'
import { MicOrb } from './MicOrb'
import { Panel } from './ui/Panel'
import { Waveform } from './Waveform'

const PHASE_TEXT: Record<Phase, string> = {
  idle: 'Ready',
  recording: 'Listening…',
  transcribing: 'Transcribing…',
  translating: 'Translating…',
  speaking: 'Speaking translation…',
}

export function Console({ api }: { api: AssistantApi }) {
  const { phase, partial, direction } = api
  const recording = phase === 'recording'
  const active = phase !== 'idle'

  return (
    <Panel accent="brand" className="p-5 sm:p-7">
      {/* top row: direction + status */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <DirectionControl value={direction} onChange={api.setDirection} disabled={active} />
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'h-2 w-2 rounded-full',
              recording ? 'bg-rose-500' : active ? 'bg-amber-400' : 'bg-slate-500',
              active && 'animate-pulse',
            )}
          />
          <span className="text-[13px] font-semibold tracking-wide text-slate-300">{PHASE_TEXT[phase]}</span>
        </div>
      </div>

      {/* mic + waveform */}
      <div className="flex flex-col items-center gap-6 sm:flex-row sm:gap-7">
        <MicOrb phase={phase} onStart={api.startRecording} onStop={api.stopRecording} />

        <div className="flex w-full flex-1 flex-col gap-3">
          <Waveform active={recording} levelRef={api.levelRef} className="h-16 w-full sm:h-20" />

          {/* detected-language pill / hint */}
          <div className="flex min-h-7 items-center justify-center gap-2 sm:justify-start">
            {partial ? (
              <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-bold tracking-wide">
                <span style={{ color: partial.src === 'ja' ? 'var(--color-ja)' : 'var(--color-en)' }}>
                  {LANG_SHORT[partial.src]}
                </span>
                <ArrowRight className="h-3 w-3 text-slate-500" />
                <span style={{ color: partial.tgt === 'ja' ? 'var(--color-ja)' : 'var(--color-en)' }}>
                  {LANG_SHORT[partial.tgt]}
                </span>
                {direction === 'auto' && <span className="ml-1 text-[10px] font-medium text-slate-500">auto-detected</span>}
              </div>
            ) : (
              <p className="text-center text-[13px] text-slate-400 sm:text-left">
                {recording ? 'Speak now — tap stop when you pause.' : 'Tap the mic and speak Japanese or English.'}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* footer actions */}
      <div className="mt-6 flex items-center justify-between border-t border-white/8 pt-4">
        <span className="text-[11px] text-slate-500">
          Pause-to-translate · streaming spoken output ≈0.9&nbsp;s
        </span>
        <button
          type="button"
          onClick={api.reset}
          disabled={active}
          className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold text-slate-400 transition-colors hover:bg-white/5 hover:text-white disabled:opacity-40"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset session
        </button>
      </div>
    </Panel>
  )
}
