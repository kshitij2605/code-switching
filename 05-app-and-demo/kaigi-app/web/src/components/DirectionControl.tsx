import { Sparkles } from 'lucide-react'
import { cn } from '../lib/cn'
import { DIRECTION_LABELS, type Direction } from '../lib/types'

interface DirectionControlProps {
  value: Direction
  onChange: (d: Direction) => void
  disabled?: boolean
}

const ORDER: Direction[] = ['auto', 'ja2en', 'en2ja']

export function DirectionControl({ value, onChange, disabled }: DirectionControlProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Translation direction"
      className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 p-1 backdrop-blur"
    >
      {ORDER.map((d) => {
        const selected = value === d
        return (
          <button
            key={d}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={disabled}
            onClick={() => onChange(d)}
            className={cn(
              'relative flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[13px] font-semibold transition-all duration-200 sm:px-4',
              'disabled:cursor-not-allowed disabled:opacity-50',
              selected
                ? 'bg-gradient-to-br from-brand-300 to-brand-500 text-slate-950 shadow-glow'
                : 'text-slate-400 hover:text-white',
            )}
          >
            {d === 'auto' && <Sparkles className="h-3.5 w-3.5" strokeWidth={2.25} />}
            {DIRECTION_LABELS[d]}
          </button>
        )
      })}
    </div>
  )
}
