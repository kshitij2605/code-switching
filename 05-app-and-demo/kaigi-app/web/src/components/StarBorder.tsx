import type { ReactNode } from 'react'
import { cn } from '../lib/cn'

interface StarBorderButtonProps {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  className?: string
  /** glow color of the traveling border */
  color?: string
  /** animation duration, e.g. "6s" */
  speed?: string
}

// React Bits "Star Border" adapted to a themed CTA button (teal glow on a deep
// inner). Stars hide when disabled; glow pauses under reduced-motion.
export function StarBorderButton({
  children,
  onClick,
  disabled = false,
  className,
  color = 'rgba(255, 255, 255, 0.85)',
  speed = '6s',
}: StarBorderButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'relative inline-block overflow-hidden rounded-xl py-[1px] transition-transform duration-200',
        disabled ? 'cursor-not-allowed opacity-60' : 'hover:scale-[1.02] active:scale-95',
        className,
      )}
    >
      {!disabled && (
        <>
          <span
            className="absolute bottom-[-11px] right-[-250%] z-0 h-1/2 w-[300%] rounded-full opacity-60"
            style={{
              background: `radial-gradient(circle, ${color}, transparent 10%)`,
              animation: `star-movement-bottom ${speed} linear infinite alternate`,
            }}
          />
          <span
            className="absolute top-[-10px] left-[-250%] z-0 h-1/2 w-[300%] rounded-full opacity-60"
            style={{
              background: `radial-gradient(circle, ${color}, transparent 10%)`,
              animation: `star-movement-top ${speed} linear infinite alternate`,
            }}
          />
        </>
      )}
      <span
        className={cn(
          'relative z-[1] flex items-center justify-center gap-1.5 rounded-xl border px-4 py-2 text-[13px] font-semibold',
          'bg-gradient-to-b from-slate-900 to-slate-950',
          disabled ? 'border-white/10 text-slate-500' : 'border-brand-400/40 text-brand-100',
        )}
      >
        {children}
      </span>
    </button>
  )
}
