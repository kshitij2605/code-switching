import { useRef, useState, type MouseEvent, type ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface PanelProps {
  children: ReactNode
  className?: string
  /** subtle accent border tint, e.g. for language-coded panels */
  accent?: 'ja' | 'en' | 'brand' | 'none'
  /** cursor-following spotlight glow (React Bits "Spotlight Card") */
  spotlight?: boolean
}

const accentRing: Record<NonNullable<PanelProps['accent']>, string> = {
  ja: 'before:bg-[#d8c8c0]',
  en: 'before:bg-[#c0cdd8]',
  brand: 'before:bg-brand-300',
  none: '',
}

/** The core frosted surface used across the console, with an optional
 *  cursor-following teal spotlight for premium depth (Spotlight Card).
 *  The spotlight sits behind content (negative z within an isolated context),
 *  so panel flex layouts are untouched. */
export function Panel({ children, className, accent = 'none', spotlight = true }: PanelProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ x: 0, y: 0 })
  const [opacity, setOpacity] = useState(0)

  const onMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()
    setPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
  }

  return (
    <div
      ref={ref}
      onMouseMove={spotlight ? onMove : undefined}
      onMouseEnter={spotlight ? () => setOpacity(1) : undefined}
      onMouseLeave={spotlight ? () => setOpacity(0) : undefined}
      className={cn(
        'glass relative isolate overflow-hidden rounded-2xl',
        accent !== 'none' && [
          "before:absolute before:inset-x-0 before:top-0 before:z-[2] before:h-px before:opacity-60 before:content-['']",
          accentRing[accent],
        ],
        className,
      )}
    >
      {spotlight && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 transition-opacity duration-500"
          style={{
            zIndex: -1,
            opacity,
            background: `radial-gradient(420px circle at ${pos.x}px ${pos.y}px, rgba(255,255,255,0.08), transparent 70%)`,
          }}
        />
      )}
      {children}
    </div>
  )
}
