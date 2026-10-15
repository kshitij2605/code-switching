import type { CSSProperties } from 'react'

interface ShinyTextProps {
  text: string
  className?: string
  /** seconds per sheen sweep */
  speed?: number
  /** base (resting) text color */
  color?: string
  /** sheen highlight color */
  shine?: string
}

// React Bits "Shiny Text", reimplemented deps-free (CSS background-clip sheen)
// so it doesn't pull in framer-motion. Sheen pauses under reduced-motion.
export function ShinyText({
  text,
  className,
  speed = 4,
  color = 'rgba(148, 163, 184, 0.7)',
  shine = '#5eead4',
}: ShinyTextProps) {
  const style: CSSProperties = {
    backgroundImage: `linear-gradient(110deg, ${color} 0%, ${color} 40%, ${shine} 50%, ${color} 60%, ${color} 100%)`,
    backgroundSize: '200% auto',
    WebkitBackgroundClip: 'text',
    backgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    color: 'transparent',
    animation: `shiny-sheen ${speed}s linear infinite`,
  }
  return (
    <span className={className} style={style}>
      {text}
    </span>
  )
}
