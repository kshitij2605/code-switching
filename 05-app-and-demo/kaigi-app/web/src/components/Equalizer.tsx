import { cn } from '../lib/cn'

/** Pure-CSS equalizer bars for the "speaking" state. */
export function Equalizer({ bars = 5, className }: { bars?: number; className?: string }) {
  return (
    <div className={cn('flex items-end gap-[3px]', className)} aria-hidden="true">
      {Array.from({ length: bars }).map((_, i) => (
        <span
          key={i}
          className="w-[3px] rounded-full bg-current"
          style={{
            height: '100%',
            animation: 'eq 0.9s ease-in-out infinite',
            animationDelay: `${(i % bars) * 0.12}s`,
            transformOrigin: 'bottom',
          }}
        />
      ))}
      <style>{`@keyframes eq { 0%,100% { transform: scaleY(0.3) } 50% { transform: scaleY(1) } }`}</style>
    </div>
  )
}
