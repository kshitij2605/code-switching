import gsap from 'gsap'
import { useEffect, useRef } from 'react'
import { useReducedMotion } from '../lib/useReducedMotion'

interface CountUpProps {
  value: number
  className?: string
  suffix?: string
}

/** Animates a number toward `value` with GSAP (instant under reduced-motion). */
export function CountUp({ value, className, suffix = '' }: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const prev = useRef(0)
  const reduced = useReducedMotion()

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (reduced) {
      el.textContent = `${Math.round(value)}${suffix}`
      prev.current = value
      return
    }
    const obj = { v: prev.current }
    const tween = gsap.to(obj, {
      v: value,
      duration: 0.7,
      ease: 'power2.out',
      onUpdate: () => {
        el.textContent = `${Math.round(obj.v)}${suffix}`
      },
    })
    prev.current = value
    return () => {
      tween.kill()
    }
  }, [value, suffix, reduced])

  return <span ref={ref} className={className}>{`0${suffix}`}</span>
}
