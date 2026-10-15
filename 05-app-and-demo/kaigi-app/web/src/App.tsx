import { useGSAP } from '@gsap/react'
import gsap from 'gsap'
import { useRef } from 'react'
import { AuroraBackground } from './components/AuroraBackground'
import { Console } from './components/Console'
import { DualPanels } from './components/DualPanels'
import { Footer } from './components/Footer'
import { Hero } from './components/Hero'
import { MinutesPanel } from './components/MinutesPanel'
import { SpokenOutput } from './components/SpokenOutput'
import { StatStrip } from './components/StatStrip'
import { TopBar } from './components/TopBar'
import { useAssistant } from './lib/useAssistant'
import { useReducedMotion } from './lib/useReducedMotion'
import { useTheme } from './lib/useTheme'

export default function App() {
  const api = useAssistant()
  const { theme, toggle } = useTheme()
  const reduced = useReducedMotion()
  const scope = useRef<HTMLDivElement>(null)

  useGSAP(
    () => {
      if (reduced) {
        gsap.set('[data-reveal]', { opacity: 1, y: 0 })
        return
      }
      gsap.from('[data-reveal]', {
        opacity: 0,
        y: 26,
        duration: 0.7,
        ease: 'power3.out',
        stagger: 0.08,
      })
    },
    { scope, dependencies: [reduced] },
  )

  const speaking = api.phase === 'speaking' && api.partial

  return (
    <div ref={scope} className="flex min-h-dvh flex-col">
      <AuroraBackground />
      <TopBar status={api.status} theme={theme} onToggleTheme={toggle} />

      <Hero />

      <main className="mx-auto w-full max-w-6xl flex-1 space-y-4 px-4 sm:px-6 lg:px-8">
        <div data-reveal>
          <Console api={api} />
        </div>

        {speaking && api.partial ? <SpokenOutput lang={api.partial.tgt} /> : null}

        <div data-reveal>
          <StatStrip api={api} />
        </div>

        <div data-reveal>
          <DualPanels api={api} />
        </div>

        <div data-reveal>
          <MinutesPanel api={api} />
        </div>
      </main>

      <Footer />
    </div>
  )
}
