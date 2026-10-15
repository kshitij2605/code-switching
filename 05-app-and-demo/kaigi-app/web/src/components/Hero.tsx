import { AudioLines, FileText, Languages } from 'lucide-react'
import type { ReactNode } from 'react'
import { ShinyText } from './ShinyText'
import { SplitText } from './SplitText'

const FEATURES: { icon: ReactNode; label: string }[] = [
  { icon: <AudioLines className="h-3.5 w-3.5" />, label: 'Code-switch transcription' },
  { icon: <Languages className="h-3.5 w-3.5" />, label: 'Bidirectional speech translation' },
  { icon: <FileText className="h-3.5 w-3.5" />, label: 'Confidential auto-minutes' },
]

export function Hero() {
  return (
    <section className="mx-auto max-w-3xl px-4 pt-12 pb-8 text-center sm:pt-16">
      <div
        data-reveal
        className="mb-5 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[12px] font-semibold"
      >
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-brand-400" />
        </span>
        <ShinyText
          text="Real-time · on-device · LFM2.5"
          speed={5}
          color="rgba(170, 174, 184, 0.6)"
          shine="#ffffff"
        />
      </div>

      <SplitText
        text="会議を、そのまま翻訳。"
        tag="h1"
        splitType="chars"
        delay={40}
        duration={1}
        from={{ opacity: 0, y: 36 }}
        to={{ opacity: 1, y: 0 }}
        className="font-jp text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl dark:text-white"
      />
      <p
        data-reveal
        className="mx-auto mt-4 max-w-xl text-base text-slate-600 sm:text-lg dark:text-slate-300"
      >
        Speak Japanese or English — the other side hears the translation in under a second. Live transcription and
        confidential minutes, <span className="font-semibold text-slate-800 dark:text-white">100% on-device.</span>
      </p>

      <div data-reveal className="mt-6 flex flex-wrap items-center justify-center gap-2">
        {FEATURES.map((f) => (
          <span
            key={f.label}
            className="inline-flex items-center gap-1.5 rounded-full border border-brand-400/20 bg-brand-400/8 px-3 py-1.5 text-[12px] font-semibold text-brand-200"
          >
            {f.icon}
            {f.label}
          </span>
        ))}
      </div>
    </section>
  )
}
