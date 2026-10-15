import { Cpu } from 'lucide-react'

const MODELS = ['LFM2.5-Audio-1.5B', 'LFM2.5-Audio-1.5B-JP', 'LFM2.5-1.2B-JP']

export function Footer() {
  return (
    <footer className="mx-auto mt-12 max-w-7xl px-4 pb-10 sm:px-6 lg:px-8">
      <div className="glass flex flex-col items-center justify-between gap-4 rounded-2xl px-5 py-5 sm:flex-row">
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <Cpu className="h-4 w-4 text-brand-300" />
          <span>
            Built on <span className="font-semibold text-slate-700 dark:text-slate-200">Liquid AI LFM2.5</span> · runs
            fully on-device
          </span>
        </div>
        <div className="flex flex-wrap items-center justify-center gap-2">
          {MODELS.map((m) => (
            <span
              key={m}
              className="tnum rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] font-medium text-slate-300"
            >
              {m}
            </span>
          ))}
        </div>
      </div>
      <p className="mt-4 text-center text-[11px] text-slate-500">
        会議 Kaigi — confidential JP↔EN meetings, nothing leaves the room. APPI-friendly by design.
      </p>
    </footer>
  )
}
