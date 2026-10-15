import { Moon, ShieldCheck, Sun } from 'lucide-react'
import { cn } from '../lib/cn'
import type { BackendStatus } from '../lib/types'

interface TopBarProps {
  status: BackendStatus
  theme: 'dark' | 'light'
  onToggleTheme: () => void
}

export function TopBar({ status, theme, onToggleTheme }: TopBarProps) {
  return (
    <header className="sticky top-0 z-40">
      <div className="glass border-x-0 border-t-0 border-b border-white/8">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-3 px-4 sm:px-6 lg:px-8">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="relative grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-brand-300 to-brand-500 shadow-glow">
              <span className="font-jp text-lg font-bold text-slate-950">会</span>
            </div>
            <div className="leading-tight">
              <div className="flex items-center gap-2">
                <span className="font-jp text-base font-bold tracking-tight text-slate-900 dark:text-white">
                  会議 <span className="font-sans">Kaigi</span>
                </span>
              </div>
              <span className="hidden text-[11px] font-medium text-slate-500 sm:block dark:text-slate-400">
                On-device meeting assistant
              </span>
            </div>
          </div>

          {/* Right cluster */}
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="hidden items-center gap-1.5 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1.5 text-emerald-300 md:flex dark:text-emerald-300">
              <ShieldCheck className="h-3.5 w-3.5" strokeWidth={2.25} />
              <span className="text-[11px] font-semibold tracking-wide">100% on-device · 0 bytes to cloud</span>
            </div>

            <StatusPill status={status} />

            <button
              type="button"
              onClick={onToggleTheme}
              aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 bg-white/5 text-slate-300 transition-colors hover:bg-white/10 hover:text-white dark:text-slate-300"
            >
              {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4 text-slate-700" />}
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}

function StatusPill({ status }: { status: BackendStatus }) {
  const live = status.mode === 'live'
  return (
    <div
      className={cn(
        'flex items-center gap-1.5 rounded-full border px-2.5 py-1.5 text-[11px] font-semibold',
        live
          ? 'border-brand-400/30 bg-brand-400/10 text-brand-200'
          : 'border-amber-400/30 bg-amber-400/10 text-amber-200',
      )}
      title={live ? `Live backend · ${status.device}` : 'Demo mode — no backend connected'}
    >
      <span className={cn('h-1.5 w-1.5 rounded-full', live ? 'bg-brand-400' : 'bg-amber-400')} />
      <span className="tracking-wide uppercase">{live ? `Live · ${status.device}` : 'Demo'}</span>
    </div>
  )
}
