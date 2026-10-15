import { FileText, Lock, Sparkles } from 'lucide-react'
import type { ReactNode } from 'react'
import type { AssistantApi } from '../lib/useAssistant'
import { StarBorderButton } from './StarBorder'
import { Panel } from './ui/Panel'

export function MinutesPanel({ api }: { api: AssistantApi }) {
  const { minutes, minutesLoading, stats } = api
  const canGenerate = stats.count > 0 && !minutesLoading

  return (
    <Panel accent="brand" className="flex flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/8 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-lg bg-brand-400/12 text-brand-300">
            <FileText className="h-4 w-4" />
          </span>
          <div className="leading-tight">
            <h2 className="flex items-center gap-1.5 text-sm font-bold tracking-tight text-slate-900 dark:text-white">
              Minutes
              <Lock className="h-3 w-3 text-emerald-400" />
            </h2>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">Summary · decisions · action items</p>
          </div>
        </div>
        <StarBorderButton onClick={api.generateMinutes} disabled={!canGenerate}>
          <Sparkles className="h-4 w-4" />
          {minutes ? 'Regenerate' : 'Generate minutes'}
        </StarBorderButton>
      </div>

      <div className="p-4 sm:p-5">
        {minutesLoading ? (
          <Skeleton />
        ) : minutes ? (
          <Markdown source={minutes} />
        ) : (
          <div className="flex flex-col items-center justify-center gap-2 py-10 text-center">
            <p className="max-w-[40ch] text-sm text-slate-500">
              {stats.count > 0
                ? 'Generate a confidential summary with decisions and owner-tagged action items — all on-device.'
                : 'Capture some conversation, then generate minutes here. Nothing leaves the device.'}
            </p>
          </div>
        )}
      </div>
    </Panel>
  )
}

function Skeleton() {
  return (
    <div className="space-y-3">
      {[90, 70, 80, 55, 75].map((w, i) => (
        <div
          key={i}
          className="h-3.5 rounded-full bg-gradient-to-r from-white/5 via-white/10 to-white/5 bg-[length:200%_100%]"
          style={{ width: `${w}%`, animation: 'shimmer 1.6s linear infinite', animationDelay: `${i * 0.1}s` }}
        />
      ))}
      <style>{`@keyframes shimmer { 0% { background-position: 200% 0 } 100% { background-position: -200% 0 } }`}</style>
    </div>
  )
}

/** Minimal markdown renderer: ## headings, - bullets, **bold**, JP-aware. */
function Markdown({ source }: { source: string }) {
  const blocks: ReactNode[] = []
  const lines = source.split('\n')
  let list: string[] = []

  const flush = () => {
    if (!list.length) return
    blocks.push(
      <ul key={`ul${blocks.length}`} className="my-2 space-y-1.5">
        {list.map((item, i) => (
          <li key={i} className="flex gap-2 text-[14px] leading-relaxed text-slate-700 dark:text-slate-200">
            <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-brand-400" />
            <span className="font-jp">{inline(item)}</span>
          </li>
        ))}
      </ul>,
    )
    list = []
  }

  for (const raw of lines) {
    const line = raw.trimEnd()
    if (line.startsWith('## ')) {
      flush()
      blocks.push(
        <h3
          key={`h${blocks.length}`}
          className="font-jp mt-4 mb-1 text-[13px] font-bold tracking-wide text-brand-300 uppercase first:mt-0"
        >
          {line.slice(3)}
        </h3>,
      )
    } else if (line.startsWith('- ')) {
      list.push(line.slice(2))
    } else if (line) {
      flush()
      blocks.push(
        <p key={`p${blocks.length}`} className="my-1.5 text-[14px] text-slate-700 dark:text-slate-200">
          {inline(line)}
        </p>,
      )
    }
  }
  flush()
  return <div>{blocks}</div>
}

function inline(text: string): ReactNode {
  return text.split(/(\*\*[^*]+\*\*)/g).map((seg, i) =>
    seg.startsWith('**') && seg.endsWith('**') ? (
      <strong key={i} className="font-bold text-slate-900 dark:text-white">
        {seg.slice(2, -2)}
      </strong>
    ) : (
      seg
    ),
  )
}
