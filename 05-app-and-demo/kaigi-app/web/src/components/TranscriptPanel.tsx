import { Check, Copy } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { cn } from '../lib/cn'
import { LANG_SHORT, type Lang } from '../lib/types'
import { CodeSwitchText } from './CodeSwitchText'
import { Panel } from './ui/Panel'

export interface Line {
  id: string
  text: string
  lang: Lang
}

interface TranscriptPanelProps {
  title: string
  subtitle: string
  icon: ReactNode
  accent: 'ja' | 'en' | 'brand'
  lines: Line[]
  partial?: { text: string; lang: Lang } | null
  streaming?: boolean
  emptyText: string
}

export function TranscriptPanel({
  title,
  subtitle,
  icon,
  accent,
  lines,
  partial,
  streaming,
  emptyText,
}: TranscriptPanelProps) {
  const [copied, setCopied] = useState(false)
  const allText = [...lines.map((l) => l.text), partial?.text].filter(Boolean).join('\n')
  const isEmpty = lines.length === 0 && !partial?.text

  const copy = async () => {
    if (!allText) return
    try {
      await navigator.clipboard.writeText(allText)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {
      /* clipboard blocked */
    }
  }

  return (
    <Panel accent={accent} className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-white/8 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span
            className="grid h-7 w-7 place-items-center rounded-lg"
            style={{
              color: accent === 'ja' ? 'var(--color-ja)' : accent === 'en' ? 'var(--color-en)' : 'var(--color-brand-300)',
              background: 'rgba(255,255,255,0.07)',
            }}
          >
            {icon}
          </span>
          <div className="leading-tight">
            <h2 className="text-sm font-bold tracking-tight text-slate-900 dark:text-white">{title}</h2>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">{subtitle}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={copy}
          disabled={!allText}
          aria-label="Copy text"
          className="grid h-8 w-8 place-items-center rounded-lg text-slate-400 transition-colors hover:bg-white/5 hover:text-white disabled:opacity-30"
        >
          {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
        </button>
      </div>

      <div className="min-h-[180px] flex-1 space-y-3 overflow-y-auto p-4 sm:max-h-[42vh]">
        {isEmpty ? (
          <div className="flex h-full min-h-[140px] items-center justify-center">
            <p className="max-w-[24ch] text-center text-sm text-slate-500">{emptyText}</p>
          </div>
        ) : (
          <>
            {lines.map((l) => (
              <LineRow key={l.id} line={l} />
            ))}
            {partial?.text ? (
              <LineRow line={{ id: 'partial', text: partial.text, lang: partial.lang }} streaming={streaming} />
            ) : null}
          </>
        )}
      </div>
    </Panel>
  )
}

function LineRow({ line, streaming }: { line: Line; streaming?: boolean }) {
  return (
    <div className="group flex gap-2.5">
      <span
        className="mt-0.5 shrink-0 rounded-md px-1.5 py-0.5 text-[10px] font-bold tracking-wider"
        style={{
          color: line.lang === 'ja' ? 'var(--color-ja)' : 'var(--color-en)',
          background: 'rgba(255,255,255,0.06)',
        }}
      >
        {LANG_SHORT[line.lang]}
      </span>
      <p
        className={cn(
          'text-[15px] leading-relaxed text-slate-800 dark:text-slate-100',
          line.lang === 'ja' && 'font-jp',
        )}
      >
        <CodeSwitchText text={line.text} lang={line.lang} />
        {streaming && (
          <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-brand-400 align-middle" />
        )}
      </p>
    </div>
  )
}
