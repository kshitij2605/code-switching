import { Fragment } from 'react'
import type { Lang } from '../lib/types'

// Latin-script runs (true English switches) inside Japanese — the loanword/switch
// boundary Kaigi is tuned to preserve. We tint them so the code-switch is visible.
const LATIN_RUN = /([A-Za-z][A-Za-z0-9'._-]*(?:\s[A-Za-z][A-Za-z0-9'._-]*)*)/g

interface Props {
  text: string
  lang: Lang
  className?: string
}

export function CodeSwitchText({ text, lang, className }: Props) {
  // Only the Japanese side carries embedded English worth highlighting.
  if (lang !== 'ja') return <span className={className}>{text}</span>

  const parts = text.split(LATIN_RUN)
  return (
    <span className={className}>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <span key={i} className="font-sans font-semibold text-[#f5f5f7]">
            {part}
          </span>
        ) : (
          <Fragment key={i}>{part}</Fragment>
        ),
      )}
    </span>
  )
}
