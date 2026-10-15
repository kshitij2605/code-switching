import { Languages, ScrollText } from 'lucide-react'
import type { AssistantApi } from '../lib/useAssistant'
import { TranscriptPanel, type Line } from './TranscriptPanel'

export function DualPanels({ api }: { api: AssistantApi }) {
  const { utterances, partial, phase } = api

  const sourceLines: Line[] = utterances.map((u) => ({ id: u.id, text: u.srcText, lang: u.src }))
  const targetLines: Line[] = utterances.map((u) => ({ id: u.id, text: u.tgtText, lang: u.tgt }))

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <TranscriptPanel
        title="Transcript"
        subtitle="Source · code-switch aware"
        icon={<ScrollText className="h-4 w-4" />}
        accent="ja"
        lines={sourceLines}
        partial={partial ? { text: partial.srcText, lang: partial.src } : null}
        streaming={phase === 'transcribing'}
        emptyText="Your spoken words appear here — Japanese with embedded English preserved."
      />
      <TranscriptPanel
        title="Translation"
        subtitle="Target · spoken aloud"
        icon={<Languages className="h-4 w-4" />}
        accent="en"
        lines={targetLines}
        partial={partial ? { text: partial.tgtText, lang: partial.tgt } : null}
        streaming={phase === 'translating'}
        emptyText="The translation streams here and plays through the other side's speaker."
      />
    </div>
  )
}
