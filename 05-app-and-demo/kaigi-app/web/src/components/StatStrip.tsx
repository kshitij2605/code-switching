import { Clock, MessageSquareText, Type } from 'lucide-react'
import type { ReactNode } from 'react'
import type { AssistantApi } from '../lib/useAssistant'
import { CountUp } from './CountUp'

export function StatStrip({ api }: { api: AssistantApi }) {
  const { stats } = api
  return (
    <div className="grid grid-cols-3 gap-3">
      <Stat icon={<MessageSquareText className="h-4 w-4" />} label="Utterances">
        <CountUp value={stats.count} />
      </Stat>
      <Stat icon={<Clock className="h-4 w-4" />} label="Avg latency">
        <CountUp value={stats.avgLatency || 900} suffix=" ms" />
      </Stat>
      <Stat icon={<Type className="h-4 w-4" />} label="Words">
        <CountUp value={stats.words} />
      </Stat>
    </div>
  )
}

function Stat({ icon, label, children }: { icon: ReactNode; label: string; children: ReactNode }) {
  return (
    <div className="glass flex flex-col gap-1 rounded-xl px-3 py-3 sm:px-4">
      <div className="flex items-center gap-1.5 text-brand-300">{icon}</div>
      <div className="tnum text-xl font-bold text-slate-900 sm:text-2xl dark:text-white">{children}</div>
      <div className="text-[11px] font-medium tracking-wide text-slate-500">{label}</div>
    </div>
  )
}
