import { Volume2 } from 'lucide-react'
import { LANG_LABEL, type Lang } from '../lib/types'
import { Equalizer } from './Equalizer'

/** "Now speaking" banner shown while the translation is played aloud. */
export function SpokenOutput({ lang }: { lang: Lang }) {
  return (
    <div
      className="glass flex items-center gap-3 rounded-xl border-brand-400/20 px-4 py-2.5 text-brand-200"
      style={{ animation: 'spk-in 0.4s ease-out both' }}
    >
      <Volume2 className="h-4 w-4 shrink-0" />
      <span className="text-[13px] font-semibold">
        Speaking aloud · <span className={lang === 'ja' ? 'font-jp' : ''}>{LANG_LABEL[lang]}</span> voice
      </span>
      <Equalizer className="ml-auto h-4 text-brand-400" />
      <style>{`@keyframes spk-in { from { opacity: 0; transform: translateY(8px) } to { opacity: 1; transform: translateY(0) } }`}</style>
    </div>
  )
}
