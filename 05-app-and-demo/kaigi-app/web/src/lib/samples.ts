import type { Lang } from './types'

/**
 * Realistic bilingual meeting lines used by the simulated backend.
 * The JA lines deliberately embed English in Latin script (code-switch) — the
 * exact phenomenon Kaigi's fine-tuned ASR is built to transcribe correctly.
 */
export interface SamplePair {
  src: Lang
  srcText: string
  tgtText: string
}

export const SAMPLE_TURNS: SamplePair[] = [
  {
    src: 'ja',
    srcText: '来週の sprint review までに、API のデプロイを完了させましょう。',
    tgtText: "Let's finish the API deployment before next week's sprint review.",
  },
  {
    src: 'en',
    srcText: 'Can you share the quarterly revenue forecast before the board meeting?',
    tgtText: '取締役会の前に、四半期の売上予測を共有してもらえますか？',
  },
  {
    src: 'ja',
    srcText: 'この KPI については、マーケティングチームと sync する必要があります。',
    tgtText: 'We need to sync with the marketing team about this KPI.',
  },
  {
    src: 'en',
    srcText: 'I think we should prioritize the onboarding flow for the next release.',
    tgtText: '次のリリースでは、オンボーディングフローを優先すべきだと思います。',
  },
  {
    src: 'ja',
    srcText: 'コンプライアンス上、この議事録は cloud にアップロードできません。',
    tgtText: 'For compliance reasons, these minutes cannot be uploaded to the cloud.',
  },
  {
    src: 'en',
    srcText: "Let's assign the security audit to Tanaka-san and review it on Friday.",
    tgtText: 'セキュリティ監査を田中さんに割り当てて、金曜日にレビューしましょう。',
  },
]

/** Demo minutes returned by the simulated backend (markdown). */
export const SAMPLE_MINUTES = `## 要約 / Summary
- The team aligned on shipping the API deployment ahead of next week's sprint review.
- Quarterly revenue forecast to be circulated before the board meeting.
- All meeting records stay on-device for APPI compliance — nothing leaves the room.

## 決定事項 / Decisions
- Prioritize the onboarding flow for the next release.
- Keep minutes fully on-device (no cloud upload).

## アクションアイテム / Action Items
- **[担当 / Owner: 田中]** Lead the security audit, review Friday.
- **[担当 / Owner: Marketing]** Sync on the activation KPI before sign-off.
- **[担当 / Owner: Eng]** Complete API deployment before the sprint review.`
