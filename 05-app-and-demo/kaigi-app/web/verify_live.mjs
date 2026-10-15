import { chromium } from 'playwright'
import { mkdirSync, writeFileSync } from 'node:fs'

const OUT = '.verify'
mkdirSync(OUT, { recursive: true })

// Generate a 2s 440Hz 16-bit PCM mono WAV for Chromium's fake mic.
function wav(sec = 2, sr = 16000) {
  const n = sec * sr
  const buf = Buffer.alloc(44 + n * 2)
  buf.write('RIFF', 0)
  buf.writeUInt32LE(36 + n * 2, 4)
  buf.write('WAVE', 8)
  buf.write('fmt ', 12)
  buf.writeUInt32LE(16, 16)
  buf.writeUInt16LE(1, 20)
  buf.writeUInt16LE(1, 22)
  buf.writeUInt32LE(sr, 24)
  buf.writeUInt32LE(sr * 2, 28)
  buf.writeUInt16LE(2, 32)
  buf.writeUInt16LE(16, 34)
  buf.write('data', 36)
  buf.writeUInt32LE(n * 2, 40)
  for (let i = 0; i < n; i++) buf.writeInt16LE(Math.round(Math.sin((2 * Math.PI * 440 * i) / sr) * 8000), 44 + i * 2)
  return buf
}
const wavPath = `${OUT}/fake.wav`
writeFileSync(wavPath, wav())

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const logs = []

const browser = await chromium.launch({
  args: [
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${process.cwd()}\\${wavPath}`,
  ],
})
const ctx = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
  permissions: ['microphone'],
})
const page = await ctx.newPage()
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`))
page.on('pageerror', (e) => logs.push(`PAGEERROR ${e.message}`))
page.on('websocket', (ws) => logs.push(`WS OPEN ${ws.url()}`))

await page.goto('http://localhost:5174/', { waitUntil: 'networkidle' })
await sleep(1500) // allow /api/status probe to flip to live

// Confirm the status pill shows LIVE
const statusText = await page.locator('header').innerText()
logs.push(`HEADER: ${statusText.replace(/\n/g, ' | ')}`)

await page.getByRole('button', { name: 'Start recording' }).click()
await sleep(900)
await page.getByRole('button', { name: 'Stop recording' }).click()
await sleep(3000) // transcript + translation + audio round-trip

await page.screenshot({ path: `${OUT}/live-result.png`, fullPage: true })

const transcript = await page.locator('text=Transcript').locator('xpath=ancestor::div[contains(@class,"glass")]').first().innerText()
logs.push(`TRANSCRIPT PANEL: ${transcript.replace(/\n/g, ' | ')}`)

await browser.close()
console.log(logs.join('\n'))
