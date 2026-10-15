import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const OUT = '.verify'
mkdirSync(OUT, { recursive: true })
const URL = 'http://localhost:5174/'
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const errors = []

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
const page = await ctx.newPage()
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
page.on('pageerror', (e) => errors.push(`PAGEERROR ${e.message}`))

// --- Flow first, on a fresh page (avoids viewport churn before interaction) ---
await page.goto(URL, { waitUntil: 'networkidle' })
await sleep(800)
await page.getByRole('button', { name: 'Start recording' }).click()
await sleep(700)
await page.screenshot({ path: `${OUT}/flow-1-recording.png` })
await page.getByRole('button', { name: 'Stop recording' }).click()
await sleep(900)
await page.screenshot({ path: `${OUT}/flow-2-streaming.png` })

for (let i = 0; i < 3; i++) {
  await page.getByRole('button', { name: 'Start recording' }).click()
  await sleep(400)
  await page.getByRole('button', { name: 'Stop recording' }).click()
  await sleep(2600)
}
await page.screenshot({ path: `${OUT}/flow-3-filled.png`, fullPage: true })

await page.getByRole('button', { name: /Generate minutes/i }).click()
await sleep(2300)
await page.screenshot({ path: `${OUT}/flow-4-minutes.png`, fullPage: true })

await page.getByRole('button', { name: /Switch to light mode/i }).click()
await sleep(600)
await page.screenshot({ path: `${OUT}/light-mode.png`, fullPage: true })

// --- Responsive idle screenshots ---
const sizes = [
  { name: 'desktop', w: 1440, h: 900 },
  { name: 'tablet', w: 768, h: 1024 },
  { name: 'mobile', w: 375, h: 812 },
]
for (const s of sizes) {
  await page.setViewportSize({ width: s.w, height: s.h })
  await page.goto(URL, { waitUntil: 'networkidle' })
  await sleep(1300)
  await page.screenshot({ path: `${OUT}/${s.name}-idle.png`, fullPage: true })
}

await browser.close()
console.log(errors.length ? 'CONSOLE/PAGE ERRORS:\n' + errors.join('\n') : 'No console/page errors.')
console.log('Screenshots →', OUT)
