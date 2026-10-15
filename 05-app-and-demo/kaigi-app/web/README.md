# 会議 Kaigi — Web UI

A modern, responsive React front end for the on-device JP↔EN meeting assistant.
Live translation console (animated mic + waveform), dual code-switch-aware
transcript/translation panels, spoken-output indicator, and confidential
auto-minutes — dark-first with a light theme, fluid GSAP motion, and a fully
responsive layout from mobile to laptop.

**Stack:** React 19 · Vite · TypeScript · Tailwind v4 · GSAP · ogl (React Bits
"Threads" animated background) · lucide-react · self-hosted fonts (Plus Jakarta
Sans / Noto Sans JP / JetBrains Mono — no CDN calls, fitting an on-device privacy
product).

The page backdrop is the React Bits **Threads** line field (WebGL/OGL), tuned to
the brand teal. It pauses under `prefers-reduced-motion` and renders a single
static frame on software renderers (SwiftShader/headless) so it can never starve
the main thread on weak hardware.

## Run

```bash
cd web
npm install
npm run dev        # http://localhost:5174
```

By default the UI runs in **Demo mode** — a fully simulated cascade with
realistic code-switch sample turns. No backend required; everything (mic button,
waveform, streaming text, minutes) works standalone.

### Connect the real on-device models (Live mode)

Start the FastAPI bridge (it wraps the Python `Assistant`), then reload the UI —
it auto-detects the backend and switches the header pill to **Live**:

```bash
# from the repo root
pip install -e ".[web]"          # + ".[live]" (GPU) or ".[cpu]" for the models
python scripts/serve_api.py --mock          # wiring only, no models
python scripts/serve_api.py                 # real models (GPU)
python scripts/serve_api.py --asr-model <your-finetuned-cs-asr>
```

The Vite dev server proxies `/api` and `/ws` to `http://localhost:8000`.
The browser captures mic audio, encodes it to 16-bit PCM WAV, and streams it to
the bridge over a WebSocket; transcript, translation, and spoken audio stream
back. If the backend is unreachable, the UI falls back to Demo mode.

## Layout

```
src/
  App.tsx                 # shell + GSAP entrance stagger
  index.css               # Tailwind v4 theme tokens (teal/slate, dark-first) + light overrides
  components/             # TopBar, Hero, Console, MicOrb, Waveform, DirectionControl,
                          # DualPanels, TranscriptPanel, CodeSwitchText, SpokenOutput,
                          # Equalizer, MinutesPanel, StatStrip, CountUp, Footer,
                          # AuroraBackground, Threads (React Bits WebGL backdrop),
                          # ShinyText, StarBorder (React Bits); Panel has a React
                          # Bits "Spotlight Card" cursor glow built in
  lib/
    useAssistant.ts       # orchestration: simulated cascade + live WebSocket client
    useMicrophone.ts      # mic capture, amplitude metering, WAV encode
    useTheme.ts, useReducedMotion.ts, types.ts, samples.ts, cn.ts
```

## Verify (optional)

`verify.mjs` / `verify_live.mjs` drive the app with Playwright (screenshots at
375 / 768 / 1440, the full interaction flow, light mode, and the live path with a
fake mic). They need `npm i -D playwright`. Output lands in `.verify/`.

## Build

```bash
npm run build      # tsc + vite build → dist/
```
