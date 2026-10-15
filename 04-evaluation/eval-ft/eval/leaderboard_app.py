#!/usr/bin/env python3
"""Streamlit UI: EN-JP code-switching ASR evaluation leaderboard.

Run from this directory (04-evaluation/eval-ft/):

  CKPT_ROOT=/path/to/checkpoints CUDA_VISIBLE_DEVICES=0 \
    streamlit run eval/leaderboard_app.py

CKPT_ROOT (default "checkpoints") is where fine-tuned LoRA adapters live; set
CUDA_VISIBLE_DEVICES to the GPU to evaluate on.

View the benchmark leaderboard (CS + JA-only + EN-only) and evaluate a new LFM
LoRA checkpoint on the frozen benchmark, adding it to the board. Evaluation uses
the STANDARD prompt "Perform ASR." by default (override for legacy checkpoints).
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leaderboard as L  # noqa: E402

st.set_page_config(page_title="CS-ASR Leaderboard", page_icon="🏆", layout="wide")

# Seed reference rows on first run; backfill newer metrics (JA-WER, decomposition).
_ents = L.seed_leaderboard()
if L.needs_recompute(_ents):
    L.recompute_all()

st.title("🏆 EN-JP Code-Switching ASR — Evaluation Leaderboard")
st.caption(
    "Frozen 3-subset benchmark · **Code-switching** (CS-FLEURS, ScriptAcc/MER) · "
    "**Japanese-only** & **English-only** (FLEURS, matched forgetting control). "
    "Higher ScriptAcc is better; lower MER / JA-CER / EN-WER is better."
)

# ---- metric column spec: (key, subset, label, higher_is_better) ----
COLS = [
    ("script_acc", "cs", "CS·ScriptAcc↑", True),
    ("mer", "cs", "CS·MER↓", False),
    ("en_wer", "cs", "CS·EnWER↓", False),
    ("ja_cer", "ja", "JA·CER↓", False),
    ("ja_wer", "ja", "JA·WER↓", False),
    ("en_wer", "en", "EN·WER↓", False),
    ("script_acc", "en", "EN·ScriptAcc↑", True),
]


def to_dataframe(entries: list[dict]) -> pd.DataFrame:
    rows = []
    for e in entries:
        row = {"Model": e["name"], "Prompt": e.get("prompt", "—")}
        for mkey, sub, label, _ in COLS:
            row[label] = e.get(sub, {}).get(mkey)
        row["N"] = "/".join(str(e.get(s, {}).get("n", "?")) for s in ("cs", "ja", "en"))
        row["When"] = (e.get("ts") or "reference")
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== #
# Leaderboard
# =========================================================================== #
entries = L.load_leaderboard()
df = to_dataframe(entries)

sort_opts = [label for _, _, label, _ in COLS]
c1, c2 = st.columns([3, 1])
with c2:
    sort_by = st.selectbox("Sort by", sort_opts, index=0)
asc = not [hib for _, _, label, hib in COLS if label == sort_by][0]
if not df.empty:
    df = df.sort_values(sort_by, ascending=asc, na_position="last").reset_index(drop=True)

# highlight best (max for ↑, min for ↓) per metric column
def _highlight(col: pd.Series):
    label = col.name
    spec = [(hib) for _, _, lb, hib in COLS if lb == label]
    if not spec:
        return [""] * len(col)
    hib = spec[0]
    vals = pd.to_numeric(col, errors="coerce")
    best = vals.max() if hib else vals.min()
    return ["background-color: #1b5e20; color: white; font-weight: bold"
            if v == best else "" for v in vals]

with c1:
    st.subheader("Leaderboard")
if df.empty:
    st.info("No entries yet — evaluate a checkpoint below.")
else:
    styler = df.style.apply(_highlight, subset=sort_opts).format(
        {lb: "{:.1f}" for _, _, lb, _ in COLS}, na_rep="—")
    st.dataframe(styler, width="stretch", hide_index=True)
    st.caption("Green = best in column. CS = code-switching · JA/EN = monolingual forgetting controls. "
               "JA·CER char-level, JA·WER word-level (fugashi). ⚠️ step_1000 LoRA is legacy "
               "(trained on 'Transcribe the audio.').")

    # ---- error decomposition drill-down ----
    with st.expander("📊 Error decomposition — substitutions / deletions / insertions per metric"):
        by_name = {e["name"]: e for e in entries}
        pick = st.selectbox("Model", list(by_name.keys()),
                            index=len(by_name) - 1, key="decomp_model")
        e = by_name[pick]
        st.caption("**sub** = heard but wrong (mishearing) · **ins** = extra/hallucinated · "
                   "**del** = dropped · rate = (sub+del+ins)/ref")
        METR = [("mer", "MER"), ("ja_cer", "JA-CER"), ("ja_wer", "JA-WER"), ("en_wer", "EN-WER")]
        for sub, label in [("cs", "Code-switching"), ("ja", "Japanese-only"), ("en", "English-only")]:
            d = e.get(sub, {})
            edits = d.get("edits")
            if not edits:
                continue
            rows = []
            for mk, mlabel in METR:
                c = edits.get(mk)
                if c and c["ref"]:
                    rows.append({"metric": mlabel,
                                 "rate %": round(100 * (c["sub"] + c["del"] + c["ins"]) / c["ref"], 1),
                                 "sub": c["sub"], "del": c["del"], "ins": c["ins"], "ref": c["ref"]})
            if rows:
                st.markdown(f"**{label}**  (n={d.get('n','?')})")
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                if sub == "cs" and d.get("script_bd"):
                    bd = d["script_bd"]
                    st.caption(f"EN-word script fate (of {bd.get('en_ref','?')}): latin **{bd.get('latin')}** · "
                               f"katakana {bd.get('katakana')} · other-JP {bd.get('other_jp')} · dropped {bd.get('dropped')}")

st.divider()

# =========================================================================== #
# Evaluate a new checkpoint
# =========================================================================== #
st.subheader("➕ Evaluate a checkpoint")

ckpts = L.list_lora_checkpoints()
if not ckpts:
    st.warning(f"No LoRA checkpoints (adapter_config.json) found under {L.CKPT_ROOT}")
else:
    names = [L.ckpt_display_name(c) for c in ckpts]
    cc1, cc2 = st.columns(2)
    with cc1:
        sel = st.selectbox(f"Checkpoint  ·  {len(ckpts)} found under {L.CKPT_ROOT}", names,
                           index=len(names) - 1)
        adapter_path = ckpts[names.index(sel)]
        default_name = f"LFM LoRA · {sel}"
        entry_name = st.text_input("Leaderboard name", value=default_name)
    with cc2:
        prompt = st.text_input("System prompt", value=L.DEFAULT_PROMPT,
                               help="STANDARD is 'Perform ASR.' Override ONLY for legacy "
                                    "checkpoints trained on a different prompt "
                                    "(e.g. lfm_lora/* used 'Transcribe the audio.').")
        quick = st.checkbox("Quick test (20 utts/subset)", value=False,
                            help="Fast sanity check; not a real benchmark number.")
        st.caption("Runs on GPU 2 · ~3–10 min full (CS+JA+EN), ~1 min quick.")

    if st.button("🚀 Evaluate & add to leaderboard", type="primary"):
        limit = 20 if quick else None
        log_lines: list[str] = []
        with st.status(f"Evaluating **{sel}** with prompt `{prompt}`…", expanded=True) as status:
            log_box = st.empty()
            stage_names = {k: v["label"] for k, v in L.BENCH.items()}

            def progress(stage, line):
                log_lines.append(f"[{stage_names.get(stage, stage)}] {line}")
                log_box.code("\n".join(log_lines[-22:]))

            try:
                t0 = time.time()
                res = L.evaluate_lfm_checkpoint(adapter_path, prompt=prompt, limit=limit,
                                                progress=progress)
                entry = {
                    "name": entry_name, "kind": "lfm-lora", "checkpoint": adapter_path,
                    "prompt": prompt, "legacy": prompt != L.DEFAULT_PROMPT,
                    "ts": time.strftime("%Y-%m-%d %H:%M"), **res,
                }
                L.upsert_entry(entry)
                status.update(label=f"✅ Done in {time.time()-t0:.0f}s — added "
                                    f"**{entry_name}**" + (" (quick)" if quick else ""),
                              state="complete", expanded=False)
                cols = st.columns(4)
                cols[0].metric("CS ScriptAcc ↑", f"{res['cs']['script_acc']:.1f}")
                cols[1].metric("CS MER ↓", f"{res['cs']['mer']:.1f}")
                cols[2].metric("JA-CER ↓", f"{res['ja']['ja_cer']:.1f}")
                cols[3].metric("EN-WER ↓", f"{res['en']['en_wer']:.1f}")
                st.button("🔄 Refresh leaderboard")  # triggers rerun
            except Exception as e:  # noqa: BLE001
                status.update(label="❌ Evaluation failed", state="error")
                st.error(str(e))
