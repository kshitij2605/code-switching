"""Minutes stage: (code-switched) meeting transcript -> structured bilingual minutes.

Test-first verdict (validated on host GPU): base `LFM2.5-1.2B-JP` + this prompt + a one-shot example produces
usable draft minutes with mostly-correct owner attribution. `-Thinking` over-reasons and doesn't converge -> use JP.
See docs/minutes.md.
"""

from csmeeting.minutes.prompt import SYSTEM_PROMPT, build_messages  # noqa: F401
