"""Machine-translation stage: JA->EN prompt, eval harness (decide if fine-tuning is needed), and SFT data prep."""

from csmeeting.mt.prompt import SYSTEM_PROMPT, build_messages  # noqa: F401
