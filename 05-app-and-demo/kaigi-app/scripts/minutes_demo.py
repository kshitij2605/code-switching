#!/usr/bin/env python
"""Generate bilingual minutes from a (code-switched) meeting transcript.

  pip install -e ".[mt]"   # transformers
  python scripts/minutes_demo.py                 # built-in sample transcripts
  python scripts/minutes_demo.py --transcript meeting.txt
  python scripts/minutes_demo.py --mock          # print the prompt only (no model)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLES = {
    "Sprint (eng)": """田中: では、今週のsprintのstatusを確認しましょう。Smithさん、authのfeatureはどうですか。
Smith: ほぼdoneです。ただ、login周りでbugが一つ残っていて、root causeはまだ特定できていません。明日までにfixする予定です。
佐藤: そのbug、priorityはhighですか。
Smith: はい、releaseをblockするので、highです。
田中: では、Smithさんが明日中にそのbugをfixして、佐藤さんがreviewをお願いします。
佐藤: 了解しました。あと、QAのcoverageが今70%なので、来週までに85%まで上げたいです。
田中: いいですね。それはアクションアイテムにしましょう。担当は佐藤さんで。
Smith: deploymentのscheduleですが、来週の金曜でいいですか。
田中: はい、金曜のreleaseで決定です。それまでにdocumentationもupdateしておいてください。それは私が担当します。""",
    "Client (sales)": """鈴木: 今日はABC社とのmeetingのfollow upです。先方はenterpriseプランに興味があるみたいです。
Johnson: ただ、pricingがちょっとtightだとfeedbackをもらいました。15%のdiscountを提案できないか、financeに確認したいです。
鈴木: わかりました。Johnsonさんがfinanceに確認して、来週水曜までにproposalをupdateしてください。
Johnson: 了解です。あと、技術的なrequirementとして、SSO integrationが必須だそうです。
鈴木: それはengineeringに相談ですね。私がengineering teamにSSOのfeasibilityを確認します。
Johnson: contractのdraftはいつ出しますか。
鈴木: proposalがfinalizeしたら、legalにcontractをお願いします。targetは今月末です。""",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-JP")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--transcript", help="path to a transcript file (default: built-in samples)")
    ap.add_argument("--mock", action="store_true", help="print the prompt only, no model")
    args = ap.parse_args()

    if args.transcript:
        items = [(Path(args.transcript).name, Path(args.transcript).read_text(encoding="utf-8"))]
    else:
        items = list(SAMPLES.items())

    if args.mock:
        from csmeeting.minutes.prompt import build_messages
        for name, tr in items:
            print("=" * 72, "\n", name)
            for m in build_messages(tr):
                print(f"\n[{m['role']}]\n{m['content']}")
        return

    from csmeeting.minutes.generate import MinutesGenerator
    gen = MinutesGenerator(model_id=args.model, device=args.device)
    for name, tr in items:
        print("=" * 72)
        print("TRANSCRIPT:", name)
        print("-" * 72)
        print(gen.generate(tr))
        print()


if __name__ == "__main__":
    main()
