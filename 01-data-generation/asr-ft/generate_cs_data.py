"""
Generate 500 Japanese-English code-switching audio samples using edge-tts.

Output per sample:
  - audio file (wav)
  - metadata with: full_transcription, all_japanese, all_english,
                   japanese_parts, english_parts, segments
"""

import asyncio
import edge_tts
import json
import random
import io
import subprocess
import wave
from pathlib import Path
import imageio_ffmpeg

_FFMPEG     = imageio_ffmpeg.get_ffmpeg_exe()
_RATE       = 24000
_CHANNELS   = 1
_SAMPWIDTH  = 2   # 16-bit

def _mp3_to_pcm(mp3_bytes: bytes) -> bytes:
    """Decode MP3 bytes → raw signed-16-bit PCM frames via ffmpeg."""
    result = subprocess.run(
        [_FFMPEG, "-v", "error", "-f", "mp3", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(_RATE), "-ac", str(_CHANNELS), "pipe:1"],
        input=mp3_bytes, capture_output=True
    )
    return result.stdout

def _pcm_to_wav(pcm_frames: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPWIDTH)
        wf.setframerate(_RATE)
        wf.writeframes(pcm_frames)
    return buf.getvalue()

def _silence(ms: int) -> bytes:
    return b"\x00" * (_RATE * ms // 1000 * _SAMPWIDTH * _CHANNELS)

AUDIO_OUT = Path("data/code_switching/audio")
META_OUT  = Path("data/code_switching/metadata")
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
META_OUT.mkdir(parents=True, exist_ok=True)

VOICE_JA   = "ja-JP-NanamiNeural"
VOICE_EN   = "en-US-JennyNeural"
SILENCE_MS = 180   # pause between segments
N_SAMPLES  = 500
PROXY      = None  # set to "http://host:port/" if behind a proxy
random.seed(42)

# ---------------------------------------------------------------------------
# Vocabulary bank
# Each EN entry: (en_text, ja_equivalent)
# Each JA entry: (ja_text, en_equivalent)
# ---------------------------------------------------------------------------

EN_VOCAB = [
    # Work / business
    ("schedule",        "スケジュール"),
    ("meeting",         "ミーティング"),
    ("deadline",        "締め切り"),
    ("project",         "プロジェクト"),
    ("presentation",    "プレゼン"),
    ("overtime",        "残業"),
    ("feedback",        "フィードバック"),
    ("report",          "レポート"),
    ("team",            "チーム"),
    ("budget",          "予算"),
    ("client",          "クライアント"),
    ("agenda",          "議題"),
    ("target",          "目標"),
    ("goal",            "ゴール"),
    ("task",            "タスク"),
    ("review",          "レビュー"),
    ("proposal",        "提案"),
    ("contract",        "契約"),
    # Tech
    ("bug",             "バグ"),
    ("fix",             "修正"),
    ("update",          "アップデート"),
    ("app",             "アプリ"),
    ("system",          "システム"),
    ("data",            "データ"),
    ("server",          "サーバー"),
    ("network",         "ネットワーク"),
    ("code",            "コード"),
    ("model",           "モデル"),
    ("training",        "トレーニング"),
    ("release",         "リリース"),
    ("backup",          "バックアップ"),
    ("error",           "エラー"),
    ("log",             "ログ"),
    ("deploy",          "デプロイ"),
    # Daily life
    ("coffee",          "コーヒー"),
    ("lunch",           "ランチ"),
    ("break",           "休憩"),
    ("party",           "パーティー"),
    ("trip",            "旅行"),
    ("shopping",        "ショッピング"),
    ("movie",           "映画"),
    ("music",           "音楽"),
    ("game",            "ゲーム"),
    ("event",           "イベント"),
    ("weekend",         "週末"),
    ("holiday",         "休日"),
    ("gym",             "ジム"),
    ("dinner",          "ディナー"),
    ("recipe",          "レシピ"),
    # Adjectives / states (EN used in JA speech)
    ("tired",           "疲れた"),
    ("excited",         "ワクワク"),
    ("nervous",         "緊張してる"),
    ("stressed",        "ストレスがたまってる"),
    ("happy",           "嬉しい"),
    ("sad",             "悲しい"),
    ("busy",            "忙しい"),
    ("free",            "暇"),
    ("interesting",     "面白い"),
    ("difficult",       "難しい"),
    ("amazing",         "すごい"),
    ("perfect",         "完璧"),
    ("serious",         "真剣"),
    ("random",          "ランダム"),
    ("basically",       "基本的に"),
    ("actually",        "実は"),
    ("totally",         "完全に"),
    ("literally",       "文字通り"),
    # Common EN phrases used in JA speech
    ("please check",    "確認してください"),
    ("no problem",      "問題ない"),
    ("good job",        "よくやった"),
    ("of course",       "もちろん"),
    ("let's go",        "行こう"),
    ("well done",       "よくできた"),
    ("I need",          "必要だ"),
    ("right now",       "今すぐ"),
    ("never mind",      "気にしないで"),
    ("sounds good",     "いいね"),
    ("take care",       "気をつけて"),
    ("good luck",       "頑張って"),
    ("my bad",          "私のせい"),
    ("fair enough",     "なるほど"),
    ("for real",        "本当に"),
    ("kind of",         "ちょっと"),
    ("a lot",           "たくさん"),
    ("not yet",         "まだ"),
    ("even so",         "それでも"),
    ("after all",       "やっぱり"),
    ("worth it",        "価値がある"),
    ("I see",           "なるほど"),
]

JA_VOCAB = [
    # Subject / topic starters
    ("明日の",          "Tomorrow's"),
    ("今日の",          "Today's"),
    ("この",            "this"),
    ("その",            "that"),
    ("先週の",          "Last week's"),
    ("来週の",          "Next week's"),
    ("最近",            "Recently"),
    ("いつも",          "Always"),
    ("全然",            "Not at all"),
    ("少し",            "A little"),
    ("ちょっと",        "Just a little"),
    ("なんか",          "Somehow"),
    ("やっぱり",        "After all"),
    ("もう",            "Already"),
    ("まだ",            "Still"),
    ("また",            "Again"),
    ("特に",            "Especially"),
    ("絶対",            "Definitely"),
    ("多分",            "Probably"),
    ("確かに",          "Certainly"),
    # Verbs / verb phrases
    ("確認して",        "please confirm"),
    ("教えてください",  "please tell me"),
    ("考えてみる",      "I'll think about it"),
    ("手伝ってくれる",  "can you help me"),
    ("やってみた",      "I tried it"),
    ("終わらせた",      "I finished"),
    ("始めましょう",    "let's start"),
    ("準備ができた",    "I'm ready"),
    ("話し合おう",      "let's discuss"),
    ("気をつけて",      "be careful"),
    ("頑張ってる",      "I'm doing my best"),
    ("諦めないで",      "don't give up"),
    ("覚えてる",        "I remember"),
    ("忘れた",          "I forgot"),
    ("わかった",        "I understood"),
    ("知らなかった",    "I didn't know"),
    ("びっくりした",    "I was surprised"),
    ("楽しみにしてる",  "I'm looking forward to it"),
    ("行ってきます",    "I'm heading out"),
    ("戻ってきた",      "I came back"),
    # Adjectives / descriptors
    ("難しい",          "difficult"),
    ("面白い",          "interesting"),
    ("大変だ",          "it's tough"),
    ("忙しい",          "I'm busy"),
    ("疲れた",          "I'm tired"),
    ("嬉しい",          "I'm happy"),
    ("すごく",          "very"),
    ("本当に",          "really"),
    ("かなり",          "quite"),
    # Sentence-final / connectors
    ("だから",          "so"),
    ("でも",            "but"),
    ("だけど",          "but"),
    ("なんだ",          "it is"),
    ("だよ",            "you know"),
    ("だね",            "right?"),
    ("かな",            "I wonder"),
    ("ね",              "right?"),
    ("よ",              "you know"),
    ("な",              "huh"),
    ("か",              "I see"),
    ("わ",              "I'm telling you"),
    ("もん",            "because"),
    ("てか",            "or rather"),
    ("しかも",          "moreover"),
    ("それに",          "besides"),
    ("だって",          "because"),
    ("じゃあ",          "well then"),
    ("えっと",          "um"),
    ("あの",            "um"),
]

# ---------------------------------------------------------------------------
# Sentence templates
# Each template is a list of (lang, slot_type) tuples.
# slot_type: "en_word", "en_phrase", "ja_start", "ja_verb", "ja_adj",
#            "ja_conn", "fixed_ja:<text>:<en_equiv>", "fixed_en:<text>:<ja_equiv>"
# ---------------------------------------------------------------------------

def make_en(text, ja_equiv):
    return {"lang": "en", "text": text, "ja_equiv": ja_equiv}

def make_ja(text, en_equiv):
    return {"lang": "ja", "text": text, "en_equiv": en_equiv}


TOPICS = [
    # ---- Work / office ----
    lambda: [
        make_ja(random.choice([("明日の","Tomorrow's"),("来週の","Next week's"),("今週の","This week's")])[0],
                random.choice([("明日の","Tomorrow's"),("来週の","Next week's"),("今週の","This week's")])[1]),
        make_en(*random.choice([("meeting","ミーティング"),("schedule","スケジュール"),("deadline","締め切り"),("presentation","プレゼン")])),
        make_ja(*random.choice([("は大丈夫","is fine"),("を確認して","please confirm"),("の準備できた","preparation is done"),("が変わった","has changed")])),
        make_en(*random.choice([("right?","そうでしょ?"),("I think","私は思う"),("for sure","確かに"),("no problem","問題ない")])),
        make_ja(*random.choice([("それに","Besides"),("それと","And also"),("しかも","Moreover"),("あと","Also")])),
        make_en(*random.choice([("the report","レポート"),("the budget","予算"),("the proposal","提案"),("the review","レビュー")])),
        make_ja(*random.choice([("も終わらせないと","also needs to be finished"),("も確認したほうがいい","should also check"),("もやっておいて","also do it")])),
        make_ja(*random.choice([("だよね","right?"),("だと思う","I think so"),("かな","I wonder"),("だね","isn't it")])),
    ],
    lambda: [
        make_ja(*random.choice([("今日は","Today"),("今","Now"),("さっき","Earlier")])),
        make_en(*random.choice([("the client","クライアント"),("the team","チーム"),("the manager","マネージャー")])),
        make_ja(*random.choice([("との","with the"),("からの","from the"),("への","to the")])),
        make_en(*random.choice([("feedback","フィードバック"),("meeting","ミーティング"),("call","電話")])),
        make_ja(*random.choice([("が来た","arrived"),("があった","there was"),("を受けた","I received")])),
        make_ja(*random.choice([("だけど","but"),("でも","but"),("ただ","however")])),
        make_en(*random.choice([("it was okay","大丈夫だった"),("it went well","うまくいった"),("not so good","あまりよくなかった"),("kind of tough","ちょっと大変だった")])),
        make_ja(*random.choice([("よ","you know"),("な","huh"),("ね","right?")])),
    ],
    # ---- Technology ----
    lambda: [
        make_ja(*random.choice([("この","This"),("その","That"),("例の","That famous")])),
        make_en(*random.choice([("bug","バグ"),("error","エラー"),("issue","問題")])),
        make_ja(*random.choice([("が","(subject marker)"),("は","(topic marker)")])),
        make_en(*random.choice([("finally fixed","やっと直した"),("hard to fix","直しにくい"),("still there","まだある"),("really annoying","本当に厄介な")])),
        make_ja(*random.choice([("だよ","you know"),("みたい","it seems"),("らしい","apparently"),("そう","so they say")])),
        make_ja(*random.choice([("次は","Next"),("それより","Rather than that"),("その後","After that")])),
        make_en(*random.choice([("the update","アップデート"),("the deploy","デプロイ"),("the release","リリース"),("the backup","バックアップ")])),
        make_ja(*random.choice([("を確認して","please confirm"),("をやる","will do"),("が心配","I'm worried about"),("を待ってる","I'm waiting for")])),
        make_ja(*random.choice([("ね","right?"),("よ","you know"),("かな","I wonder")])),
    ],
    lambda: [
        make_en(*random.choice([("the system","システム"),("the server","サーバー"),("the network","ネットワーク"),("the app","アプリ")])),
        make_ja(*random.choice([("が","(subject)"),("は","(topic)")])),
        make_en(*random.choice([("down again","また落ちてる"),("slow today","今日は遅い"),("working fine","うまく動いてる"),("acting weird","おかしい")])),
        make_ja(*random.choice([("だから","so"),("なので","therefore"),("だし","and")])),
        make_en(*random.choice([("I need to","必要がある"),("we should","すべき"),("let's","しよう")])),
        make_en(*random.choice([("check the log","ログを確認"),("restart it","再起動する"),("update it","アップデートする"),("fix it now","今すぐ直す")])),
        make_ja(*random.choice([("かな","I wonder"),("と思う","I think"),("だろうな","I guess"),("みたいだ","it seems")])),
    ],
    # ---- Daily life ----
    lambda: [
        make_ja(*random.choice([("ちょっと","Just"),("そろそろ","About time for a"),("今日は","Today")])),
        make_en(*random.choice([("break","休憩"),("coffee","コーヒー"),("lunch","ランチ")])),
        make_ja(*random.choice([("取ろうか","shall we take"),("飲もう","let's drink"),("行こうか","shall we go for")])),
        make_ja(*random.choice([("どこ行く","where shall we go"),("何食べる","what shall we eat"),("何時にする","what time")])),
        make_en(*random.choice([("any suggestions?","何かある?"),("I'm fine with anything","何でもいい"),("let's decide quickly","早く決めよう")])),
        make_ja(*random.choice([("あそこの","that place's"),("新しい","new"),("近くの","nearby")])),
        make_en(*random.choice([("place","場所"),("restaurant","レストラン"),("cafe","カフェ")])),
        make_ja(*random.choice([("はどう","how about"),("がいいかな","might be good"),("に行こう","let's go to")])),
        make_ja(*random.choice([("ね","right?"),("よ","you know"),("かな","I wonder")])),
    ],
    lambda: [
        make_ja(*random.choice([("週末は","This weekend"),("今度の休みに","On the next holiday"),("明日の夜","Tomorrow evening")])),
        make_en(*random.choice([("a party","パーティー"),("an event","イベント"),("a trip","旅行")])),
        make_ja(*random.choice([("があるんだ","there is"),("に行く予定","I'm planning to go to"),("を計画してる","I'm planning")])),
        make_ja(*random.choice([("すごく","very"),("ちょっと","a little"),("本当に","really")])),
        make_en(*random.choice([("excited","ワクワク"),("nervous","緊張してる"),("happy","嬉しい"),("looking forward","楽しみ")])),
        make_ja(*random.choice([("だよ","you know"),("なんだ","you see"),("わ","I'm telling you")])),
        make_ja(*random.choice([("一緒に","together"),("もし良かったら","if you'd like"),("よかったら","if you want")])),
        make_en(*random.choice([("join us","一緒に来て"),("come along","来てよ"),("let's go together","一緒に行こう")])),
        make_ja(*random.choice([("よ","!"),("よね","right?"),("ね","right?")])),
    ],
    # ---- Emotions / reactions ----
    lambda: [
        make_ja(*random.choice([("本当に","Really"),("すごく","Very"),("めちゃくちゃ","Extremely"),("かなり","Quite")])),
        make_en(*random.choice([("tired","疲れた"),("stressed","ストレスがたまってる"),("exhausted","ヘトヘト"),("burned out","燃え尽きた")])),
        make_ja(*random.choice([("だよ","you know"),("わ","I'm telling you"),("なんだよね","it's like")])),
        make_en(*random.choice([("I need","必要だ"),("I want","欲しい"),("gotta have","必要不可欠")])),
        make_ja(*random.choice([("少し","a little"),("もっと","more"),("ちゃんとした","proper")])),
        make_en(*random.choice([("rest","休息"),("sleep","睡眠"),("time off","休み"),("break","休憩")])),
        make_ja(*random.choice([("が必要だと思う","I think I need"),("をとらないと","I need to take"),("がほしい","I want")])),
        make_ja(*random.choice([("ね","right?"),("よ","I'm telling you"),("かな","I wonder")])),
    ],
    lambda: [
        make_ja(*random.choice([("あの話","That story"),("その件","That matter"),("例のやつ","That thing")])),
        make_en(*random.choice([("was amazing","すごかった"),("was so sad","すごく悲しかった"),("blew my mind","びっくりした"),("was unexpected","予想外だった")])),
        make_ja(*random.choice([("よね","right?"),("だよね","right?"),("ね","right?")])),
        make_ja(*random.choice([("私も","I also"),("みんなも","everyone also"),("正直","honestly")])),
        make_en(*random.choice([("didn't expect that","予想してなかった"),("was surprised","びっくりした"),("felt the same way","同じ気持ちだった")])),
        make_ja(*random.choice([("なんだよね","you know"),("んだよ","I'm telling you"),("だったよ","it was")])),
    ],
    # ---- Education ----
    lambda: [
        make_ja(*random.choice([("今日の","Today's"),("先生の","The teacher's"),("昨日の","Yesterday's")])),
        make_en(*random.choice([("lecture","授業"),("class","クラス"),("lesson","レッスン"),("seminar","セミナー")])),
        make_ja(*random.choice([("は","was"),("が","(subject)")])),
        make_en(*random.choice([("really difficult","本当に難しかった"),("so interesting","すごく面白かった"),("a bit boring","少し退屈だった"),("very helpful","とても役に立った")])),
        make_ja(*random.choice([("だったね","wasn't it?"),("だったよ","it was"),("だと思う","I think")])),
        make_ja(*random.choice([("でも","But"),("それでも","Even so"),("だから","So")])),
        make_en(*random.choice([("the homework","宿題"),("the assignment","課題"),("the exam","試験")])),
        make_ja(*random.choice([("は終わった","is done"),("がまだ","is still"),("が難しい","is difficult")])),
        make_ja(*random.choice([("よ","!"),("ね","right?"),("かな","I wonder"),("だよ","you know")])),
    ],
    # ---- Health / wellbeing ----
    lambda: [
        make_ja(*random.choice([("最近","Recently"),("ここのところ","Lately"),("ちょっと前から","For a while now")])),
        make_en(*random.choice([("really tired","本当に疲れてる"),("not feeling well","体調が悪い"),("kind of sick","ちょっと具合が悪い")])),
        make_ja(*random.choice([("んだよね","you know"),("みたい","it seems"),("らしい","apparently")])),
        make_ja(*random.choice([("だから","So"),("なので","Therefore"),("でも","But")])),
        make_en(*random.choice([("I think","思う"),("maybe","多分"),("probably","おそらく")])),
        make_ja(*random.choice([("早めに","early"),("ちゃんと","properly"),("十分に","sufficiently")])),
        make_en(*random.choice([("rest","休む"),("sleep","睡眠を取る"),("see a doctor","医者に行く"),("take medicine","薬を飲む")])),
        make_ja(*random.choice([("したほうがいいかな","should probably"),("が必要だよ","is necessary"),("しようかな","I should maybe")])),
        make_ja(*random.choice([("ね","right?"),("よ","I'm telling you"),("だと思うよ","I think")])),
    ],
    # ---- Shopping / consumer ----
    lambda: [
        make_ja(*random.choice([("この","This"),("あの","That"),("さっき見た","The one I just saw")])),
        make_en(*random.choice([("shirt","シャツ"),("bag","バッグ"),("jacket","ジャケット"),("shoes","シューズ")])),
        make_ja(*random.choice([("すごく","very"),("ちょっと","a little"),("かなり","quite")])),
        make_en(*random.choice([("good","いい"),("nice","素敵"),("cool","かっこいい"),("cute","かわいい")])),
        make_ja(*random.choice([("と思わない","don't you think?"),("じゃない","isn't it?"),("だよね","right?")])),
        make_ja(*random.choice([("値段は","The price is"),("でも","But"),("ただ","However")])),
        make_en(*random.choice([("a bit expensive","少し高い"),("reasonable","手頃"),("on sale","セール中"),("worth it","価値がある")])),
        make_ja(*random.choice([("かな","I wonder"),("だと思う","I think"),("だよ","you know")])),
    ],
]


def pick_segments():
    topic_fn = random.choice(TOPICS)
    segs = topic_fn()
    # optionally add 1-2 extra fillers
    if random.random() < 0.4:
        segs.insert(0, make_ja(*random.choice(JA_VOCAB[:20])))
    if random.random() < 0.3:
        segs.append(make_ja(*random.choice([("わかった","Understood"),("了解","Got it"),("ありがとう","Thank you"),("よろしく","Please")])))
    return segs


def build_transcriptions(segs):
    full     = " ".join(s["text"] for s in segs)
    all_ja   = " ".join(s["ja_equiv"] if s["lang"] == "en" else s["text"] for s in segs)
    all_en   = " ".join(s["en_equiv"] if s["lang"] == "ja" else s["text"] for s in segs)
    jp_parts = " ".join(s["text"] for s in segs if s["lang"] == "ja")
    en_parts = " ".join(s["text"] for s in segs if s["lang"] == "en")
    return full, all_ja, all_en, jp_parts, en_parts


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------

async def tts_bytes(text: str, voice: str) -> bytes:
    comm = edge_tts.Communicate(text, voice, proxy=PROXY)
    buf = b""
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf += chunk["data"]
    return buf


async def build_audio(segs) -> tuple:
    pcm_all = b""
    sil = _silence(SILENCE_MS)
    for seg in segs:
        voice = VOICE_JA if seg["lang"] == "ja" else VOICE_EN
        mp3   = await tts_bytes(seg["text"], voice)
        pcm_all += _mp3_to_pcm(mp3) + sil
    wav_bytes = _pcm_to_wav(pcm_all)
    duration  = len(pcm_all) / (_RATE * _SAMPWIDTH * _CHANNELS)
    return wav_bytes, duration


async def generate_sample(idx: int) -> dict:
    segs = pick_segments()
    full, all_ja, all_en, jp_parts, en_parts = build_transcriptions(segs)

    wav_bytes, duration = await build_audio(segs)
    filename = f"cs_{idx:03d}.wav"
    filepath = AUDIO_OUT / filename
    filepath.write_bytes(wav_bytes)

    return {
        "id":                f"cs_{idx:03d}",
        "audio_file":        str(filepath),
        "full_transcription": full,
        "all_japanese":       all_ja,
        "all_english":        all_en,
        "japanese_parts":     jp_parts,
        "english_parts":      en_parts,
        "duration_seconds":   round(duration, 2),
        "num_segments":       len(segs),
        "segments":           segs,
    }


async def main():
    samples = []
    for i in range(1, N_SAMPLES + 1):
        print(f"  [{i}/{N_SAMPLES}] cs_{i:03d}", flush=True)
        try:
            sample = await generate_sample(i)
            samples.append(sample)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)

    out_path = META_OUT / "transcriptions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(samples)} samples → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
