"""
Generate 300 English-only audio samples using edge-tts.

Each sample has:
  - audio file (wav, English voice)
  - transcription (English)
  - japanese_translation (for "transcribe in Japanese" fine-tuning target)
"""

import asyncio
import edge_tts
import json
import io
import subprocess
import wave
import random
from pathlib import Path
import imageio_ffmpeg

_FFMPEG    = imageio_ffmpeg.get_ffmpeg_exe()
_RATE      = 24000
_CHANNELS  = 1
_SAMPWIDTH = 2

def _mp3_to_wav(mp3_bytes: bytes) -> tuple:
    pcm = subprocess.run(
        [_FFMPEG, "-v", "error", "-f", "mp3", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(_RATE), "-ac", str(_CHANNELS), "pipe:1"],
        input=mp3_bytes, capture_output=True
    ).stdout
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(_SAMPWIDTH)
        wf.setframerate(_RATE)
        wf.writeframes(pcm)
    duration = len(pcm) / (_RATE * _SAMPWIDTH * _CHANNELS)
    return buf.getvalue(), duration

AUDIO_OUT = Path("data/english_only/audio")
META_OUT  = Path("data/english_only/metadata")
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
META_OUT.mkdir(parents=True, exist_ok=True)

VOICE_EN  = "en-US-JennyNeural"
N_SAMPLES = 300
PROXY     = None  # set to "http://host:port/" if behind a proxy
random.seed(123)

# ---------------------------------------------------------------------------
# Sentence pairs (en_text, ja_translation)
# 6 topics × 50 pairs = 300
# ---------------------------------------------------------------------------

WORK_PAIRS = [
    ("Let's confirm the schedule for tomorrow's meeting.", "明日の会議のスケジュールを確認しましょう。"),
    ("The deadline for this project is next Friday.", "このプロジェクトの締め切りは来週の金曜日です。"),
    ("Can you send me the report by end of day?", "今日中にレポートを送ってもらえますか？"),
    ("We need to prepare for the client presentation.", "クライアントへのプレゼンの準備が必要です。"),
    ("The budget for this quarter looks tight.", "今四半期の予算は厳しそうです。"),
    ("I have back-to-back meetings all afternoon.", "午後はずっと会議が続きます。"),
    ("Please share your feedback on the proposal.", "提案についてのフィードバックをお願いします。"),
    ("The team is working overtime this week.", "今週はチーム全員が残業しています。"),
    ("We need to finalize the contract by Monday.", "月曜日までに契約を確定させる必要があります。"),
    ("Let's review the agenda before the meeting starts.", "会議が始まる前に議題を確認しましょう。"),
    ("The client approved the new design proposal.", "クライアントが新しいデザイン提案を承認しました。"),
    ("I need to write the monthly progress report today.", "今日は月次進捗レポートを書かなければなりません。"),
    ("The project is behind schedule by two weeks.", "プロジェクトは2週間遅れています。"),
    ("Can we reschedule the team meeting to Thursday?", "チームの会議を木曜日に変更できますか？"),
    ("The new manager starts working here next Monday.", "新しいマネージャーは来週の月曜日から勤務します。"),
    ("I need to discuss the budget with the finance team.", "財務チームと予算について話し合う必要があります。"),
    ("Please review the contract before signing it.", "サインする前に契約書を確認してください。"),
    ("Our target for this year is to increase sales by twenty percent.", "今年の目標は売上を20パーセント増加させることです。"),
    ("The proposal needs to be revised before submission.", "提案書は提出前に修正が必要です。"),
    ("Let's set up a quick call to discuss the next steps.", "次のステップについて話し合うために簡単な電話をしましょう。"),
    ("The presentation went really well this morning.", "今朝のプレゼンはとてもうまくいきました。"),
    ("We received positive feedback from the client.", "クライアントから良いフィードバックをもらいました。"),
    ("The quarterly review is scheduled for next week.", "四半期レビューは来週に予定されています。"),
    ("I'll be out of office tomorrow afternoon.", "明日の午後は外出します。"),
    ("Can you take notes during the meeting?", "会議中にメモを取ってもらえますか？"),
    ("The annual report needs to be submitted by Friday.", "年次報告書は金曜日までに提出する必要があります。"),
    ("We have a new client starting from next month.", "来月から新しいクライアントが始まります。"),
    ("Please prepare a summary of today's discussion.", "今日の議論の要約を作成してください。"),
    ("The task has been assigned to the development team.", "タスクは開発チームに割り当てられました。"),
    ("I need your approval before moving forward.", "進める前にあなたの承認が必要です。"),
    ("Let's wrap up the meeting in the next ten minutes.", "あと10分で会議を終わらせましょう。"),
    ("The invoice has been sent to the accounting department.", "請求書は経理部門に送りました。"),
    ("We are expecting a big order from the new client.", "新しいクライアントから大きな注文を期待しています。"),
    ("The training session starts at nine in the morning.", "トレーニングセッションは朝9時から始まります。"),
    ("Please confirm your attendance for the workshop.", "ワークショップへの参加を確認してください。"),
    ("I need to speak with the HR department today.", "今日は人事部門と話す必要があります。"),
    ("The sales figures for last month were impressive.", "先月の売上数字は印象的でした。"),
    ("Can you handle the customer complaint today?", "今日は顧客の苦情に対応してもらえますか？"),
    ("We should update the project timeline immediately.", "すぐにプロジェクトのスケジュールを更新すべきです。"),
    ("The meeting has been postponed until next Tuesday.", "会議は来週の火曜日まで延期されました。"),
    ("I have a call with the overseas team tonight.", "今夜は海外チームとの電話があります。"),
    ("The new office policy was announced this morning.", "新しいオフィスポリシーが今朝発表されました。"),
    ("Please submit the expense report by end of month.", "月末までに経費精算書を提出してください。"),
    ("We need to hire two more engineers for the project.", "プロジェクトにエンジニアをあと2人採用する必要があります。"),
    ("The software demo is scheduled for Friday afternoon.", "ソフトウェアのデモは金曜日の午後に予定されています。"),
    ("Let me know if you need any help with the report.", "レポートについて何か助けが必要なら教えてください。"),
    ("The board approved the new budget for next year.", "取締役会は来年の新しい予算を承認しました。"),
    ("I'll send you the meeting notes by this evening.", "今夜までに会議のメモを送ります。"),
    ("The project kickoff meeting is tomorrow at ten.", "プロジェクトキックオフ会議は明日10時です。"),
    ("We need to finalize the design before the end of the week.", "週末までにデザインを確定させる必要があります。"),
]

TECH_PAIRS = [
    ("The bug in the login module has been fixed.", "ログインモジュールのバグが修正されました。"),
    ("We need to deploy the update to the production server.", "本番サーバーにアップデートをデプロイする必要があります。"),
    ("The application is running slow due to a memory leak.", "メモリリークのためアプリケーションが遅く動作しています。"),
    ("Please back up the database before running the migration.", "マイグレーションを実行する前にデータベースをバックアップしてください。"),
    ("The new feature was merged into the main branch.", "新しい機能がメインブランチにマージされました。"),
    ("There is an error in the API response format.", "APIレスポンスフォーマットにエラーがあります。"),
    ("The server went down for about thirty minutes last night.", "昨夜、サーバーが約30分間ダウンしました。"),
    ("We need to update the SDK to the latest version.", "SDKを最新バージョンにアップデートする必要があります。"),
    ("The code review comments have been addressed.", "コードレビューのコメントに対応しました。"),
    ("The network latency is causing timeout issues.", "ネットワークの遅延がタイムアウトの問題を引き起こしています。"),
    ("Please write unit tests for the new functions.", "新しい関数のユニットテストを書いてください。"),
    ("The system logs show an unusual spike in traffic.", "システムログにトラフィックの異常な急増が見られます。"),
    ("I need to refactor this module to improve performance.", "パフォーマンスを向上させるためにこのモジュールをリファクタリングする必要があります。"),
    ("The pull request is ready for review.", "プルリクエストはレビューの準備ができています。"),
    ("We should upgrade the database engine to the newer version.", "データベースエンジンを新しいバージョンにアップグレードすべきです。"),
    ("The CI pipeline failed because of a dependency issue.", "依存関係の問題でCIパイプラインが失敗しました。"),
    ("Please check the error logs before tomorrow's standup.", "明日のスタンドアップ前にエラーログを確認してください。"),
    ("The API documentation needs to be updated.", "APIドキュメントを更新する必要があります。"),
    ("We are migrating the application to a cloud-based infrastructure.", "アプリケーションをクラウドベースのインフラに移行しています。"),
    ("The security vulnerability has been patched.", "セキュリティの脆弱性にパッチが当たりました。"),
    ("I'm working on optimizing the database queries.", "データベースクエリの最適化に取り組んでいます。"),
    ("The staging environment is ready for testing.", "ステージング環境はテストの準備ができています。"),
    ("Can you review the architecture design document?", "アーキテクチャ設計書をレビューしてもらえますか？"),
    ("The mobile app update is live on the app store.", "モバイルアプリのアップデートがアプリストアで公開されています。"),
    ("The data pipeline is failing at the transformation step.", "データパイプラインが変換ステップで失敗しています。"),
    ("We need to increase the server capacity for peak hours.", "ピーク時のためにサーバーの容量を増やす必要があります。"),
    ("The machine learning model needs more training data.", "機械学習モデルにはより多くのトレーニングデータが必要です。"),
    ("The configuration file has been updated correctly.", "設定ファイルが正しく更新されました。"),
    ("Please document your changes in the changelog.", "変更をチェンジログに記録してください。"),
    ("The load balancer is distributing traffic evenly.", "ロードバランサーがトラフィックを均等に分散しています。"),
    ("There is a memory overflow in the rendering engine.", "レンダリングエンジンにメモリオーバーフローがあります。"),
    ("The container image has been pushed to the registry.", "コンテナイメージがレジストリにプッシュされました。"),
    ("We need to implement proper error handling in this service.", "このサービスに適切なエラーハンドリングを実装する必要があります。"),
    ("The feature flag can be enabled in the configuration.", "機能フラグは設定で有効にできます。"),
    ("The test suite is taking too long to run.", "テストスイートの実行に時間がかかりすぎています。"),
    ("Please merge the hotfix branch into the release branch.", "ホットフィックスブランチをリリースブランチにマージしてください。"),
    ("The API rate limit is being exceeded by some clients.", "一部のクライアントがAPIのレート制限を超えています。"),
    ("The database schema migration completed successfully.", "データベーススキーマのマイグレーションが正常に完了しました。"),
    ("We should add caching to reduce the database load.", "データベースの負荷を減らすためにキャッシュを追加すべきです。"),
    ("The authentication service is responding correctly now.", "認証サービスが正常に応答しています。"),
    ("Can you set up the development environment for the new team member?", "新しいチームメンバーの開発環境をセットアップしてもらえますか？"),
    ("The endpoint is returning a five hundred error.", "エンドポイントが500エラーを返しています。"),
    ("I'll push the changes to the repository after testing.", "テスト後に変更をリポジトリにプッシュします。"),
    ("The new algorithm reduces processing time by forty percent.", "新しいアルゴリズムにより処理時間が40パーセント短縮されます。"),
    ("The front-end and back-end teams need to sync on the API spec.", "フロントエンドとバックエンドのチームがAPI仕様について同期する必要があります。"),
    ("Please add proper input validation to the form.", "フォームに適切な入力バリデーションを追加してください。"),
    ("The release has been rolled back due to a critical bug.", "重大なバグのためリリースがロールバックされました。"),
    ("We need to monitor the system performance after deployment.", "デプロイ後にシステムのパフォーマンスを監視する必要があります。"),
    ("The webhook integration is working as expected.", "Webhook統合が期待通りに動作しています。"),
    ("I'll write a script to automate the data cleanup process.", "データクリーンアッププロセスを自動化するスクリプトを書きます。"),
    ("The SSL certificate needs to be renewed before it expires.", "SSLcertificate is expiring and needs to be renewed."),
]

DAILY_PAIRS = [
    ("I'm going to grab a coffee before the meeting.", "会議の前にコーヒーを買いに行きます。"),
    ("What are you planning to have for lunch today?", "今日のランチは何を食べる予定ですか？"),
    ("The weather is so nice, let's eat outside.", "天気がいいので、外で食べましょう。"),
    ("I need to go to the gym after work today.", "今日は仕事の後にジムに行かなければなりません。"),
    ("The commute was really bad this morning.", "今朝の通勤はとても大変でした。"),
    ("Let's grab dinner together this evening.", "今夜、一緒に夕食を食べましょう。"),
    ("I tried a new recipe last weekend and it was great.", "先週末に新しいレシピを試してみたらとても美味しかった。"),
    ("The grocery store near my house is really convenient.", "家の近くのスーパーはとても便利です。"),
    ("I need to pick up some medicine on the way home.", "帰り道に薬を買っていかなければなりません。"),
    ("The new cafe that opened downtown looks really nice.", "ダウンタウンに新しくできたカフェがとてもよさそうです。"),
    ("I usually wake up at six thirty every morning.", "私はいつも毎朝6時半に起きます。"),
    ("The laundry has been piling up all week.", "洗濯物が一週間ずっとたまっています。"),
    ("I finally finished cleaning my apartment.", "やっとアパートの掃除が終わりました。"),
    ("Let's try that new ramen restaurant everyone is talking about.", "みんなが話している新しいラーメン屋を試してみましょう。"),
    ("I need to renew my monthly commuter pass today.", "今日は定期券を更新しなければなりません。"),
    ("The park nearby is beautiful during cherry blossom season.", "近くの公園は桜の季節にとても綺麗です。"),
    ("I went grocery shopping and spent way too much money.", "スーパーに買い物に行ったら使いすぎてしまいました。"),
    ("Can you recommend a good restaurant around here?", "この辺りで良いレストランを教えてもらえますか？"),
    ("I slept really well last night for once.", "久しぶりに昨夜はとてもよく眠れました。"),
    ("The line at the post office was incredibly long today.", "今日の郵便局の行列はものすごく長かったです。"),
    ("I'm trying to eat healthier these days.", "最近、もっと健康的な食事をするようにしています。"),
    ("It was so hot today that I didn't want to go outside.", "今日はとても暑くて外に出たくありませんでした。"),
    ("Let me know when you're free to meet up this week.", "今週いつ会えるか教えてください。"),
    ("The train was delayed by about twenty minutes this morning.", "今朝は電車が約20分遅延しました。"),
    ("I finally got around to organizing my bookshelf.", "やっと本棚を整理しました。"),
    ("The sunset was absolutely beautiful from my window.", "窓からの夕日がとても綺麗でした。"),
    ("I've been drinking too much coffee lately.", "最近コーヒーを飲みすぎています。"),
    ("Let's check out the weekend market together.", "一緒に週末のマーケットに行きましょう。"),
    ("I forgot to take out the trash this morning.", "今朝ゴミを出し忘れました。"),
    ("The air conditioner in my room broke down last night.", "昨夜、部屋のエアコンが壊れました。"),
    ("I had a really good workout at the gym today.", "今日はジムでとても良いトレーニングができました。"),
    ("Let's plan something fun for the long weekend.", "連休に何か楽しいことを計画しましょう。"),
    ("I discovered a great podcast about history.", "歴史に関する素晴らしいポッドキャストを見つけました。"),
    ("The flowers in the garden are blooming beautifully.", "庭の花がきれいに咲いています。"),
    ("I need to get my haircut sometime this week.", "今週中に髪を切りに行かなければなりません。"),
    ("The bookstore downtown has a great selection.", "ダウンタウンの本屋はとても品揃えが良いです。"),
    ("I've been watching a lot of documentaries lately.", "最近ドキュメンタリーをたくさん見ています。"),
    ("The vending machine on the third floor is out of order.", "3階の自動販売機が故障しています。"),
    ("I managed to catch the early train this morning.", "今朝は早い電車に乗ることができました。"),
    ("Let's order delivery tonight instead of cooking.", "今夜は料理せずにデリバリーを注文しましょう。"),
    ("I just got a new phone and I'm still setting it up.", "新しい電話を買ったばかりでまだセットアップ中です。"),
    ("The neighborhood festival is happening this weekend.", "今週末に近所のお祭りがあります。"),
    ("I need to take my cat to the vet for a checkup.", "猫を検診のために獣医に連れて行かなければなりません。"),
    ("The flowers at the florist near the station look amazing.", "駅近くの花屋の花がとても素敵です。"),
    ("Can we stop at the convenience store on the way?", "途中でコンビニに寄れますか？"),
    ("I finally finished reading that novel I started months ago.", "何ヶ月も前に始めた小説をやっと読み終えました。"),
    ("The recycling bins in this building are confusing.", "このビルの分別ゴミ箱がわかりにくいです。"),
    ("I'm thinking of going for a morning jog tomorrow.", "明日は朝ジョギングしようかと思っています。"),
    ("The restaurant by the river has an amazing view.", "川沿いのレストランからの眺めは素晴らしいです。"),
    ("Let's meet at the coffee shop at noon.", "正午にコーヒーショップで会いましょう。"),
]

SOCIAL_PAIRS = [
    ("Are you coming to the party this Saturday?", "今週土曜日のパーティーに来ますか？"),
    ("The concert last night was absolutely incredible.", "昨夜のコンサートは本当に素晴らしかったです。"),
    ("Let's plan a trip somewhere during the summer holidays.", "夏休みにどこかに旅行を計画しましょう。"),
    ("I watched a really good movie last night.", "昨夜とても良い映画を見ました。"),
    ("Have you heard the new album from that band?", "あのバンドの新しいアルバムを聴きましたか？"),
    ("We should catch up over coffee sometime soon.", "近いうちにコーヒーを飲みながら近況報告しましょう。"),
    ("The art exhibition at the museum was very inspiring.", "美術館の美術展はとても感動的でした。"),
    ("I'm thinking of taking up photography as a hobby.", "趣味として写真を始めようかと思っています。"),
    ("The game yesterday was so exciting to watch.", "昨日の試合はとても興奮する試合でした。"),
    ("We should organize a barbecue party this summer.", "今夏にバーベキューパーティーを企画しましょう。"),
    ("I just started reading a really interesting book.", "とても面白い本を読み始めました。"),
    ("The hiking trail in the mountains was breathtaking.", "山のハイキングコースからの眺めは息をのむほど素晴らしかった。"),
    ("Have you tried the new virtual reality experience in town?", "街で新しいVR体験を試してみましたか？"),
    ("Let's go see that new action movie this weekend.", "今週末にその新しいアクション映画を見に行きましょう。"),
    ("The jazz festival downtown was such a great atmosphere.", "ダウンタウンのジャズフェスティバルはとても素晴らしい雰囲気でした。"),
    ("I've been playing guitar for about three years now.", "ギターを弾いて3年ほどになります。"),
    ("We should visit that famous temple while we're here.", "ここにいる間にあの有名なお寺を訪れましょう。"),
    ("I'm looking for a good book recommendation.", "良い本の推薦を探しています。"),
    ("The food at the new Italian restaurant was excellent.", "新しいイタリアンレストランの料理は素晴らしかったです。"),
    ("I'm planning to run a half marathon next spring.", "来春にハーフマラソンを走る予定です。"),
    ("Did you see the news about the music festival?", "音楽フェスティバルのニュースを見ましたか？"),
    ("We went camping last weekend and it was so much fun.", "先週末にキャンプに行ってとても楽しかったです。"),
    ("Let's do a road trip next month.", "来月ロードトリップをしましょう。"),
    ("I tried a pottery class and really enjoyed it.", "陶芸のクラスを試してみてとても楽しかったです。"),
    ("The stand-up comedy show was hilarious last night.", "昨夜のスタンドアップコメディショーはとても面白かったです。"),
    ("Are you interested in joining our book club?", "私たちの読書クラブに参加しますか？"),
    ("I got tickets to the baseball game next weekend.", "来週末の野球の試合のチケットを手に入れました。"),
    ("The rooftop bar has an amazing view of the city.", "屋上バーから街の眺めが素晴らしいです。"),
    ("Let's go to karaoke after dinner tonight.", "今夜は夕食後にカラオケに行きましょう。"),
    ("I've been binge-watching that drama series all week.", "一週間ずっとそのドラマシリーズを一気見しています。"),
    ("The street food at the night market was amazing.", "ナイトマーケットの屋台料理は素晴らしかったです。"),
    ("I'm learning how to cook Japanese cuisine.", "日本料理の作り方を習っています。"),
    ("We should organize a team outing next month.", "来月チームの外出を企画しましょう。"),
    ("I've been getting into yoga lately.", "最近ヨガにはまっています。"),
    ("The photo exhibition at the gallery was really moving.", "ギャラリーの写真展はとても感動的でした。"),
    ("Let's celebrate your promotion with a nice dinner.", "昇進をご馳走でお祝いしましょう。"),
    ("I found a great deal on flights for the holidays.", "休日のフライトでお得な値段を見つけました。"),
    ("The escape room we tried last week was very challenging.", "先週試したエスケープルームはとても難しかったです。"),
    ("I want to learn how to surf this summer.", "今夏はサーフィンを覚えたいです。"),
    ("We had a great time at the wine tasting event.", "ワインテイスティングイベントでとても楽しい時間を過ごしました。"),
    ("Let's visit the aquarium this weekend.", "今週末に水族館を訪れましょう。"),
    ("I started practicing meditation every morning.", "毎朝瞑想を練習し始めました。"),
    ("The comedy movie I watched last night made me cry from laughing.", "昨夜見たコメディ映画は笑いすぎて泣きました。"),
    ("We should check out that new izakaya near the station.", "駅近くの新しい居酒屋を試してみましょう。"),
    ("I'm planning a surprise birthday party for my friend.", "友達のためにサプライズ誕生日パーティーを計画しています。"),
    ("The fireworks festival next week is going to be spectacular.", "来週の花火大会は壮観になるでしょう。"),
    ("Let's take a day trip to the countryside this weekend.", "今週末に田舎へ日帰り旅行をしましょう。"),
    ("I've been learning watercolor painting recently.", "最近水彩画を習っています。"),
    ("Did you see the documentary about ocean conservation?", "海洋保護に関するドキュメンタリーを見ましたか？"),
    ("The outdoor music event was cancelled due to rain.", "雨のため屋外音楽イベントが中止されました。"),
    ("I want to start volunteering at the local animal shelter.", "地元の動物シェルターでボランティアを始めたいです。"),
]

EDUCATION_PAIRS = [
    ("The professor assigned a lot of homework this week.", "今週、教授がたくさんの宿題を出しました。"),
    ("I need to study for the exam scheduled for next week.", "来週に予定されている試験のために勉強しなければなりません。"),
    ("The library is open until ten o'clock tonight.", "今夜、図書館は10時まで開いています。"),
    ("Can you explain this concept to me again?", "このコンセプトをもう一度説明してもらえますか？"),
    ("The research paper is due at the end of the semester.", "研究論文は学期末に提出です。"),
    ("I'm having trouble understanding this chapter.", "この章を理解するのに苦労しています。"),
    ("The group project deadline is approaching fast.", "グループプロジェクトの締め切りが近づいています。"),
    ("Let's form a study group for the final exam.", "期末試験のためにスタディグループを作りましょう。"),
    ("The scholarship application requires a letter of recommendation.", "奨学金の申請には推薦状が必要です。"),
    ("I attended an online seminar about machine learning yesterday.", "昨日、機械学習についてのオンラインセミナーに参加しました。"),
    ("The graduation ceremony is scheduled for next spring.", "卒業式は来春に予定されています。"),
    ("I need to choose my courses for next semester.", "来学期の授業を選ばなければなりません。"),
    ("The internship application deadline is tomorrow morning.", "インターンシップの申請締め切りは明日の朝です。"),
    ("I'm writing a thesis on artificial intelligence.", "人工知能についての論文を書いています。"),
    ("The new curriculum starts from next academic year.", "新しいカリキュラムは来年度から始まります。"),
    ("I received a scholarship to study abroad next year.", "来年留学するための奨学金を受けました。"),
    ("Can you share your notes from the lecture yesterday?", "昨日の授業のノートを共有してもらえますか？"),
    ("The university campus is beautiful in autumn.", "大学のキャンパスは秋にとても美しいです。"),
    ("I'm participating in a coding competition next month.", "来月コーディングコンペティションに参加します。"),
    ("The professor's office hours are on Tuesday afternoons.", "教授のオフィスアワーは火曜日の午後です。"),
    ("I failed the quiz and need to improve my understanding.", "クイズに失敗したので理解を深める必要があります。"),
    ("The online learning platform has great resources.", "オンライン学習プラットフォームには素晴らしいリソースがあります。"),
    ("I've been working on my presentation for three days.", "3日間プレゼンに取り組んでいます。"),
    ("The guest speaker at today's lecture was very insightful.", "今日の講義のゲストスピーカーはとても洞察力がありました。"),
    ("I finally submitted my research paper this morning.", "今朝やっと研究論文を提出しました。"),
    ("The course covers topics from mathematics to philosophy.", "このコースは数学から哲学までのトピックをカバーしています。"),
    ("I need to improve my English speaking skills.", "英語のスピーキングスキルを向上させる必要があります。"),
    ("The exam results will be announced next week.", "試験の結果は来週発表されます。"),
    ("I'm thinking of changing my major to data science.", "専攻をデータサイエンスに変更しようかと思っています。"),
    ("The student council is organizing a cultural festival.", "学生会が文化祭を企画しています。"),
    ("I enjoy discussing philosophy with my classmates.", "クラスメートと哲学について議論するのが好きです。"),
    ("The professor cancelled class because of a conference.", "教授が会議のため授業をキャンセルしました。"),
    ("I've been attending extra tutoring sessions.", "補習授業に参加しています。"),
    ("The assignment requires both research and analysis.", "課題にはリサーチと分析の両方が必要です。"),
    ("I need to borrow a textbook from the library.", "図書館から教科書を借りる必要があります。"),
    ("The language lab is available for practice every day.", "語学ラボは毎日練習のために利用できます。"),
    ("I'm struggling with the programming assignment.", "プログラミングの課題に苦労しています。"),
    ("The field trip to the science museum is next Friday.", "理科博物館への校外学習は来週の金曜日です。"),
    ("I received helpful comments on my draft essay.", "下書きのエッセイに役立つコメントをもらいました。"),
    ("The elective course on creative writing is very popular.", "創作文章の選択科目はとても人気があります。"),
    ("I need to register for courses before the deadline.", "締め切りまでに授業を登録しなければなりません。"),
    ("The study abroad program in Europe sounds interesting.", "ヨーロッパの留学プログラムは面白そうです。"),
    ("I passed my driving theory test on the first attempt.", "一回目の挑戦で運転の学科試験に合格しました。"),
    ("The new student orientation is tomorrow morning.", "新入生オリエンテーションは明日の朝です。"),
    ("I've been spending a lot of time in the library.", "図書館でたくさんの時間を過ごしています。"),
    ("The documentary we watched in class was very thought-provoking.", "授業で見たドキュメンタリーはとても考えさせられました。"),
    ("I need to improve my essay writing before the next submission.", "次の提出前にエッセイの書き方を改善する必要があります。"),
    ("The workshop on public speaking was really helpful.", "プレゼンテーションのワークショップはとても役に立ちました。"),
    ("I'm applying for a research grant this semester.", "今学期、研究助成金に申請しています。"),
    ("The final exam covers everything from the whole semester.", "期末試験は学期全体の内容を含みます。"),
    ("I want to pursue a master's degree after graduation.", "卒業後に修士号を取得したいです。"),
]

HEALTH_PAIRS = [
    ("I've been feeling a bit under the weather lately.", "最近少し体調が優れません。"),
    ("Make sure to drink plenty of water throughout the day.", "一日中十分な水を飲むようにしてください。"),
    ("I need to schedule a checkup with my doctor.", "医者と検診の予約をしなければなりません。"),
    ("I've been exercising regularly and feeling much better.", "定期的に運動していてとても元気になっています。"),
    ("The gym opens at six in the morning on weekdays.", "ジムは平日の朝6時に開きます。"),
    ("I've been trying to get at least eight hours of sleep.", "少なくとも8時間の睡眠を取るようにしています。"),
    ("Eating a balanced diet is really important for your health.", "バランスの良い食事は健康にとても重要です。"),
    ("I'm recovering from a cold so I need to rest.", "風邪から回復中なので休息が必要です。"),
    ("My allergies are really bad during spring.", "春は花粉症がひどいです。"),
    ("I started taking vitamins every morning.", "毎朝ビタミンを飲み始めました。"),
    ("The doctor recommended I reduce my sugar intake.", "医者から糖分の摂取を減らすよう勧められました。"),
    ("I've been feeling more energetic since I started jogging.", "ジョギングを始めてから体力がついてきました。"),
    ("Mental health is just as important as physical health.", "精神的な健康は身体的な健康と同じくらい重要です。"),
    ("I pulled a muscle while exercising at the gym.", "ジムで運動中に筋肉を痛めてしまいました。"),
    ("The pharmacist recommended this medicine for my headache.", "薬剤師が頭痛にこの薬を推薦しました。"),
    ("I've been drinking herbal tea to help me sleep better.", "よく眠れるようにハーブティーを飲んでいます。"),
    ("Stretching before exercise helps prevent injuries.", "運動前のストレッチは怪我の予防になります。"),
    ("I need to take a break and reduce my stress levels.", "休んでストレスレベルを下げる必要があります。"),
    ("The dentist appointment is on Thursday morning.", "歯医者の予約は木曜日の朝です。"),
    ("I've been tracking my steps with a fitness app.", "フィットネスアプリで歩数を記録しています。"),
    ("Meditation has helped me feel much calmer.", "瞑想で気持ちがとても落ち着きました。"),
    ("I need to take my medication with food.", "食事と一緒に薬を飲む必要があります。"),
    ("Regular exercise helps maintain a healthy weight.", "定期的な運動は健康的な体重の維持に役立ちます。"),
    ("I've been having back pain from sitting too long.", "長時間座っていて腰痛があります。"),
    ("The hospital is offering free health screenings this month.", "今月、病院で無料の健康診断を実施しています。"),
    ("I should cut down on caffeine before bed.", "就寝前はカフェインを控えるべきです。"),
    ("Taking a walk after dinner aids digestion.", "夕食後の散歩は消化を助けます。"),
    ("I've lost five kilograms since I changed my diet.", "食事を変えてから5キロ体重が減りました。"),
    ("The gym instructor recommended a new workout routine.", "ジムのインストラクターが新しいトレーニングルーティンを勧めました。"),
    ("Staying hydrated is important especially in summer.", "特に夏は水分を補給することが大切です。"),
    ("I need to book a physiotherapy session for my knee.", "膝の理学療法のセッションを予約する必要があります。"),
    ("Fresh fruits and vegetables are essential for good health.", "新鮮な果物と野菜は良い健康に不可欠です。"),
    ("I've been practicing deep breathing exercises.", "腹式呼吸の練習をしています。"),
    ("The hospital wait times have been very long recently.", "最近、病院の待ち時間がとても長いです。"),
    ("I'm feeling much better after a good night's sleep.", "よく眠れて気分がとても良くなりました。"),
    ("I need to reduce my screen time before bedtime.", "就寝前のスクリーンタイムを減らす必要があります。"),
    ("Going for walks in nature really helps clear my mind.", "自然の中を散歩すると気持ちがとてもすっきりします。"),
    ("I've started cooking healthier meals at home.", "家でより健康的な食事を作り始めました。"),
    ("The fitness challenge at work has been very motivating.", "職場でのフィットネスチャレンジはとてもモチベーションが上がります。"),
    ("I've been experiencing a lot of stress at work.", "職場でとてもストレスを感じています。"),
    ("Taking short breaks during work improves productivity.", "仕事中に短い休憩を取ると生産性が上がります。"),
    ("I should get my blood pressure checked regularly.", "定期的に血圧を測るべきです。"),
    ("I've been sleeping early and waking up refreshed.", "早く寝て気持ちよく目覚めています。"),
    ("The swimming pool at the sports center is always clean.", "スポーツセンターのプールはいつも清潔です。"),
    ("I need to see a nutritionist about my diet.", "食事について栄養士に相談する必要があります。"),
    ("The local running club meets every Sunday morning.", "地元のランニングクラブは毎週日曜の朝に集まります。"),
    ("I've been adding more protein to my daily meals.", "毎日の食事にタンパク質を多く加えています。"),
    ("Getting a massage helped relieve my muscle tension.", "マッサージを受けて筋肉の緊張がほぐれました。"),
    ("I'm allergic to cats but I love them anyway.", "猫アレルギーですが、それでも猫が大好きです。"),
    ("I feel so much more energized after morning exercise.", "朝の運動の後はとても元気になります。"),
    ("Quitting sugar was hard but it made a big difference.", "砂糖をやめるのは大変でしたが、大きな変化がありました。"),
]

ALL_PAIRS = {
    "work":      WORK_PAIRS[:50],
    "tech":      TECH_PAIRS[:50],
    "daily":     DAILY_PAIRS[:50],
    "social":    SOCIAL_PAIRS[:50],
    "education": EDUCATION_PAIRS[:50],
    "health":    HEALTH_PAIRS[:50],
}

assert sum(len(v) for v in ALL_PAIRS.values()) >= N_SAMPLES


def get_all_pairs():
    flat = []
    for topic, pairs in ALL_PAIRS.items():
        for en, ja in pairs:
            flat.append({"topic": topic, "en": en, "ja": ja})
    random.shuffle(flat)
    return flat[:N_SAMPLES]


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


async def generate_sample(idx: int, pair: dict) -> dict:
    mp3  = await tts_bytes(pair["en"], VOICE_EN)
    wav, duration = _mp3_to_wav(mp3)

    filename = f"en_{idx:03d}.wav"
    filepath = AUDIO_OUT / filename
    filepath.write_bytes(wav)

    return {
        "id":                    f"en_{idx:03d}",
        "audio_file":            str(filepath),
        "topic":                 pair["topic"],
        "english_transcription": pair["en"],
        "japanese_translation":  pair["ja"],
        "duration_seconds":      round(duration, 2),
    }


async def main():
    pairs = get_all_pairs()
    samples = []
    for i, pair in enumerate(pairs, start=1):
        print(f"  [{i}/{N_SAMPLES}] en_{i:03d}", flush=True)
        try:
            sample = await generate_sample(i, pair)
            samples.append(sample)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)

    out_path = META_OUT / "transcriptions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(samples)} samples → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
