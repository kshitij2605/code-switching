"""
Generate 300 Japanese-only audio samples using edge-tts.

Each sample has:
  - audio file (wav, Japanese voice)
  - japanese_transcription
  - english_translation (for "transcribe in English" fine-tuning target)
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

AUDIO_OUT = Path("data/japanese_only/audio")
META_OUT  = Path("data/japanese_only/metadata")
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
META_OUT.mkdir(parents=True, exist_ok=True)

VOICE_JA  = "ja-JP-NanamiNeural"
N_SAMPLES = 300
PROXY     = None  # set to "http://host:port/" if behind a proxy
random.seed(456)

# ---------------------------------------------------------------------------
# Sentence pairs (ja_text, en_translation) — 6 topics × 50 pairs
# ---------------------------------------------------------------------------

WORK_PAIRS = [
    ("明日の会議の資料を準備しておいてください。", "Please prepare the materials for tomorrow's meeting."),
    ("このプロジェクトの締め切りは来週の金曜日です。", "The deadline for this project is next Friday."),
    ("報告書を今日中に提出してもらえますか？", "Could you submit the report by today?"),
    ("クライアントへのプレゼンの準備が必要です。", "We need to prepare for the client presentation."),
    ("今四半期の予算は思ったより厳しいです。", "The budget for this quarter is tighter than expected."),
    ("午後はずっと会議が続くので大変です。", "It will be tough since meetings continue all afternoon."),
    ("提案書についてのご意見をお聞かせください。", "Please let us hear your opinion on the proposal."),
    ("今週はチーム全員が残業しています。", "The entire team is working overtime this week."),
    ("月曜日までに契約を確定させる必要があります。", "We need to finalize the contract by Monday."),
    ("会議が始まる前に議題を確認しましょう。", "Let's confirm the agenda before the meeting starts."),
    ("クライアントが新しいデザイン案を承認しました。", "The client approved the new design proposal."),
    ("今日は月次進捗報告を作成しなければなりません。", "I need to prepare the monthly progress report today."),
    ("プロジェクトが二週間遅延しています。", "The project is behind schedule by two weeks."),
    ("チームの会議を木曜日に変更できますか？", "Can we change the team meeting to Thursday?"),
    ("新しいマネージャーが来週月曜日から勤務します。", "The new manager starts working here next Monday."),
    ("財務チームと予算について話し合う必要があります。", "We need to discuss the budget with the finance team."),
    ("サインする前に契約書を確認してください。", "Please review the contract before signing."),
    ("今年の目標は売上を二十パーセント増やすことです。", "Our goal this year is to increase sales by twenty percent."),
    ("提案書は提出前に修正が必要です。", "The proposal needs to be revised before submission."),
    ("次のステップについて話し合いましょう。", "Let's discuss the next steps together."),
    ("今朝のプレゼンはとてもうまくいきました。", "The presentation went very well this morning."),
    ("クライアントから良いフィードバックをもらいました。", "We received positive feedback from the client."),
    ("四半期レビューは来週に予定されています。", "The quarterly review is scheduled for next week."),
    ("明日の午後は外出の予定があります。", "I have plans to go out tomorrow afternoon."),
    ("会議中にメモを取ってもらえますか？", "Could you take notes during the meeting?"),
    ("年次報告書は金曜日までに提出する必要があります。", "The annual report needs to be submitted by Friday."),
    ("来月から新しいクライアントが始まります。", "A new client starts from next month."),
    ("今日の議論の要約を作成してください。", "Please prepare a summary of today's discussion."),
    ("タスクは開発チームに割り当てられました。", "The task has been assigned to the development team."),
    ("進める前にご承認をいただく必要があります。", "We need your approval before moving forward."),
    ("あと十分で会議を終わらせましょう。", "Let's wrap up the meeting in the next ten minutes."),
    ("請求書は経理部門に送りました。", "The invoice has been sent to the accounting department."),
    ("新しいクライアントから大きな注文が来る予定です。", "We are expecting a large order from the new client."),
    ("トレーニングセッションは朝九時から始まります。", "The training session starts at nine in the morning."),
    ("ワークショップへのご参加をご確認ください。", "Please confirm your attendance for the workshop."),
    ("今日は人事部門と話す必要があります。", "I need to speak with the HR department today."),
    ("先月の売上数字は印象的でした。", "The sales figures for last month were impressive."),
    ("今日は顧客の苦情に対応してもらえますか？", "Can you handle the customer complaint today?"),
    ("すぐにプロジェクトのスケジュールを更新すべきです。", "We should update the project schedule immediately."),
    ("会議は来週の火曜日まで延期されました。", "The meeting has been postponed until next Tuesday."),
    ("今夜は海外チームとの電話があります。", "I have a call with the overseas team tonight."),
    ("新しいオフィス方針が今朝発表されました。", "The new office policy was announced this morning."),
    ("月末までに経費精算書を提出してください。", "Please submit the expense report by end of month."),
    ("このプロジェクトにエンジニアをあと二人採用する必要があります。", "We need to hire two more engineers for this project."),
    ("ソフトウェアのデモは金曜日の午後に予定されています。", "The software demo is scheduled for Friday afternoon."),
    ("報告書についてお手伝いが必要な場合はお知らせください。", "Let me know if you need any help with the report."),
    ("取締役会が来年の新しい予算を承認しました。", "The board approved the new budget for next year."),
    ("今夜までに会議のメモをお送りします。", "I will send you the meeting notes by this evening."),
    ("プロジェクトキックオフ会議は明日の十時です。", "The project kickoff meeting is tomorrow at ten."),
    ("週末までにデザインを確定させる必要があります。", "We need to finalize the design by the end of the week."),
]

TECH_PAIRS = [
    ("ログインモジュールのバグが修正されました。", "The bug in the login module has been fixed."),
    ("本番サーバーにアップデートをデプロイする必要があります。", "We need to deploy the update to the production server."),
    ("メモリリークのためアプリケーションが遅く動作しています。", "The application is running slow due to a memory leak."),
    ("マイグレーション前にデータベースをバックアップしてください。", "Please back up the database before running the migration."),
    ("新しい機能がメインブランチにマージされました。", "The new feature was merged into the main branch."),
    ("APIのレスポンス形式にエラーがあります。", "There is an error in the API response format."),
    ("昨夜サーバーが約三十分ダウンしました。", "The server went down for about thirty minutes last night."),
    ("SDKを最新バージョンに更新する必要があります。", "We need to update the SDK to the latest version."),
    ("コードレビューのコメントに対応しました。", "The code review comments have been addressed."),
    ("ネットワークの遅延がタイムアウトの問題を引き起こしています。", "Network latency is causing timeout issues."),
    ("新しい関数のユニットテストを書いてください。", "Please write unit tests for the new functions."),
    ("システムログにトラフィックの異常な増加が見られます。", "The system logs show an unusual spike in traffic."),
    ("パフォーマンス向上のためにこのモジュールをリファクタリングします。", "I will refactor this module to improve performance."),
    ("プルリクエストがレビューの準備できています。", "The pull request is ready for review."),
    ("データベースエンジンを新バージョンにアップグレードすべきです。", "We should upgrade the database engine to the newer version."),
    ("依存関係の問題でCIパイプラインが失敗しました。", "The CI pipeline failed because of a dependency issue."),
    ("明日のスタンドアップ前にエラーログを確認してください。", "Please check the error logs before tomorrow's standup."),
    ("APIドキュメントを更新する必要があります。", "The API documentation needs to be updated."),
    ("クラウドベースのインフラにアプリケーションを移行しています。", "We are migrating the application to cloud-based infrastructure."),
    ("セキュリティの脆弱性にパッチが当たりました。", "The security vulnerability has been patched."),
    ("データベースクエリの最適化に取り組んでいます。", "I am working on optimizing the database queries."),
    ("ステージング環境はテストの準備ができています。", "The staging environment is ready for testing."),
    ("アーキテクチャ設計書をレビューしてもらえますか？", "Can you review the architecture design document?"),
    ("モバイルアプリのアップデートがアプリストアで公開されました。", "The mobile app update is now live on the app store."),
    ("データパイプラインが変換ステップで失敗しています。", "The data pipeline is failing at the transformation step."),
    ("ピーク時のためにサーバーの容量を増やす必要があります。", "We need to increase the server capacity for peak hours."),
    ("機械学習モデルにはより多くのトレーニングデータが必要です。", "The machine learning model needs more training data."),
    ("設定ファイルが正しく更新されました。", "The configuration file has been updated correctly."),
    ("変更をチェンジログに記録してください。", "Please document your changes in the changelog."),
    ("ロードバランサーがトラフィックを均等に分散しています。", "The load balancer is distributing traffic evenly."),
    ("レンダリングエンジンにメモリオーバーフローがあります。", "There is a memory overflow in the rendering engine."),
    ("コンテナイメージがレジストリにプッシュされました。", "The container image has been pushed to the registry."),
    ("このサービスに適切なエラーハンドリングを実装する必要があります。", "We need to implement proper error handling in this service."),
    ("機能フラグは設定ファイルで有効にできます。", "The feature flag can be enabled in the configuration file."),
    ("テストスイートの実行に時間がかかりすぎています。", "The test suite is taking too long to run."),
    ("ホットフィックスブランチをリリースブランチにマージしてください。", "Please merge the hotfix branch into the release branch."),
    ("一部のクライアントがAPIのレート制限を超えています。", "Some clients are exceeding the API rate limit."),
    ("データベーススキーマの移行が正常に完了しました。", "The database schema migration completed successfully."),
    ("データベースの負荷を減らすためにキャッシュを追加すべきです。", "We should add caching to reduce the database load."),
    ("認証サービスが正常に応答するようになりました。", "The authentication service is now responding correctly."),
    ("新しいチームメンバーの開発環境をセットアップしてもらえますか？", "Can you set up the development environment for the new team member?"),
    ("エンドポイントが500エラーを返しています。", "The endpoint is returning a 500 error."),
    ("テスト後に変更をリポジトリにプッシュします。", "I will push the changes to the repository after testing."),
    ("新しいアルゴリズムにより処理時間が四十パーセント短縮されます。", "The new algorithm reduces processing time by forty percent."),
    ("フロントエンドとバックエンドのチームがAPI仕様について同期する必要があります。", "The front-end and back-end teams need to sync on the API spec."),
    ("フォームに適切な入力バリデーションを追加してください。", "Please add proper input validation to the form."),
    ("重大なバグのためリリースがロールバックされました。", "The release was rolled back due to a critical bug."),
    ("デプロイ後にシステムのパフォーマンスを監視する必要があります。", "We need to monitor the system performance after deployment."),
    ("Webhook統合が期待通りに動作しています。", "The webhook integration is working as expected."),
    ("データクリーンアップを自動化するスクリプトを書きます。", "I will write a script to automate the data cleanup process."),
]

DAILY_PAIRS = [
    ("会議の前にコーヒーを買いに行きます。", "I'm going to grab a coffee before the meeting."),
    ("今日のランチは何を食べる予定ですか？", "What are you planning to have for lunch today?"),
    ("天気がいいので外で食べましょう。", "The weather is nice so let's eat outside."),
    ("今日は仕事の後にジムに行かなければなりません。", "I need to go to the gym after work today."),
    ("今朝の通勤はとても大変でした。", "The commute was really tough this morning."),
    ("今夜一緒に夕食を食べましょう。", "Let's have dinner together tonight."),
    ("先週末に新しいレシピを試したらとても美味しかった。", "I tried a new recipe last weekend and it was delicious."),
    ("家の近くのスーパーはとても便利です。", "The supermarket near my house is very convenient."),
    ("帰り道に薬を買っていかなければなりません。", "I need to pick up medicine on the way home."),
    ("ダウンタウンに新しくできたカフェがよさそうです。", "The new cafe that opened downtown looks great."),
    ("私はいつも毎朝六時半に起きます。", "I always wake up at six thirty every morning."),
    ("洗濯物が一週間ずっとたまっています。", "The laundry has been piling up all week."),
    ("やっとアパートの掃除が終わりました。", "I finally finished cleaning my apartment."),
    ("みんなが話している新しいラーメン屋を試してみましょう。", "Let's try that new ramen restaurant everyone is talking about."),
    ("今日は定期券を更新しなければなりません。", "I need to renew my commuter pass today."),
    ("近くの公園は桜の季節にとても綺麗です。", "The nearby park is very beautiful during cherry blossom season."),
    ("スーパーに買い物に行ったら使いすぎてしまいました。", "I went grocery shopping and spent too much money."),
    ("この辺りで良いレストランを教えてもらえますか？", "Can you recommend a good restaurant around here?"),
    ("久しぶりに昨夜はとてもよく眠れました。", "I slept really well last night for the first time in a while."),
    ("今日の郵便局の行列はものすごく長かったです。", "The line at the post office was incredibly long today."),
    ("最近もっと健康的な食事をするようにしています。", "I have been trying to eat healthier lately."),
    ("今日はとても暑くて外に出たくありませんでした。", "It was so hot today that I did not want to go outside."),
    ("今週いつ会えるか教えてください。", "Let me know when you are free to meet up this week."),
    ("今朝は電車が約二十分遅延しました。", "The train was delayed by about twenty minutes this morning."),
    ("やっと本棚を整理しました。", "I finally got around to organizing my bookshelf."),
    ("窓からの夕日がとても綺麗でした。", "The sunset from my window was very beautiful."),
    ("最近コーヒーを飲みすぎています。", "I have been drinking too much coffee lately."),
    ("一緒に週末のマーケットに行きましょう。", "Let's check out the weekend market together."),
    ("今朝ゴミを出し忘れました。", "I forgot to take out the trash this morning."),
    ("昨夜部屋のエアコンが壊れました。", "The air conditioner in my room broke down last night."),
    ("今日はジムでとても良いトレーニングができました。", "I had a really good workout at the gym today."),
    ("連休に何か楽しいことを計画しましょう。", "Let's plan something fun for the long weekend."),
    ("歴史に関する素晴らしいポッドキャストを見つけました。", "I discovered a great podcast about history."),
    ("庭の花がきれいに咲いています。", "The flowers in the garden are blooming beautifully."),
    ("今週中に髪を切りに行かなければなりません。", "I need to get my haircut sometime this week."),
    ("ダウンタウンの本屋はとても品揃えが良いです。", "The bookstore downtown has a great selection."),
    ("最近ドキュメンタリーをたくさん見ています。", "I have been watching a lot of documentaries lately."),
    ("三階の自動販売機が故障しています。", "The vending machine on the third floor is out of order."),
    ("今朝は早い電車に乗ることができました。", "I managed to catch the early train this morning."),
    ("今夜は料理せずにデリバリーを注文しましょう。", "Let's order delivery tonight instead of cooking."),
    ("新しい電話を買ったばかりでまだセットアップ中です。", "I just got a new phone and I am still setting it up."),
    ("今週末に近所のお祭りがあります。", "There is a neighborhood festival this weekend."),
    ("猫を検診のために獣医に連れて行かなければなりません。", "I need to take my cat to the vet for a checkup."),
    ("駅近くの花屋の花がとても素敵です。", "The flowers at the florist near the station look amazing."),
    ("途中でコンビニに寄れますか？", "Can we stop at the convenience store on the way?"),
    ("何ヶ月も前に始めた小説をやっと読み終えました。", "I finally finished reading the novel I started months ago."),
    ("このビルの分別ゴミ箱がわかりにくいです。", "The recycling bins in this building are confusing."),
    ("明日は朝ジョギングしようかと思っています。", "I am thinking of going for a morning jog tomorrow."),
    ("川沿いのレストランからの眺めは素晴らしいです。", "The restaurant by the river has an amazing view."),
    ("正午にコーヒーショップで会いましょう。", "Let's meet at the coffee shop at noon."),
]

SOCIAL_PAIRS = [
    ("今週土曜日のパーティーに来ますか？", "Are you coming to the party this Saturday?"),
    ("昨夜のコンサートは本当に素晴らしかったです。", "The concert last night was absolutely incredible."),
    ("夏休みにどこかへ旅行を計画しましょう。", "Let's plan a trip somewhere during the summer holidays."),
    ("昨夜とても良い映画を見ました。", "I watched a really good movie last night."),
    ("あのバンドの新しいアルバムを聴きましたか？", "Have you heard the new album from that band?"),
    ("近いうちにコーヒーを飲みながら近況を話しましょう。", "Let's catch up over coffee sometime soon."),
    ("美術館の美術展はとても感動的でした。", "The art exhibition at the museum was very inspiring."),
    ("趣味として写真を始めようかと思っています。", "I am thinking of taking up photography as a hobby."),
    ("昨日の試合はとても興奮しました。", "Yesterday's game was so exciting to watch."),
    ("今夏にバーベキューパーティーを企画しましょう。", "Let's organize a barbecue party this summer."),
    ("とても面白い本を読み始めました。", "I just started reading a very interesting book."),
    ("山のハイキングコースからの眺めは息をのむほど素晴らしかった。", "The view from the mountain hiking trail was breathtaking."),
    ("街で新しいVR体験を試してみましたか？", "Have you tried the new virtual reality experience in town?"),
    ("今週末にその新しいアクション映画を見に行きましょう。", "Let's go see that new action movie this weekend."),
    ("ダウンタウンのジャズフェスティバルはとても素晴らしい雰囲気でした。", "The jazz festival downtown had such a great atmosphere."),
    ("ギターを弾いて三年ほどになります。", "I have been playing guitar for about three years now."),
    ("ここにいる間にあの有名なお寺を訪れましょう。", "Let's visit that famous temple while we are here."),
    ("良い本の推薦を探しています。", "I am looking for a good book recommendation."),
    ("新しいイタリアンレストランの料理は素晴らしかったです。", "The food at the new Italian restaurant was excellent."),
    ("来春にハーフマラソンを走る予定です。", "I am planning to run a half marathon next spring."),
    ("音楽フェスティバルのニュースを見ましたか？", "Did you see the news about the music festival?"),
    ("先週末にキャンプに行ってとても楽しかったです。", "We went camping last weekend and had so much fun."),
    ("来月ロードトリップをしましょう。", "Let's do a road trip next month."),
    ("陶芸のクラスを試してみてとても楽しかったです。", "I tried a pottery class and really enjoyed it."),
    ("昨夜のスタンドアップコメディショーはとても面白かったです。", "The stand-up comedy show was hilarious last night."),
    ("私たちの読書クラブに参加しますか？", "Are you interested in joining our book club?"),
    ("来週末の野球の試合のチケットを手に入れました。", "I got tickets to the baseball game next weekend."),
    ("屋上バーから街の眺めが素晴らしいです。", "The rooftop bar has an amazing view of the city."),
    ("今夜は夕食後にカラオケに行きましょう。", "Let's go to karaoke after dinner tonight."),
    ("一週間ずっとそのドラマシリーズを一気見しています。", "I have been binge-watching that drama series all week."),
    ("ナイトマーケットの屋台料理は素晴らしかったです。", "The street food at the night market was amazing."),
    ("日本料理の作り方を習っています。", "I am learning how to cook Japanese cuisine."),
    ("来月チームの外出を企画しましょう。", "Let's organize a team outing next month."),
    ("最近ヨガにはまっています。", "I have been getting into yoga lately."),
    ("ギャラリーの写真展はとても感動的でした。", "The photo exhibition at the gallery was really moving."),
    ("昇進をご馳走でお祝いしましょう。", "Let's celebrate your promotion with a nice dinner."),
    ("休日のフライトでお得な値段を見つけました。", "I found a great deal on flights for the holidays."),
    ("先週試したエスケープルームはとても難しかったです。", "The escape room we tried last week was very challenging."),
    ("今夏はサーフィンを覚えたいです。", "I want to learn how to surf this summer."),
    ("ワインテイスティングイベントでとても楽しい時間を過ごしました。", "We had a great time at the wine tasting event."),
    ("今週末に水族館を訪れましょう。", "Let's visit the aquarium this weekend."),
    ("毎朝瞑想を練習し始めました。", "I started practicing meditation every morning."),
    ("昨夜見たコメディ映画は笑いすぎて泣きました。", "The comedy movie I watched last night made me laugh until I cried."),
    ("駅近くの新しい居酒屋を試してみましょう。", "Let's check out that new izakaya near the station."),
    ("友達のためにサプライズ誕生日パーティーを計画しています。", "I am planning a surprise birthday party for my friend."),
    ("来週の花火大会は壮観になるでしょう。", "The fireworks festival next week is going to be spectacular."),
    ("今週末に田舎へ日帰り旅行をしましょう。", "Let's take a day trip to the countryside this weekend."),
    ("最近水彩画を習っています。", "I have been learning watercolor painting recently."),
    ("海洋保護に関するドキュメンタリーを見ましたか？", "Did you see the documentary about ocean conservation?"),
    ("雨のため屋外音楽イベントが中止されました。", "The outdoor music event was cancelled due to rain."),
    ("地元の動物シェルターでボランティアを始めたいです。", "I want to start volunteering at the local animal shelter."),
]

EDUCATION_PAIRS = [
    ("教授が今週たくさんの宿題を出しました。", "The professor assigned a lot of homework this week."),
    ("来週に予定されている試験のために勉強しなければなりません。", "I need to study for the exam scheduled for next week."),
    ("図書館は今夜十時まで開いています。", "The library is open until ten o'clock tonight."),
    ("このコンセプトをもう一度説明してもらえますか？", "Can you explain this concept to me again?"),
    ("研究論文は学期末に提出です。", "The research paper is due at the end of the semester."),
    ("この章を理解するのに苦労しています。", "I am having trouble understanding this chapter."),
    ("グループプロジェクトの締め切りが近づいています。", "The group project deadline is approaching fast."),
    ("期末試験のためにスタディグループを作りましょう。", "Let's form a study group for the final exam."),
    ("奨学金の申請には推薦状が必要です。", "The scholarship application requires a letter of recommendation."),
    ("昨日機械学習についてのオンラインセミナーに参加しました。", "I attended an online seminar about machine learning yesterday."),
    ("卒業式は来春に予定されています。", "The graduation ceremony is scheduled for next spring."),
    ("来学期の授業を選ばなければなりません。", "I need to choose my courses for next semester."),
    ("インターンシップの申請締め切りは明日の朝です。", "The internship application deadline is tomorrow morning."),
    ("人工知能についての論文を書いています。", "I am writing a thesis on artificial intelligence."),
    ("新しいカリキュラムは来年度から始まります。", "The new curriculum starts from the next academic year."),
    ("来年留学するための奨学金を受けました。", "I received a scholarship to study abroad next year."),
    ("昨日の授業のノートを共有してもらえますか？", "Can you share your notes from the lecture yesterday?"),
    ("大学のキャンパスは秋にとても美しいです。", "The university campus is very beautiful in autumn."),
    ("来月コーディングコンペティションに参加します。", "I am participating in a coding competition next month."),
    ("教授のオフィスアワーは火曜日の午後です。", "The professor's office hours are on Tuesday afternoons."),
    ("クイズに失敗したので理解を深める必要があります。", "I failed the quiz and need to improve my understanding."),
    ("オンライン学習プラットフォームには素晴らしいリソースがあります。", "The online learning platform has great resources."),
    ("三日間プレゼンに取り組んでいます。", "I have been working on my presentation for three days."),
    ("今日の講義のゲストスピーカーはとても洞察力がありました。", "The guest speaker at today's lecture was very insightful."),
    ("今朝やっと研究論文を提出しました。", "I finally submitted my research paper this morning."),
    ("このコースは数学から哲学までのトピックをカバーしています。", "This course covers topics from mathematics to philosophy."),
    ("英語のスピーキングスキルを向上させる必要があります。", "I need to improve my English speaking skills."),
    ("試験の結果は来週発表されます。", "The exam results will be announced next week."),
    ("専攻をデータサイエンスに変更しようかと思っています。", "I am thinking of changing my major to data science."),
    ("学生会が文化祭を企画しています。", "The student council is organizing a cultural festival."),
    ("クラスメートと哲学について議論するのが好きです。", "I enjoy discussing philosophy with my classmates."),
    ("教授が会議のため授業をキャンセルしました。", "The professor cancelled class because of a conference."),
    ("補習授業に参加しています。", "I have been attending extra tutoring sessions."),
    ("課題にはリサーチと分析の両方が必要です。", "The assignment requires both research and analysis."),
    ("図書館から教科書を借りる必要があります。", "I need to borrow a textbook from the library."),
    ("語学ラボは毎日練習のために利用できます。", "The language lab is available for practice every day."),
    ("プログラミングの課題に苦労しています。", "I am struggling with the programming assignment."),
    ("理科博物館への校外学習は来週の金曜日です。", "The field trip to the science museum is next Friday."),
    ("下書きのエッセイに役立つコメントをもらいました。", "I received helpful comments on my draft essay."),
    ("創作文章の選択科目はとても人気があります。", "The elective course on creative writing is very popular."),
    ("締め切りまでに授業を登録しなければなりません。", "I need to register for courses before the deadline."),
    ("ヨーロッパの留学プログラムは面白そうです。", "The study abroad program in Europe sounds interesting."),
    ("一回目の挑戦で運転の学科試験に合格しました。", "I passed my driving theory test on the first attempt."),
    ("新入生オリエンテーションは明日の朝です。", "The new student orientation is tomorrow morning."),
    ("図書館でたくさんの時間を過ごしています。", "I have been spending a lot of time in the library."),
    ("授業で見たドキュメンタリーはとても考えさせられました。", "The documentary we watched in class was very thought-provoking."),
    ("次の提出前にエッセイの書き方を改善する必要があります。", "I need to improve my essay writing before the next submission."),
    ("プレゼンテーションのワークショップはとても役に立ちました。", "The workshop on public speaking was really helpful."),
    ("今学期研究助成金に申請しています。", "I am applying for a research grant this semester."),
    ("期末試験は学期全体の内容を含みます。", "The final exam covers everything from the whole semester."),
    ("卒業後に修士号を取得したいです。", "I want to pursue a master's degree after graduation."),
]

HEALTH_PAIRS = [
    ("最近少し体調が優れません。", "I have been feeling a bit under the weather lately."),
    ("一日中十分な水を飲むようにしてください。", "Please make sure to drink plenty of water throughout the day."),
    ("医者と検診の予約をしなければなりません。", "I need to schedule a checkup with my doctor."),
    ("定期的に運動していてとても元気になっています。", "I have been exercising regularly and feeling much better."),
    ("ジムは平日の朝六時に開きます。", "The gym opens at six in the morning on weekdays."),
    ("少なくとも八時間の睡眠を取るようにしています。", "I have been trying to get at least eight hours of sleep."),
    ("バランスの良い食事は健康にとても重要です。", "Eating a balanced diet is very important for your health."),
    ("風邪から回復中なので休息が必要です。", "I am recovering from a cold so I need rest."),
    ("春は花粉症がひどいです。", "My allergies are really bad during spring."),
    ("毎朝ビタミンを飲み始めました。", "I started taking vitamins every morning."),
    ("医者から糖分の摂取を減らすよう勧められました。", "The doctor recommended that I reduce my sugar intake."),
    ("ジョギングを始めてから体力がついてきました。", "I have been feeling more energetic since I started jogging."),
    ("精神的な健康は身体的な健康と同じくらい重要です。", "Mental health is just as important as physical health."),
    ("ジムで運動中に筋肉を痛めてしまいました。", "I pulled a muscle while exercising at the gym."),
    ("薬剤師が頭痛にこの薬を推薦しました。", "The pharmacist recommended this medicine for my headache."),
    ("よく眠れるようにハーブティーを飲んでいます。", "I have been drinking herbal tea to help me sleep better."),
    ("運動前のストレッチは怪我の予防になります。", "Stretching before exercise helps prevent injuries."),
    ("休んでストレスレベルを下げる必要があります。", "I need to take a break and reduce my stress levels."),
    ("歯医者の予約は木曜日の朝です。", "The dentist appointment is on Thursday morning."),
    ("フィットネスアプリで歩数を記録しています。", "I have been tracking my steps with a fitness app."),
    ("瞑想で気持ちがとても落ち着きました。", "Meditation has helped me feel much calmer."),
    ("食事と一緒に薬を飲む必要があります。", "I need to take my medication with food."),
    ("定期的な運動は健康的な体重の維持に役立ちます。", "Regular exercise helps maintain a healthy weight."),
    ("長時間座っていて腰痛があります。", "I have been having back pain from sitting too long."),
    ("今月病院で無料の健康診断を実施しています。", "The hospital is offering free health screenings this month."),
    ("就寝前はカフェインを控えるべきです。", "I should cut down on caffeine before bed."),
    ("夕食後の散歩は消化を助けます。", "Taking a walk after dinner aids digestion."),
    ("食事を変えてから五キロ体重が減りました。", "I have lost five kilograms since I changed my diet."),
    ("ジムのインストラクターが新しいトレーニングルーティンを勧めました。", "The gym instructor recommended a new workout routine."),
    ("特に夏は水分を補給することが大切です。", "Staying hydrated is especially important in summer."),
    ("膝の理学療法のセッションを予約する必要があります。", "I need to book a physiotherapy session for my knee."),
    ("新鮮な果物と野菜は良い健康に不可欠です。", "Fresh fruits and vegetables are essential for good health."),
    ("腹式呼吸の練習をしています。", "I have been practicing deep breathing exercises."),
    ("最近病院の待ち時間がとても長いです。", "The hospital wait times have been very long recently."),
    ("よく眠れて気分がとても良くなりました。", "I am feeling much better after a good night's sleep."),
    ("就寝前のスクリーンタイムを減らす必要があります。", "I need to reduce my screen time before bedtime."),
    ("自然の中を散歩すると気持ちがとてもすっきりします。", "Going for walks in nature really helps clear my mind."),
    ("家でより健康的な食事を作り始めました。", "I have started cooking healthier meals at home."),
    ("職場でのフィットネスチャレンジはとてもモチベーションが上がります。", "The fitness challenge at work has been very motivating."),
    ("職場でとてもストレスを感じています。", "I have been experiencing a lot of stress at work."),
    ("仕事中に短い休憩を取ると生産性が上がります。", "Taking short breaks during work improves productivity."),
    ("定期的に血圧を測るべきです。", "I should get my blood pressure checked regularly."),
    ("早く寝て気持ちよく目覚めています。", "I have been sleeping early and waking up refreshed."),
    ("スポーツセンターのプールはいつも清潔です。", "The swimming pool at the sports center is always clean."),
    ("食事について栄養士に相談する必要があります。", "I need to see a nutritionist about my diet."),
    ("地元のランニングクラブは毎週日曜の朝に集まります。", "The local running club meets every Sunday morning."),
    ("毎日の食事にタンパク質を多く加えています。", "I have been adding more protein to my daily meals."),
    ("マッサージを受けて筋肉の緊張がほぐれました。", "Getting a massage helped relieve my muscle tension."),
    ("猫アレルギーですがそれでも猫が大好きです。", "I am allergic to cats but I love them anyway."),
    ("朝の運動の後はとても元気になります。", "I feel so much more energized after morning exercise."),
    ("砂糖をやめるのは大変でしたが大きな変化がありました。", "Quitting sugar was hard but it made a big difference."),
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
        for ja, en in pairs:
            flat.append({"topic": topic, "ja": ja, "en": en})
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
    mp3  = await tts_bytes(pair["ja"], VOICE_JA)
    wav, duration = _mp3_to_wav(mp3)

    filename = f"ja_{idx:03d}.wav"
    filepath = AUDIO_OUT / filename
    filepath.write_bytes(wav)

    return {
        "id":                      f"ja_{idx:03d}",
        "audio_file":              str(filepath),
        "topic":                   pair["topic"],
        "japanese_transcription":  pair["ja"],
        "english_translation":     pair["en"],
        "duration_seconds":        round(duration, 2),
    }


async def main():
    pairs = get_all_pairs()
    samples = []
    for i, pair in enumerate(pairs, start=1):
        print(f"  [{i}/{N_SAMPLES}] ja_{i:03d}", flush=True)
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
