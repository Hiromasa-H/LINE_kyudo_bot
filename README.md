# LINE 弓道 bot

弓道姿勢推定用のLINE bot。まだ開発中なので、長い動画や、条件に則さない動画を送った場合、正常に動作しない可能性があります。

<img src="md_resources/chat_sample.png" width="300px" alt="Image">


## 使い方

- 当botを友達登録する
  - 利用希望の方はご連絡ください。
- 正面から撮られた射型の動画を送信する。※この際、以下の点に気を付けてください
  - 動画は短く、30秒程度に納めてください
  - なるべく他の人が写っていないようにしてください
  - 全身が写っているようにしてください

## 開発環境

- `bash setup.sh`で必要なパッケージをインストールし、必要なファイルが作成できます。
- `python3 app.py`で起動します。
- `ngrok`を使って、外部からアクセスできるようにしてください。
  - `ngrok http 5000`で起動します。
  - `https://xxxxxxxx.ngrok.io/callback`を、LINE DevelopersのWebhook URLに設定してください。
  - また、.envファイルの`CURRENT_URL`にも設定してください。