# 設定リファレンス

エージェントの挙動は環境変数（または `.env` ファイル）で制御します。

| 変数                  | 既定値                        | 説明                                                   |
| --------------------- | ----------------------------- | ------------------------------------------------------ |
| `AGENT_PROVIDER`      | `openai`                      | `openai` または `openrouter`                           |
| `AGENT_BASE_URL`      | プロバイダー既定値            | OpenAI 互換ゲートウェイのエンドポイント上書き          |
| `AGENT_API_KEY`       | —                             | 汎用 API キー（優先）                                   |
| `OPENAI_API_KEY`      | —                             | プロバイダーが openai のときのフォールバックキー       |
| `OPENROUTER_API_KEY`  | —                             | プロバイダーが openrouter のときのフォールバックキー   |
| `AGENT_MODEL`         | `gpt-4o-mini` ほか            | モデル名                                               |
| `AGENT_TEMPERATURE`   | `0.7`                         | サンプリング温度                                       |
| `AGENT_MAX_STEPS`     | `10`                          | 1ターンあたりのモデル/ツール往復の上限                 |
| `AGENT_SYSTEM_PROMPT` | 組み込み                      | システムプロンプトの上書き                             |
| `AGENT_KNOWLEDGE_DIR` | `knowledge`                   | ナレッジ Markdown を読み込むディレクトリ               |

## ナレッジディレクトリ

`AGENT_KNOWLEDGE_DIR` で指定したディレクトリ（既定 `knowledge`）配下の
Markdown / テキストファイルを起動時に読み込みます。ファイルが見つかった場合のみ
`knowledge_search` ツールが有効になります。ディレクトリが存在しない場合でも
エラーにはならず、エージェントはナレッジなしで動作します。
