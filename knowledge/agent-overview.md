# このエージェントの概要

`sample-ai-agent-by-golang` は Go で書かれた小さな AI エージェントです。
古典的な Reason-Act（推論と行動）ループを実装しています。エージェントは
会話を大規模言語モデル（LLM）へ送信し、モデルはツール（function calling）の
実行を要求できます。エージェントはそのツールを実行して結果をモデルへ返し、
モデルが最終的な回答を生成するまでこのループを繰り返します。

LLM との実際の通信は langchaingo ライブラリが担当するため、同一バイナリで
OpenAI・OpenRouter・その他の OpenAI 互換ゲートウェイ（Azure OpenAI, Groq,
Together, ローカルの llama.cpp サーバなど）を、設定変更だけで対象にできます。

## アーキテクチャ

- `cmd/agent`: CLI。REPL モードとワンショットモード、ツールトレース表示。
- `internal/config`: 環境変数・.env からの設定読み込みとプロバイダー既定値。
- `internal/llm`: langchaingo を用いたチャットクライアントとエージェント向け型。
- `internal/agent`: Reason-Act ループ（モデルとツールの往復）とイベント配信。
- `internal/tools`: Tool インターフェース、レジストリ、組み込みツール。
- `internal/knowledge`: ナレッジの読み込み・チャンク分割・検索（RAG）。

## 組み込みツール

- `calculator`: 算術式を正確に評価します。
- `current_time`: タイムゾーン対応で現在時刻を返します。
- `http_get`: 読み取り専用の HTTP GET でウェブページを取得します。
- `knowledge_search`: このナレッジベースから関連情報を検索します。
