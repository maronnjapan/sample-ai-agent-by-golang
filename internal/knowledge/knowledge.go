// Package knowledge はエージェントが回答時に参照するナレッジ（知識）を読み込み、
// チャンクに分割し、関連チャンクを検索する小さなインメモリのナレッジベースを提供します。
//
// ドキュメントのチャンク分割には langchaingo の標準コンポーネント
// （textsplitter / schema.Document）を利用しています。これにより、独自の
// チャンク分割ロジックを持たず、RAG（Retrieval-Augmented Generation /
// 検索拡張生成）でよく使われる標準的な前処理パイプラインに乗せています。
// ファイルの読み込みは標準ライブラリで行います（重い documentloaders の
// 依存を避けるため）。
//
// 検索は外部のベクトルデータベースや埋め込み API を必要としない、軽量な
// キーワード（語の出現頻度）ベースのスコアリングで行います。サンプルとして
// 自己完結し、決定論的にテストできることを優先した設計です。
package knowledge

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
)

// 読み込み対象とするファイル拡張子です。
var knowledgeExtensions = map[string]bool{
	".md":       true,
	".markdown": true,
	".txt":      true,
}

const (
	// defaultChunkSize はチャンク1つあたりの目安サイズ（ルーン数）です。
	// 検索の粒度を細かくするため、langchaingo の既定値より小さめにしています。
	defaultChunkSize = 700
	// defaultChunkOverlap は隣接チャンク間で重複させるルーン数です。
	// 文や見出しの境界で文脈が途切れるのを緩和します。
	defaultChunkOverlap = 100
)

// Store は読み込み済みのナレッジチャンクをメモリ上に保持します。
// 構築後は読み取り専用として扱われ、複数ゴルーチンからの Search 呼び出しに
// 対して安全です。
type Store struct {
	// chunks は分割済みの全ナレッジチャンクです。各チャンクは
	// Metadata["source"] に元ファイルの相対パスを保持します。
	chunks []schema.Document
}

// Load は dir 配下（再帰的）の Markdown / テキストファイルを読み込み、
// langchaingo の textsplitter でチャンクに分割した Store を返します。
//
// dir が存在しない場合はエラーではなく空の Store を返します。これにより、
// ナレッジディレクトリを用意していない利用者でもエージェントをそのまま
// 起動できます。読み込めたファイルが1つもない場合も空の Store になります。
func Load(dir string) (*Store, error) {
	store := &Store{}

	info, err := os.Stat(dir)
	if err != nil {
		if os.IsNotExist(err) {
			// ディレクトリ未作成は正常系として扱い、空の Store を返します。
			return store, nil
		}
		return nil, fmt.Errorf("knowledge: stat %q: %w", dir, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("knowledge: %q is not a directory", dir)
	}

	// チャンク分割器: langchaingo の再帰文字分割を使います。長さは
	// ルーン数で数え、日本語混在のドキュメントでも妥当に分割されるようにします。
	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(defaultChunkSize),
		textsplitter.WithChunkOverlap(defaultChunkOverlap),
		textsplitter.WithLenFunc(utf8.RuneCountInString),
	)

	walkErr := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if !knowledgeExtensions[strings.ToLower(filepath.Ext(path))] {
			return nil
		}

		rel, relErr := filepath.Rel(dir, path)
		if relErr != nil {
			rel = path
		}

		docs, loadErr := loadFile(path, rel, splitter)
		if loadErr != nil {
			return loadErr
		}
		store.chunks = append(store.chunks, docs...)
		return nil
	})
	if walkErr != nil {
		return nil, fmt.Errorf("knowledge: load %q: %w", dir, walkErr)
	}

	return store, nil
}

// loadFile は1ファイルを読み込み、textsplitter で分割して source メタデータを
// 付与したチャンク列を返します。
func loadFile(path, source string, splitter textsplitter.TextSplitter) ([]schema.Document, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("knowledge: read %q: %w", path, err)
	}

	// langchaingo の textsplitter でチャンクに分割します。
	chunks, err := splitter.SplitText(string(content))
	if err != nil {
		return nil, fmt.Errorf("knowledge: split %q: %w", path, err)
	}

	// 各チャンクを schema.Document に変換し、出典（ソース）を記録しておきます。
	// これにより回答時に引用元を提示できます。
	docs := make([]schema.Document, 0, len(chunks))
	for _, chunk := range chunks {
		if strings.TrimSpace(chunk) == "" {
			continue
		}
		docs = append(docs, schema.Document{
			PageContent: chunk,
			Metadata:    map[string]any{"source": source},
		})
	}
	return docs, nil
}

// Len は保持しているチャンク数を返します。
func (s *Store) Len() int { return len(s.chunks) }

// Sources は読み込んだソースファイルの一覧をソート順で返します。
// 起動時のバナー表示などに利用できます。
func (s *Store) Sources() []string {
	seen := map[string]bool{}
	for _, c := range s.chunks {
		if src, ok := c.Metadata["source"].(string); ok {
			seen[src] = true
		}
	}
	out := make([]string, 0, len(seen))
	for src := range seen {
		out = append(out, src)
	}
	sort.Strings(out)
	return out
}

// Search は query に関連するチャンクをスコアの高い順に最大 k 件返します。
// スコアは query を語に分割し、各語がチャンク内に出現する回数の合計です
// （簡易な語頻度ベース）。スコアが 0 のチャンクは結果に含めません。
// 返り値の各 Document の Score フィールドに算出スコアを格納します。
func (s *Store) Search(query string, k int) []schema.Document {
	if k <= 0 {
		k = 4
	}
	terms := tokenize(query)
	if len(terms) == 0 || len(s.chunks) == 0 {
		return nil
	}

	type scored struct {
		doc   schema.Document
		score float32
	}
	results := make([]scored, 0, len(s.chunks))
	for _, c := range s.chunks {
		lower := strings.ToLower(c.PageContent)
		var score float32
		for _, term := range terms {
			if n := strings.Count(lower, term); n > 0 {
				// 語が長いほど具体的とみなし、わずかに重み付けします。
				score += float32(n) * weightForTerm(term)
			}
		}
		if score <= 0 {
			continue
		}
		doc := c
		doc.Score = score
		results = append(results, scored{doc: doc, score: score})
	}

	// スコア降順、同点はチャンクが短いもの（より焦点が絞られたもの）を優先します。
	sort.SliceStable(results, func(i, j int) bool {
		if results[i].score != results[j].score {
			return results[i].score > results[j].score
		}
		return len(results[i].doc.PageContent) < len(results[j].doc.PageContent)
	})

	if len(results) > k {
		results = results[:k]
	}
	out := make([]schema.Document, len(results))
	for i, r := range results {
		out[i] = r.doc
	}
	return out
}

// weightForTerm は語の長さに応じた軽い重みを返します。短い一般的な語より
// 長い具体的な語を優先するための単純なヒューリスティックです。
func weightForTerm(term string) float32 {
	n := utf8.RuneCountInString(term)
	switch {
	case n >= 6:
		return 2.0
	case n >= 3:
		return 1.5
	default:
		return 1.0
	}
}

// tokenize は検索クエリを小文字の語へ分割します。英数字の連なりを1語とし、
// それ以外（空白・記号・日本語の区切りなど）を境界として扱います。日本語など
// 空白で区切られない言語では、連続する非 ASCII 文字列がまとめて1語になります。
func tokenize(s string) []string {
	s = strings.ToLower(s)
	fields := strings.FieldsFunc(s, func(r rune) bool {
		if r > unicode.MaxASCII {
			// 非 ASCII（日本語など）は語の一部として残します。
			return false
		}
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	// 1文字の ASCII 語（"a", "i" など）はノイズになりやすいので除外します。
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if utf8.RuneCountInString(f) == 1 && f[0] <= unicode.MaxASCII {
			continue
		}
		out = append(out, f)
	}
	return out
}
