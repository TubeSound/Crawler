
# HTML構造解析・分類

---

# 1. 目的

* 保存済みHTML構造の特徴を定量化して、クラスタ分類をおこなう。
* 標準的な構成から外れた構成かどうかを検出
    → general-purpose-crawlerのセクション分割戦略(split_sections)に合うか、必要なら追加


# 2. 処理フロー

1. meta.json一覧取得
2. 対応するHTMLを読み込み
3. メトリックを計算
4. 構造を抽出
5. CSV / JSONL出力


# 3. メトリック（評価指標）

## 基本
* url_depth

## 文字数
* archived_body_length: HTMLのbody文字数
* body_text_length: HTMLのbodyテキスト長
* total_text_length: テキスト文字数
* link_text_ratio: リンク内テキスト文字数 / 全テキスト文字数


## タグ数
* link_count: hrefタグ数
* section_count: sectionタグ数
* h1_count: h1タグ数
* h2_count: h2タグ数
* h3_count: h3タグ数
* div_count: divタグ数
* li_count: liタグ数

## 比率

* main_text_ratio: main領域のテキスト文字数 / body_text_length
* section_text_ratio: section内テキスト文字数 / body_text_length
* section_with_heading_ratio: 見出し(h1/h2/h3)を持つsection数 / section総数

## 意味分割推定

* dominant_segmentation_type: ページ内で最も支配的なコンテンツ分割単位
候補ごとに「テキスト量」を測ってbody全体に対する比率の最も大きいものを採用

    **divはネストを除去する必要がある**

分類：

```text
section_dominant
heading_dominant
div_dominant
li_dominant
```

## JS動的UIメトリクス

* hidden_element_count: classにhidden/is-hidden/u-hidden/visually-hiddenを含む
* accordion_like_count: アコーディオンや折りたたみ構造の数
```
details タグ
summary タグ
aria-expanded
aria-controls
class/id に accordion, collapse, expand, toggle, faq
ボタン直後に hidden要素がある
data-bs-toggle="collapse" など
```

* modal_like_count: モーダルダイアログ数

```
dialog タグ
role="dialog"
role="alertdialog"
class/id に modal, dialog, popup, lightbox, drawer, overlay
aria-modal="true"
```

* js_event_handler_count: HTML属性のJSイベントハンドラの数
```
onclick
ondblclick
onchange
oninput
onsubmit
onfocus
onblur
onmouseenter
onmouseleave
onmouseover
onmouseout
onkeydown
onkeyup
onload
```

# 課題　

## FAQのアンサーページ判定

FAQページでQとAがペアで取得する必要があるが、取得には個別対応が必要そう。
アンサーページかどうかの判定をおこなたい。


次の例ではURLパスからわかりやすい例

### docomo/faq
https://www.docomo.ne.jp/faq/detail?categoryId=12374&faqId=145188

### goo id
https://point.d.goo.ne.jp/j/faq/110
