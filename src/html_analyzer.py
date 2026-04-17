from __future__ import annotations

import argparse
import csv
import json
import math
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

try:
    from html_fetcher import HtmlArchiveStore
except ModuleNotFoundError:
    from .html_fetcher import HtmlArchiveStore


WHITESPACE_RE = re.compile(r"\s+")
NOISE_CLASS_RE = re.compile(
    r"(nav|menu|breadcrumb|footer|header|sidebar|pager|pagination|sns|social|share|banner|ad|advert|modal|dialog)",
    re.IGNORECASE,
)
ACCORDION_KEYWORDS = (
    "accordion", "collapse", "collapsed", "expand", "toggle", "faq", "tab"
)
MODAL_KEYWORDS = (
    "modal", "popup", "dialog", "overlay", "lightbox", "cookie", "consent"
)
EVENT_HANDLER_ATTRS = (
    "onclick", "ondblclick", "onchange", "oninput", "onsubmit", "onload",
    "onmouseover", "onmouseout", "onfocus", "onblur", "onkeydown", "onkeyup",
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    if m == 0:
        return 0.0
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance) / m


def count_url_depth(url: str) -> int:
    path = urlparse(url).path
    return len([p for p in path.split("/") if p])


def get_tag_text(tag: Tag | BeautifulSoup) -> str:
    return normalize_text(tag.get_text(" ", strip=True))


def get_tag_text_len(tag: Tag | BeautifulSoup) -> int:
    return len(get_tag_text(tag))


def has_heading_descendant(tag: Tag) -> bool:
    return tag.find(["h1", "h2", "h3", "h4", "h5", "h6"]) is not None


def get_tag_class_id_text(tag: Tag) -> str:
    classes = tag.get("class", [])
    class_text = " ".join(classes) if isinstance(classes, list) else str(classes)
    id_text = str(tag.get("id", ""))
    return f"{class_text} {id_text}".strip()


def is_noise_like(tag: Tag) -> bool:
    if tag.name in {"header", "footer", "nav", "aside"}:
        return True
    return bool(NOISE_CLASS_RE.search(get_tag_class_id_text(tag)))


def is_non_content_tag(tag: Tag) -> bool:
    return tag.name in {"script", "style", "noscript", "svg", "canvas", "iframe"}


def has_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def tag_meta_text(tag: Tag) -> str:
    return get_tag_class_id_text(tag).lower()


def is_hidden_like(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True

    if str(tag.get("aria-hidden", "")).lower() == "true":
        return True

    style = str(tag.get("style", "")).replace(" ", "").lower()
    if any(token in style for token in ("display:none", "visibility:hidden", "opacity:0")):
        return True

    meta = tag_meta_text(tag)
    return any(token in meta for token in ("hidden", "collapse", "collapsed", "closed"))


def is_accordion_like(tag: Tag) -> bool:
    meta = tag_meta_text(tag)
    if has_any_keyword(meta, ACCORDION_KEYWORDS):
        return True

    if tag.name in {"details", "summary"}:
        return True

    aria_expanded = str(tag.get("aria-expanded", "")).lower()
    return aria_expanded in {"true", "false"}


def is_modal_like(tag: Tag) -> bool:
    meta = tag_meta_text(tag)
    if has_any_keyword(meta, MODAL_KEYWORDS):
        return True

    role = str(tag.get("role", "")).lower()
    if role in {"dialog", "alertdialog"}:
        return True

    style = str(tag.get("style", "")).replace(" ", "").lower()
    return "position:fixed" in style or "position:sticky" in style


def count_event_handler_attrs(soup: BeautifulSoup) -> int:
    count = 0
    for tag in soup.find_all(True):
        count += sum(1 for attr in EVENT_HANDLER_ATTRS if tag.has_attr(attr))
    return count


def count_script_ui_keywords(soup: BeautifulSoup) -> int:
    keywords = (
        "addeventlistener", "classlist.toggle", "classlist.add", "classlist.remove",
        "style.display", "display='none'", 'display="none"', "aria-expanded",
    )
    count = 0
    for script in soup.find_all("script"):
        script_text = script.get_text(" ", strip=True).lower()
        if not script_text:
            continue
        count += sum(1 for keyword in keywords if keyword in script_text)
    return count


def compute_js_dynamic_ui_score(
    hidden_element_count: int,
    accordion_like_count: int,
    modal_like_count: int,
    event_handler_count: int,
    script_ui_keyword_count: int,
) -> int:
    score = 0
    if hidden_element_count >= 3:
        score += 1
    if accordion_like_count >= 1:
        score += 1
    if modal_like_count >= 1:
        score += 1
    if event_handler_count >= 3:
        score += 1
    if script_ui_keyword_count >= 2:
        score += 1
    return score


def get_body_text_length(html: str, parser: str = "html.parser") -> int:
    soup = BeautifulSoup(html, parser)
    body = soup.body
    if body is None:
        return 0
    return get_tag_text_len(body)


def is_empty_body(html: str, parser: str = "html.parser") -> bool:
    soup = BeautifulSoup(html, parser)
    body = soup.body
    if body is None:
        return True
    return get_tag_text_len(body) == 0


class PlaywrightHtmlRenderer:
    def __init__(self, wait_timeout_ms: int = 15000) -> None:
        self.wait_timeout_ms = wait_timeout_ms
        self.playwright = None
        self.browser = None
        self.context = None

    def __enter__(self) -> PlaywrightHtmlRenderer:
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        self.context = self.browser.new_context(
            locale="ja-JP",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 1400},
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def render(self, url: str) -> str:
        if self.context is None:
            raise RuntimeError("PlaywrightHtmlRenderer is not started.")

        page = self.context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)

        try:
            page.wait_for_load_state("networkidle", timeout=self.wait_timeout_ms)
        except PlaywrightTimeoutError:
            pass

        try:
            page.wait_for_function(
                """() => {
                    const body = document.body;
                    if (!body) return false;
                    const text = (body.innerText || '').trim();
                    return text.length > 0 && (
                        document.querySelector('main, section, h1, h2, h3') ||
                        document.body.childElementCount > 0
                    );
                }""",
                timeout=self.wait_timeout_ms,
            )
        except PlaywrightTimeoutError:
            pass

        html = page.content()
        page.close()
        return html


def render_html_with_playwright(url: str, wait_timeout_ms: int = 15000) -> str:
    with PlaywrightHtmlRenderer(wait_timeout_ms=wait_timeout_ms) as renderer:
        return renderer.render(url)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class HtmlStructureMetrics:
    url: Optional[str]
    url_depth: int
    title: str
    total_text_length: int
    body_text_length: int
    empty_body: int
    link_count: int
    link_text_ratio: float
    section_count: int
    div_count: int
    li_count: int
    main_text_ratio: float
    section_text_ratio: float
    h1_count: int
    h2_count: int
    h3_count: int
    section_with_heading_ratio: float
    dominant_segmentation_type: str
    hidden_element_count: int
    accordion_like_count: int
    modal_like_count: int
    js_event_handler_count: int
    script_ui_keyword_count: int
    js_dynamic_ui_score: int


@dataclass
class AnalyzerJob:
    html_dir: str
    output_csv: str
    output_jsonl: str
    rendered_html_dir: str
    limit: int | None = None
    render_mode: str = "fast"


# 普段はここを書き換えるだけで解析対象を切り替える。
# render_mode:
#   off       = 保存済みHTMLだけを解析する
#   fast      = bodyが空のページだけPlaywrightでレンダリングする
#   precision = 全ページをPlaywrightでレンダリングする
ACTIVE_JOB = "ocn_support" #"docomo_faq"
USE_CLI_ARGS = False

ANALYZER_JOBS: dict[str, AnalyzerJob] = {
    "docomo_faq": AnalyzerJob(
        html_dir="output/docomo_faq/html",
        output_csv="output/docomo_faq/feature/docomo_faq_feature.csv",
        output_jsonl="output/docomo_faq/feature/docomo_faq_feature.jsonl",
        rendered_html_dir="output/docomo_faq/rendered_html",
        limit=None,
        render_mode="fast",
    ),
    "point_goo": AnalyzerJob(
        html_dir="output/point_goo/rendered_html",
        output_csv="output/point_goo/feature/point_goo_feature.csv",
        output_jsonl="output/point_goo/feature/point_goo_feature.jsonl",
        rendered_html_dir="output/point_goo/rendered_html",
        limit=None,
        render_mode="off",
    ),
    "gooID": AnalyzerJob(
        html_dir="output/gooID/rendered_html",
        output_csv="output/gooID/feature/gooID_feature.csv",
        output_jsonl="output/gooID/feature/gooID_feature.jsonl",
        rendered_html_dir="output/gooID/rendered_html",
        limit=None,
        render_mode="off",
    ),
    "ocn_support": AnalyzerJob(
        html_dir="output/ocn_support/html",
        output_csv="output/ocn_support/feature/ocn_support_feature.csv",
        output_jsonl="output/ocn_support/feature/ocn_support_feature.jsonl",
        rendered_html_dir="output/ocn_support/rendered_html",
        limit=None,
        render_mode="off",
    ),
}


class AnalyzeHtmlStructure:
    def __init__(
        self,
        parser: str = "html.parser",
        min_block_ratio: float = 0.01,
        div_min_text_len: int = 80,
    ) -> None:
        self.parser = parser
        self.min_block_ratio = min_block_ratio
        self.div_min_text_len = div_min_text_len

    # --- 修正：analyze() を置き換え ---

    def analyze(self, html: str, url: Optional[str] = None):
        body_text_length = get_body_text_length(html, parser=self.parser)
        empty_body = int(body_text_length == 0)

        soup = BeautifulSoup(html, self.parser)
        self._remove_non_content_nodes(soup)

        # ★追加
        section_outline = self._extract_section_outline(soup)

        total_text_length = get_tag_text_len(soup)
        title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""

        links = soup.find_all("a")
        link_text_length = sum(get_tag_text_len(a) for a in links)

        section_tags = soup.find_all("section")
        main_text_length = sum(get_tag_text_len(tag) for tag in soup.find_all("main"))
        section_text_length_total = sum(get_tag_text_len(tag) for tag in section_tags)

        section_with_heading_count = sum(
            1 for tag in section_tags if has_heading_descendant(tag)
        )

        hidden_element_count = sum(1 for tag in soup.find_all(True) if is_hidden_like(tag))
        accordion_like_count = sum(1 for tag in soup.find_all(True) if is_accordion_like(tag))
        modal_like_count = sum(1 for tag in soup.find_all(True) if is_modal_like(tag))

        js_event_handler_count = count_event_handler_attrs(soup)
        script_ui_keyword_count = count_script_ui_keywords(soup)

        js_dynamic_ui_score = compute_js_dynamic_ui_score(
            hidden_element_count=hidden_element_count,
            accordion_like_count=accordion_like_count,
            modal_like_count=modal_like_count,
            event_handler_count=js_event_handler_count,
            script_ui_keyword_count=script_ui_keyword_count,
        )

        metrics = HtmlStructureMetrics(
            url=url,
            url_depth=count_url_depth(url) if url else 0,
            title=title,
            total_text_length=total_text_length,
            body_text_length=body_text_length,
            empty_body=empty_body,
            link_count=len(links),
            link_text_ratio=safe_div(link_text_length, total_text_length),
            section_count=len(section_tags),
            div_count=len(soup.find_all("div")),
            li_count=len(soup.find_all("li")),
            main_text_ratio=safe_div(main_text_length, total_text_length),
            section_text_ratio=safe_div(section_text_length_total, total_text_length),
            h1_count=len(soup.find_all("h1")),
            h2_count=len(soup.find_all("h2")),
            h3_count=len(soup.find_all("h3")),
            section_with_heading_ratio=safe_div(
                section_with_heading_count,
                len(section_tags),
            ),
            dominant_segmentation_type=self._detect_dominant_segmentation_type(
                soup,
                total_text_length,
            ),
            hidden_element_count=hidden_element_count,
            accordion_like_count=accordion_like_count,
            modal_like_count=modal_like_count,
            js_event_handler_count=js_event_handler_count,
            script_ui_keyword_count=script_ui_keyword_count,
            js_dynamic_ui_score=js_dynamic_ui_score,
        )

        # ★変更：tupleで返す
        return metrics, section_outline
        
    # --- 追加：AnalyzeHtmlStructure 内にこの関数を追加 ---

    def _extract_section_outline(self, soup: BeautifulSoup) -> str:
        outlines: list[list[str]] = []

        sections = soup.find_all("section")

        # sectionが無ければ全体を1セクション扱い
        if not sections:
            sections = [soup]

        for section in sections:
            headings: list[str] = []
            for tag in section.find_all(["h1", "h2", "h3"]):
                headings.append(tag.name)
            outlines.append(headings)

        return json.dumps(outlines, ensure_ascii=False)    

    def analyze_file(
        self,
        html_path: str | Path,
        url: Optional[str] = None,
    ) -> HtmlStructureMetrics:
        html = Path(html_path).read_text(encoding="utf-8")
        return self.analyze(html=html, url=url)

    def analyze_archived(
        self,
        store: HtmlArchiveStore,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ) -> HtmlStructureMetrics:
        if key:
            meta, html = store.load_by_key(key)
        elif url:
            meta, html = store.load_by_url(url)
        else:
            raise ValueError("url or key is required.")

        return self.analyze(html=html, url=meta.get("url", url))

    def _remove_non_content_nodes(self, soup: BeautifulSoup) -> None:
        for tag in soup.find_all(is_non_content_tag):
            tag.decompose()

    def _detect_dominant_segmentation_type(
        self,
        soup: BeautifulSoup,
        total_text_length: int,
    ) -> str:
        if total_text_length == 0:
            return "undivided"

        section_lengths = self._extract_section_lengths(soup, total_text_length)
        heading_lengths = self._extract_heading_segment_lengths(soup, total_text_length)
        div_lengths = self._extract_div_candidate_lengths(soup, total_text_length)
        li_lengths = self._extract_li_lengths(soup, total_text_length)

        score_map = {
            "section_dominant": self._calc_segmentation_score(
                section_lengths,
                total_text_length,
                prefer_uniform=True,
            ),
            "heading_dominant": self._calc_segmentation_score(
                heading_lengths,
                total_text_length,
                prefer_uniform=True,
            ),
            "div_dominant": self._calc_segmentation_score(
                div_lengths,
                total_text_length,
                prefer_uniform=False,
            ),
            "li_dominant": self._calc_segmentation_score(
                li_lengths,
                total_text_length,
                prefer_uniform=True,
            ),
        }

        best_label = max(score_map, key=score_map.get)
        best_score = score_map[best_label]

        if best_score < 0.20:
            return "undivided"

        sorted_scores = sorted(score_map.values(), reverse=True)
        if len(sorted_scores) >= 2 and abs(sorted_scores[0] - sorted_scores[1]) < 0.08:
            return "mixed"

        return best_label

    def _extract_section_lengths(
        self,
        soup: BeautifulSoup,
        total_text_length: int,
    ) -> list[int]:
        lengths: list[int] = []
        for section in soup.find_all("section"):
            text_len = get_tag_text_len(section)
            if safe_div(text_len, total_text_length) >= self.min_block_ratio:
                lengths.append(text_len)
        return lengths

    def _extract_heading_segment_lengths(
        self,
        soup: BeautifulSoup,
        total_text_length: int,
    ) -> list[int]:
        for heading_name in ["h2", "h1", "h3"]:
            headings = soup.find_all(heading_name)
            if len(headings) >= 2:
                return self._build_segments_by_heading(headings, total_text_length)
        return []

    def _build_segments_by_heading(
        self,
        headings: list[Tag],
        total_text_length: int,
    ) -> list[int]:
        segments: list[int] = []

        for idx, heading in enumerate(headings):
            texts: list[str] = [get_tag_text(heading)]
            current = heading.next_sibling
            stop_at = headings[idx + 1] if idx + 1 < len(headings) else None

            while current is not None and current is not stop_at:
                if isinstance(current, NavigableString):
                    txt = normalize_text(str(current))
                    if txt:
                        texts.append(txt)
                elif isinstance(current, Tag):
                    texts.append(get_tag_text(current))
                current = current.next_sibling

            seg_len = len(normalize_text(" ".join(texts)))
            if safe_div(seg_len, total_text_length) >= self.min_block_ratio:
                segments.append(seg_len)

        return segments

    def _extract_div_candidate_lengths(
        self,
        soup: BeautifulSoup,
        total_text_length: int,
    ) -> list[int]:
        lengths: list[int] = []

        for div in soup.find_all("div"):
            if is_noise_like(div):
                continue

            text_len = get_tag_text_len(div)
            if text_len < self.div_min_text_len:
                continue

            if safe_div(text_len, total_text_length) < self.min_block_ratio:
                continue

            child_div_lengths = [
                get_tag_text_len(child)
                for child in div.find_all("div", recursive=False)
                if not is_noise_like(child)
            ]
            if child_div_lengths and max(child_div_lengths) >= text_len * 0.7:
                continue

            link_text_len = sum(get_tag_text_len(a) for a in div.find_all("a"))
            if safe_div(link_text_len, text_len) > 0.85:
                continue

            lengths.append(text_len)

        return lengths

    def _extract_li_lengths(
        self,
        soup: BeautifulSoup,
        total_text_length: int,
    ) -> list[int]:
        lengths: list[int] = []
        for li in soup.find_all("li"):
            text_len = get_tag_text_len(li)
            if safe_div(text_len, total_text_length) >= self.min_block_ratio:
                lengths.append(text_len)
        return lengths

    def _calc_segmentation_score(
        self,
        lengths: list[int],
        total_text_length: int,
        prefer_uniform: bool,
    ) -> float:
        if not lengths or total_text_length == 0:
            return 0.0

        ratios = [safe_div(x, total_text_length) for x in lengths]
        coverage_ratio = sum(ratios)
        chunk_count_8_20 = sum(1 for r in ratios if 0.08 <= r <= 0.20)
        chunk_count_3_8 = sum(1 for r in ratios if 0.03 <= r < 0.08)
        largest_ratio = max(ratios)
        size_cv = coefficient_of_variation(lengths)

        score = 0.0
        score += coverage_ratio * 0.45
        score += min(chunk_count_8_20, 5) * 0.08
        score += min(chunk_count_3_8, 5) * 0.03
        score += min(len(lengths), 10) * 0.01

        if largest_ratio > 0.7:
            score -= 0.15

        if prefer_uniform and size_cv > 1.5:
            score -= 0.08

        return round(max(score, 0.0), 4)


def iter_archived_html(html_dir: str | Path, limit: int | None = None):
    html_dir = Path(html_dir)
    meta_paths = sorted(html_dir.glob("*.meta.json"))
    rendered_html_paths = sorted(html_dir.glob("*.rendered.html"))

    if not html_dir.exists():
        print(f"[WARN] html_dir not found: {html_dir}")
        return

    if not meta_paths and not rendered_html_paths:
        print(f"[WARN] no .meta.json files found in: {html_dir}")
        return

    if meta_paths:
        if limit is not None:
            meta_paths = meta_paths[:limit]

        for meta_path in meta_paths:
            key = meta_path.name.removesuffix(".meta.json")
            html_path = html_dir / f"{key}.html"

            if not html_path.exists():
                print(f"[SKIP] html not found: {html_path}")
                continue

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            html = html_path.read_text(encoding="utf-8")

            yield {
                "key": key,
                "meta_path": str(meta_path),
                "html_path": str(html_path),
                "meta": meta,
                "html": html,
            }
        return

    if limit is not None:
        rendered_html_paths = rendered_html_paths[:limit]

    for rendered_html_path in rendered_html_paths:
        key = rendered_html_path.name.removesuffix(".rendered.html")
        rendered_meta_path = html_dir / f"{key}.rendered.meta.json"
        archived_html_path = html_dir.parent / "html" / f"{key}.html"
        source_meta_path = html_dir.parent / "html" / f"{key}.meta.json"

        if source_meta_path.exists():
            source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
        else:
            source_meta = {
                "url": "",
                "fetched_at": "",
                "http_status": "",
                "content_type": "",
            }

        if not rendered_meta_path.exists():
            write_rendered_meta(
                path=rendered_meta_path,
                source_meta=source_meta,
                key=key,
                url=source_meta.get("url", ""),
                archived_html_path=str(archived_html_path) if archived_html_path.exists() else "",
                rendered_html_path=str(rendered_html_path),
                render_mode="cached",
            )

        meta = json.loads(rendered_meta_path.read_text(encoding="utf-8"))
        html = rendered_html_path.read_text(encoding="utf-8")

        yield {
            "key": key,
            "meta_path": str(rendered_meta_path),
            "html_path": str(rendered_html_path),
            "meta": meta,
            "html": html,
        }


def analyze_archived_html_dir(
    html_dir: str | Path,
    output_csv: str | Path,
    output_jsonl: str | Path,
    limit: int | None = None,
    render_mode: str = "fast",
    rendered_html_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    analyzer = AnalyzeHtmlStructure()
    records: list[dict[str, Any]] = []
    rendered_html_base = Path(rendered_html_dir) if rendered_html_dir else None
    if rendered_html_base:
        rendered_html_base.mkdir(parents=True, exist_ok=True)

    use_renderer = render_mode in {"fast", "precision"}
    renderer_context = PlaywrightHtmlRenderer() if use_renderer else None

    with (renderer_context if renderer_context else nullcontext()) as renderer:
        for item in iter_archived_html(html_dir, limit=limit):
            meta = item["meta"]
            url = meta.get("url", "")
            html = item["html"]
            analysis_source = "archived_html"
            analyzed_html_path = item["html_path"]
            rendered_meta_path = ""
            render_error = ""

            if item["meta_path"].endswith(".rendered.meta.json"):
                analysis_source = "playwright_rendered_cached"
                rendered_meta_path = item["meta_path"]

            archived_body_text_length = get_body_text_length(item["html"])
            archived_empty_body = int(archived_body_text_length == 0)
            should_render = False

            if render_mode == "precision" and url:
                should_render = True
            elif render_mode == "fast" and url and archived_empty_body:
                should_render = True

            if should_render and renderer:
                rendered_html_path = (
                    rendered_html_base / f"{item['key']}.rendered.html"
                    if rendered_html_base
                    else None
                )
                rendered_meta = (
                    rendered_html_base / f"{item['key']}.rendered.meta.json"
                    if rendered_html_base
                    else None
                )

                if rendered_html_path and rendered_html_path.exists():
                    html = rendered_html_path.read_text(encoding="utf-8")
                    analysis_source = "playwright_rendered_cached"
                    analyzed_html_path = str(rendered_html_path)
                    if rendered_meta and not rendered_meta.exists():
                        write_rendered_meta(
                            path=rendered_meta,
                            source_meta=meta,
                            key=item["key"],
                            url=url,
                            archived_html_path=item["html_path"],
                            rendered_html_path=str(rendered_html_path),
                            render_mode=render_mode,
                        )
                    rendered_meta_path = str(rendered_meta) if rendered_meta else ""
                    print(f"[RENDER CACHED] {url}")
                else:
                    if render_mode == "fast" and archived_empty_body:
                        print(f"[RENDER] archived HTML has empty body: {url}")
                    else:
                        print(f"[RENDER] {url}")
                    try:
                        html = renderer.render(url)
                        analysis_source = "playwright_rendered"

                        if rendered_html_path:
                            rendered_html_path.write_text(html, encoding="utf-8")
                            analyzed_html_path = str(rendered_html_path)
                            if rendered_meta:
                                write_rendered_meta(
                                    path=rendered_meta,
                                    source_meta=meta,
                                    key=item["key"],
                                    url=url,
                                    archived_html_path=item["html_path"],
                                    rendered_html_path=str(rendered_html_path),
                                    render_mode=render_mode,
                                )
                                rendered_meta_path = str(rendered_meta)
                    except Exception as e:
                        render_error = str(e)
                        analysis_source = "render_failed_archived_html"
                        print(f"[RENDER ERROR] {url} -> {render_error}")

            metrics, section_outline = analyzer.analyze(html=html, url=url)
            record = {
                "key": item["key"],
                "meta_path": item["meta_path"],
                "rendered_meta_path": rendered_meta_path,
                "html_path": analyzed_html_path,
                "archived_html_path": item["html_path"],
                "analysis_source": analysis_source,
                "archived_body_text_length": archived_body_text_length,
                "archived_empty_body": archived_empty_body,
                "render_mode": render_mode,
                "render_error": render_error,
                "fetched_at": meta.get("fetched_at", ""),
                "http_status": meta.get("http_status", ""),
                "content_type": meta.get("content_type", ""),
                **asdict(metrics),
                "section_outline": section_outline,
            }
            records.append(record)
            print(f"[ANALYZED] {url}")

    write_metrics_csv(output_csv, records)
    write_metrics_jsonl(output_jsonl, records)
    return records


def write_rendered_meta(
    path: str | Path,
    source_meta: dict[str, Any],
    key: str,
    url: str,
    archived_html_path: str,
    rendered_html_path: str,
    render_mode: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rendered_meta = {
        **source_meta,
        "key": key,
        "url": url,
        "rendered_at": utc_now_iso(),
        "render_mode": render_mode,
        "analysis_source": "playwright_rendered",
        "archived_html_path": archived_html_path,
        "rendered_html_path": rendered_html_path,
        "source_fetched_at": source_meta.get("fetched_at", ""),
        "source_http_status": source_meta.get("http_status", ""),
        "source_content_type": source_meta.get("content_type", ""),
    }

    path.write_text(
        json.dumps(rendered_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_metrics_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "key",
        "url",
        "url_depth",
        "title",
        "total_text_length",
        "body_text_length",
        "empty_body",
        "archived_body_text_length",
        "archived_empty_body",
        "link_count",
        "link_text_ratio",
        "section_count",
        "div_count",
        "li_count",
        "main_text_ratio",
        "section_text_ratio",
        "h1_count",
        "h2_count",
        "h3_count",
        "section_with_heading_ratio",
        "dominant_segmentation_type",
        "hidden_element_count",
        "accordion_like_count",
        "modal_like_count",
        "js_event_handler_count",
        "script_ui_keyword_count",
        "section_outline",
        "http_status",
        "fetched_at",
        "content_type",
        "analysis_source",
        "render_mode",
        "render_error",
        "html_path",
        "archived_html_path",
        "meta_path",
        "rendered_meta_path",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def write_metrics_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze already archived static HTML files."
    )
    parser.add_argument(
        "--job",
        choices=sorted(ANALYZER_JOBS),
        default=ACTIVE_JOB,
        help="Analyzer job preset name.",
    )
    parser.add_argument("--html-dir", default=None)
    parser.add_argument(
        "--output-csv",
        default=None,
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--render-mode",
        choices=["off", "fast", "precision"],
        default=None,
        help="Playwright rendering strategy: off=disabled, fast=empty body only, precision=all pages.",
    )
    parser.add_argument(
        "--rendered-html-dir",
        default=None,
        help="Directory to save Playwright-rendered HTML used for analysis.",
    )
    return parser.parse_args()


def get_active_job() -> AnalyzerJob:
    if ACTIVE_JOB not in ANALYZER_JOBS:
        known_jobs = ", ".join(sorted(ANALYZER_JOBS))
        raise ValueError(f"Unknown ACTIVE_JOB: {ACTIVE_JOB}. Known jobs: {known_jobs}")
    return ANALYZER_JOBS[ACTIVE_JOB]


def job_from_args(args: argparse.Namespace) -> AnalyzerJob:
    base_job = ANALYZER_JOBS[args.job]
    return AnalyzerJob(
        html_dir=args.html_dir or base_job.html_dir,
        output_csv=args.output_csv or base_job.output_csv,
        output_jsonl=args.output_jsonl or base_job.output_jsonl,
        rendered_html_dir=args.rendered_html_dir or base_job.rendered_html_dir,
        limit=args.limit,
        render_mode=args.render_mode or base_job.render_mode,
    )


def main() -> None:
    job = job_from_args(parse_args()) if USE_CLI_ARGS else get_active_job()

    records = analyze_archived_html_dir(
        html_dir=job.html_dir,
        output_csv=job.output_csv,
        output_jsonl=job.output_jsonl,
        limit=job.limit,
        render_mode=job.render_mode,
        rendered_html_dir=job.rendered_html_dir,
    )
    print(f"[DONE] analyzed={len(records)}")


if __name__ == "__main__":
    main()
