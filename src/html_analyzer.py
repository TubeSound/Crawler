from __future__ import annotations

import argparse
import csv
import json
import math
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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


def is_unrendered_nuxt_shell(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    if body is None:
        return False

    has_nuxt_root = body.find(id="__nuxt") is not None
    has_nuxt_data = soup.find("script", id="__NUXT_DATA__") is not None
    body_text_len = get_tag_text_len(body)
    return has_nuxt_root and has_nuxt_data and body_text_len == 0


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
                        !document.querySelector('#__nuxt:empty')
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


@dataclass
class HtmlStructureMetrics:
    url: Optional[str]
    url_depth: int
    title: str
    total_text_length: int
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
    unrendered_nuxt_shell: int
    hidden_element_count: int
    accordion_like_count: int
    modal_like_count: int
    js_event_handler_count: int
    script_ui_keyword_count: int
    js_dynamic_ui_score: int


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

    def analyze(self, html: str, url: Optional[str] = None) -> HtmlStructureMetrics:
        unrendered_nuxt_shell = int(is_unrendered_nuxt_shell(html))
        soup = BeautifulSoup(html, self.parser)
        self._remove_non_content_nodes(soup)

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

        return HtmlStructureMetrics(
            url=url,
            url_depth=count_url_depth(url) if url else 0,
            title=title,
            total_text_length=total_text_length,
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
            unrendered_nuxt_shell=unrendered_nuxt_shell,
            hidden_element_count=hidden_element_count,
            accordion_like_count=accordion_like_count,
            modal_like_count=modal_like_count,
            js_event_handler_count=js_event_handler_count,
            script_ui_keyword_count=script_ui_keyword_count,
            js_dynamic_ui_score=js_dynamic_ui_score,
        )

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
            render_error = ""

            archived_unrendered_nuxt_shell = int(is_unrendered_nuxt_shell(item["html"]))
            should_render = False

            if render_mode == "precision" and url:
                should_render = True
            elif render_mode == "fast" and url and archived_unrendered_nuxt_shell:
                should_render = True

            if should_render and renderer:
                if render_mode == "fast" and archived_unrendered_nuxt_shell:
                    print(f"[RENDER] archived HTML is an empty Nuxt shell: {url}")
                else:
                    print(f"[RENDER] {url}")
                try:
                    html = renderer.render(url)
                    analysis_source = "playwright_rendered"

                    if rendered_html_base:
                        rendered_html_path = rendered_html_base / f"{item['key']}.rendered.html"
                        rendered_html_path.write_text(html, encoding="utf-8")
                        analyzed_html_path = str(rendered_html_path)
                except Exception as e:
                    render_error = str(e)
                    analysis_source = "render_failed_archived_html"
                    print(f"[RENDER ERROR] {url} -> {render_error}")

            metrics = analyzer.analyze(html=html, url=url)
            record = {
                "key": item["key"],
                "meta_path": item["meta_path"],
                "html_path": analyzed_html_path,
                "archived_html_path": item["html_path"],
                "analysis_source": analysis_source,
                "archived_unrendered_nuxt_shell": archived_unrendered_nuxt_shell,
                "render_mode": render_mode,
                "render_error": render_error,
                "fetched_at": meta.get("fetched_at", ""),
                "http_status": meta.get("http_status", ""),
                "content_type": meta.get("content_type", ""),
                **asdict(metrics),
            }
            records.append(record)
            print(f"[ANALYZED] {url}")

    write_metrics_csv(output_csv, records)
    write_metrics_jsonl(output_jsonl, records)
    return records


def write_metrics_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "key",
        "url",
        "url_depth",
        "title",
        "total_text_length",
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
        "unrendered_nuxt_shell",
        "archived_unrendered_nuxt_shell",
        "hidden_element_count",
        "accordion_like_count",
        "modal_like_count",
        "js_event_handler_count",
        "script_ui_keyword_count",
        "js_dynamic_ui_score",
        "http_status",
        "fetched_at",
        "content_type",
        "analysis_source",
        "render_mode",
        "render_error",
        "html_path",
        "archived_html_path",
        "meta_path",
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
    parser.add_argument("--html-dir", default="output/docomo_faq/html")
    parser.add_argument(
        "--output-csv",
        default="output/docomo_faq/feature/docomo_faq_feature.csv",
    )
    parser.add_argument(
        "--output-jsonl",
        default="output/docomo_faq/feature/docomo_faq_feature.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--render-mode",
        choices=["off", "fast", "precision"],
        default="fast",
        help="Playwright rendering strategy: off=disabled, fast=Nuxt shell only, precision=all pages.",
    )
    parser.add_argument(
        "--rendered-html-dir",
        default="output/docomo_faq/rendered_html",
        help="Directory to save Playwright-rendered HTML used for analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = analyze_archived_html_dir(
        html_dir=args.html_dir,
        output_csv=args.output_csv,
        output_jsonl=args.output_jsonl,
        limit=args.limit,
        render_mode=args.render_mode,
        rendered_html_dir=args.rendered_html_dir,
    )
    print(f"[DONE] analyzed={len(records)}")


if __name__ == "__main__":
    main()
