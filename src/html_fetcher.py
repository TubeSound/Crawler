from __future__ import annotations

import hashlib
import json
import math
import re
import os
import time
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


WHITESPACE_RE = re.compile(r"\s+")
NOISE_CLASS_RE = re.compile(
    r"(nav|menu|breadcrumb|footer|header|sidebar|pager|pagination|sns|social|share|banner|ad|advert|modal|dialog)",
    re.IGNORECASE,
)
NON_HTML_EXTENSIONS = {
    # images
    ".apng", ".avif", ".bmp", ".gif", ".ico", ".jfif", ".jpeg", ".jpg",
    ".pjp", ".pjpeg", ".png", ".svg", ".tif", ".tiff", ".webp",
    # video
    ".3g2", ".3gp", ".avi", ".flv", ".m2ts", ".m4v", ".mkv", ".mov",
    ".mp4", ".mpeg", ".mpg", ".ogv", ".ts", ".webm", ".wmv",
    # audio
    ".aac", ".aiff", ".flac", ".m4a", ".mid", ".midi", ".mp3", ".oga",
    ".ogg", ".opus", ".wav", ".weba", ".wma",
    # documents
    ".csv", ".doc", ".docm", ".docx", ".dot", ".dotm", ".dotx", ".epub",
    ".odp", ".ods", ".odt", ".pdf", ".pps", ".ppsm", ".ppsx", ".ppt",
    ".pptm", ".pptx", ".rtf", ".tsv", ".txt", ".xls", ".xlsb", ".xlsm",
    ".xlsx", ".xml",
    # archives and binaries
    ".7z", ".apk", ".bin", ".bz2", ".dmg", ".exe", ".gz", ".iso", ".jar",
    ".msi", ".rar", ".tar", ".tgz", ".xz", ".zip",
}


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


def get_body_text_length(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    if not soup.body:
        return 0
    return get_tag_text_len(soup.body)


def normalize_url(base_url: str, href: str | None) -> str | None:
    if not href:
        return None

    href = href.strip()
    if not href or href.startswith(("javascript:", "mailto:", "tel:")):
        return None

    url = urljoin(base_url, href)
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return None

    if is_non_html_url(url):
        return None

    return url


def is_non_html_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    suffix = Path(path).suffix
    return suffix in NON_HTML_EXTENSIONS


def filter_html_urls(urls: list[str]) -> list[str]:
    filtered_urls = []
    seen = set()

    for url in urls:
        if not isinstance(url, str):
            continue

        normalized = normalize_url(url, url)
        if not normalized:
            continue

        if normalized in seen:
            continue
        seen.add(normalized)
        filtered_urls.append(normalized)

    return filtered_urls


def extract_links_from_html(
    source_url: str,
    source_key: str,
    rendered_html_path: str | Path,
    allowed_url_prefix: str | None = None,
) -> list[dict[str, str]]:
    rendered_html_path = Path(rendered_html_path)
    soup = BeautifulSoup(rendered_html_path.read_text(encoding="utf-8"), "html.parser")
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        target_url = normalize_url(source_url, a.get("href"))
        if not target_url:
            continue

        if allowed_url_prefix and not target_url.startswith(allowed_url_prefix):
            continue

        if target_url in seen:
            continue
        seen.add(target_url)

        links.append({
            "source_url": source_url,
            "source_key": source_key,
            "source_rendered_html_path": str(rendered_html_path),
            "target_url": target_url,
            "anchor_text": get_tag_text(a),
        })

    return links


def collect_rendered_links_from_dir(
    rendered_dir: str | Path,
    allowed_url_prefix: str | None = None,
) -> list[dict[str, str]]:
    rendered_dir = Path(rendered_dir)
    links: list[dict[str, str]] = []

    for meta_path in sorted(rendered_dir.glob("*.rendered.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source_url = meta.get("url", "")
        source_key = meta.get("key", meta_path.name.removesuffix(".rendered.meta.json"))
        rendered_html_path = meta.get(
            "rendered_html_path",
            str(rendered_dir / f"{source_key}.rendered.html"),
        )

        if not source_url or not Path(rendered_html_path).exists():
            continue

        links.extend(extract_links_from_html(
            source_url=source_url,
            source_key=source_key,
            rendered_html_path=rendered_html_path,
            allowed_url_prefix=allowed_url_prefix,
        ))

    return links


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
    meta = get_tag_class_id_text(tag)
    return bool(NOISE_CLASS_RE.search(meta))


def is_non_content_tag(tag: Tag) -> bool:
    return tag.name in {"script", "style", "noscript", "svg", "canvas", "iframe"}


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class FetchResult:
    url: str
    fetched_at: str
    http_status: int
    etag: str
    last_modified: str
    title: str
    html: str
    content_type: str
    encoding: str


@dataclass
class RenderResult:
    url: str
    rendered_at: str
    title: str
    html: str
    source_key: str
    source_meta_path: str
    source_html_path: str
    console_errors: list[str]
    page_errors: list[str]
    spapp_available: bool
    spapp_render_invoked: bool





class HtmlArchiveStore:
    """
    URLごとに sha256(url) をキーとして保存する。

    files/
      <hash>.meta.json
      <hash>.html
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def key_from_url(self, url: str) -> str:
        return sha256_hex(url)

    def meta_path(self, key: str) -> Path:
        return self.base_dir / f"{key}.meta.json"

    def html_path(self, key: str) -> Path:
        return self.base_dir / f"{key}.html"

    def save(self, result: FetchResult) -> dict[str, Path]:
        key = self.key_from_url(result.url)

        meta = {
            "url": result.url,
            "fetched_at": result.fetched_at,
            "http_status": result.http_status,
            "etag": result.etag,
            "last_modified": result.last_modified,
            "title": result.title,
            "content_type": result.content_type,
            "encoding": result.encoding,
        }

        meta_path = self.meta_path(key)
        html_path = self.html_path(key)

        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(result.html, encoding="utf-8")

        return {
            "meta_path": meta_path,
            "html_path": html_path,
        }

    def load_by_url(self, url: str) -> tuple[dict[str, Any], str]:
        key = self.key_from_url(url)
        return self.load_by_key(key)

    def load_by_key(self, key: str) -> tuple[dict[str, Any], str]:
        meta = json.loads(self.meta_path(key).read_text(encoding="utf-8"))
        html = self.html_path(key).read_text(encoding="utf-8")
        return meta, html

    def exists(self, url: str) -> bool:
        key = self.key_from_url(url)
        return self.meta_path(key).exists() and self.html_path(key).exists()


class RenderedHtmlArchiveStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def rendered_meta_path(self, key: str) -> Path:
        return self.base_dir / f"{key}.rendered.meta.json"

    def rendered_html_path(self, key: str) -> Path:
        return self.base_dir / f"{key}.rendered.html"

    def exists(self, key: str) -> bool:
        return self.rendered_meta_path(key).exists() and self.rendered_html_path(key).exists()

    def is_valid(self, key: str, source_html_path: str | Path) -> bool:
        if not self.exists(key):
            return False

        source_path = Path(source_html_path)
        rendered_path = self.rendered_html_path(key)
        if not source_path.exists() or not rendered_path.exists():
            return False

        source_html = source_path.read_text(encoding="utf-8")
        rendered_html = rendered_path.read_text(encoding="utf-8")

        if sha256_hex(source_html) == sha256_hex(rendered_html):
            return False

        return get_body_text_length(rendered_html) > get_body_text_length(source_html)

    def save(self, result: RenderResult) -> dict[str, Path]:
        meta_path = self.rendered_meta_path(result.source_key)
        html_path = self.rendered_html_path(result.source_key)
        source_html = Path(result.source_html_path).read_text(encoding="utf-8")
        rendered_html = result.html
        source_sha256 = sha256_hex(source_html)
        rendered_sha256 = sha256_hex(rendered_html)
        source_body_text_length = get_body_text_length(source_html)
        rendered_body_text_length = get_body_text_length(rendered_html)

        meta = {
            "url": result.url,
            "key": result.source_key,
            "rendered_at": result.rendered_at,
            "title": result.title,
            "analysis_source": "playwright_rendered",
            "source_meta_path": result.source_meta_path,
            "source_html_path": result.source_html_path,
            "rendered_html_path": str(html_path),
            "source_html_sha256": source_sha256,
            "rendered_html_sha256": rendered_sha256,
            "rendered_differs_from_source": source_sha256 != rendered_sha256,
            "source_body_text_length": source_body_text_length,
            "rendered_body_text_length": rendered_body_text_length,
            "render_success": (
                source_sha256 != rendered_sha256
                and rendered_body_text_length > source_body_text_length
            ),
            "console_errors": result.console_errors,
            "page_errors": result.page_errors,
            "spapp_available": result.spapp_available,
            "spapp_render_invoked": result.spapp_render_invoked,
        }

        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        html_path.write_text(result.html, encoding="utf-8")

        return {
            "rendered_meta_path": meta_path,
            "rendered_html_path": html_path,
        }


class HtmlRenderer:
    def __init__(
        self,
        wait_timeout_ms: int = 15000,
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    ) -> None:
        self.wait_timeout_ms = wait_timeout_ms
        self.user_agent = user_agent
        self.playwright = None
        self.browser = None
        self.context = None

    def __enter__(self) -> HtmlRenderer:
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
            user_agent=self.user_agent,
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

    def render(
        self,
        url: str,
        source_key: str,
        source_meta_path: str | Path,
        source_html_path: str | Path,
    ) -> RenderResult:
        if self.context is None:
            raise RuntimeError("HtmlRenderer is not started.")

        console_errors: list[str] = []
        page_errors: list[str] = []
        page = self.context.new_page()
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text)
            if msg.type in {"error", "warning"}
            else None,
        )
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
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
                    return (body.innerText || '').trim().length > 0;
                }""",
                timeout=self.wait_timeout_ms,
            )
        except PlaywrightTimeoutError:
            pass

        spapp_available = False
        spapp_render_invoked = False
        try:
            spapp_available = bool(page.evaluate(
                """() => !!(
                    window.mydcm &&
                    window.mydcm.common &&
                    window.mydcm.common.SPApp &&
                    typeof window.mydcm.common.SPApp.render === 'function'
                )"""
            ))
            if spapp_available:
                page.evaluate("() => window.mydcm.common.SPApp.render()")
                spapp_render_invoked = True
                page.wait_for_timeout(1000)
        except Exception as e:
            page_errors.append(f"SPApp render failed: {e}")

        html = page.content()
        title = page.title()
        page.close()

        return RenderResult(
            url=url,
            rendered_at=utc_now_iso(),
            title=title or "",
            html=html,
            source_key=source_key,
            source_meta_path=str(source_meta_path),
            source_html_path=str(source_html_path),
            console_errors=console_errors[:20],
            page_errors=page_errors[:20],
            spapp_available=spapp_available,
            spapp_render_invoked=spapp_render_invoked,
        )


class HtmlFetcher:
    def __init__(
        self,
        timeout: tuple[int, int] = (10, 20),
        min_interval_sec: float = 1.0,
        user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    ) -> None:
        self.timeout = timeout
        self.min_interval_sec = min_interval_sec
        self.last_fetch_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def wait_if_needed(self) -> None:
        now = time.time()
        elapsed = now - self.last_fetch_time
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)

    def fetch(self, url: str) -> FetchResult:
        self.wait_if_needed()
        response = self.session.get(url, timeout=self.timeout)
        self.last_fetch_time = time.time()
        response.raise_for_status()

        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = response.apparent_encoding

        html = response.text
        title = self._extract_title(html)

        return FetchResult(
            url=url,
            fetched_at=utc_now_iso(),
            http_status=response.status_code,
            etag=response.headers.get("ETag", ""),
            last_modified=response.headers.get("Last-Modified", ""),
            title=title,
            html=html,
            content_type=response.headers.get("Content-Type", ""),
            encoding=response.encoding or "",
        )

    def fetch_and_save(
        self,
        url: str,
        store: HtmlArchiveStore,
        rendered_store: RenderedHtmlArchiveStore | None = None,
        renderer: HtmlRenderer | None = None,
        skip_if_exists: bool = True,
        skip_rendered_if_exists: bool = True,
    ) -> dict[str, Any]:
        if is_non_html_url(url):
            key = store.key_from_url(url)
            return {
                "url": url,
                "key": key,
                "saved": False,
                "reason": "non_html_resource",
                "meta_path": store.meta_path(key),
                "html_path": store.html_path(key),
            }

        key = store.key_from_url(url)

        if skip_if_exists and store.exists(url):
            result = {
                "url": url,
                "key": key,
                "saved": False,
                "reason": "already_exists",
                "meta_path": store.meta_path(key),
                "html_path": store.html_path(key),
            }

            rendered_result = self.render_and_save(
                url=url,
                key=key,
                store=store,
                rendered_store=rendered_store,
                renderer=renderer,
                skip_rendered_if_exists=skip_rendered_if_exists,
            )
            result.update(rendered_result)
            return result

        try:
            result = self.fetch(url)
        except requests.RequestException as e:
            return {
                "url": url,
                "key": key,
                "saved": False,
                "reason": "fetch_error",
                "error": str(e),
                "status_code": e.response.status_code if e.response is not None else "",
                "meta_path": store.meta_path(key),
                "html_path": store.html_path(key),
            }

        paths = store.save(result)

        fetch_result = {
            "url": url,
            "key": key,
            "saved": True,
            "meta_path": paths["meta_path"],
            "html_path": paths["html_path"],
        }

        rendered_result = self.render_and_save(
            url=url,
            key=key,
            store=store,
            rendered_store=rendered_store,
            renderer=renderer,
            skip_rendered_if_exists=skip_rendered_if_exists,
        )
        fetch_result.update(rendered_result)
        return fetch_result

    def render_and_save(
        self,
        url: str,
        key: str,
        store: HtmlArchiveStore,
        rendered_store: RenderedHtmlArchiveStore | None,
        renderer: HtmlRenderer | None,
        skip_rendered_if_exists: bool = True,
    ) -> dict[str, Any]:
        if rendered_store is None:
            return {}

        if skip_rendered_if_exists and rendered_store.is_valid(key, store.html_path(key)):
            return {
                "rendered_saved": False,
                "rendered_reason": "already_exists",
                "rendered_meta_path": rendered_store.rendered_meta_path(key),
                "rendered_html_path": rendered_store.rendered_html_path(key),
            }

        if renderer is None:
            return {
                "rendered_saved": False,
                "rendered_reason": "renderer_not_provided",
                "rendered_meta_path": rendered_store.rendered_meta_path(key),
                "rendered_html_path": rendered_store.rendered_html_path(key),
            }

        try:
            render_result = renderer.render(
                url=url,
                source_key=key,
                source_meta_path=store.meta_path(key),
                source_html_path=store.html_path(key),
            )
            rendered_paths = rendered_store.save(render_result)
            rendered_valid = rendered_store.is_valid(key, store.html_path(key))
            if not rendered_valid:
                return {
                    "rendered_saved": False,
                    "rendered_reason": "rendered_same_or_empty",
                    "rendered_meta_path": rendered_paths["rendered_meta_path"],
                    "rendered_html_path": rendered_paths["rendered_html_path"],
                }

            return {
                "rendered_saved": True,
                "rendered_meta_path": rendered_paths["rendered_meta_path"],
                "rendered_html_path": rendered_paths["rendered_html_path"],
            }
        except Exception as e:
            return {
                "rendered_saved": False,
                "rendered_reason": "render_error",
                "rendered_error": str(e),
                "rendered_meta_path": rendered_store.rendered_meta_path(key),
                "rendered_html_path": rendered_store.rendered_html_path(key),
            }

    @staticmethod
    def _extract_title(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    
    
def main():
    import sys
    no = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) >= 3 else "fetch"
    print(no)
    if no == 1:
        input_path = "./output/gooID_links.csv"
        output_dir = "./output/gooID/html"
        rendered_output_dir = "./output/gooID/rendered_html"
        rendered_links_path = "./output/gooID/rendered_links.csv"
        allowed_rendered_link_prefix = None
    elif no == 2:
        input_path = "./output/docomo_faq_links.csv"
        output_dir = "./output/docomo_faq/html"
        rendered_output_dir = "./output/docomo_faq/rendered_html"
        rendered_links_path = "./output/docomo_faq/rendered_links.csv"
        allowed_rendered_link_prefix = "https://www.docomo.ne.jp/faq/"
    elif no == 3:
        input_path = "./output/ocn_support_links.csv"
        output_dir = "./output/ocn_support/html"
        rendered_output_dir = "./output/ocn_support/rendered_html"
        rendered_links_path = "./output/ocn_support/rendered_links.csv"
        allowed_rendered_link_prefix = "https://support.ocn.ne.jp/"
    elif no == 4:
        input_path = "./output/point_d_goo_links.csv"
        output_dir = "./output/point_goo/html"
        rendered_output_dir = "./output/point_goo/rendered_html"
        rendered_links_path = "./output/point_goo/rendered_links.csv"
        allowed_rendered_link_prefix = "https://point.d.goo.ne.jp/"
    elif no == 5:
        input_path = "./output/docomo_denki_links.csv"
        output_dir = "./output/docomo_denki/html"
        rendered_output_dir = "./output/docomo_denki/rendered_html"
        rendered_links_path = "./output/docomo_denki/rendered_links.csv"
        allowed_rendered_link_prefix = "https://denki.docomo.ne.jp/"
        

    if mode == "links":
        rendered_links = collect_rendered_links_from_dir(
            rendered_output_dir,
            allowed_url_prefix=allowed_rendered_link_prefix,
        )
        pd.DataFrame(rendered_links).drop_duplicates(
            subset=["source_url", "target_url"],
        ).to_csv(
            rendered_links_path,
            index=False,
            encoding="utf-8-sig",
        )
        print({
            "mode": mode,
            "rendered_links_path": rendered_links_path,
            "link_count": len(rendered_links),
        })
        return

    df = pd.read_csv(input_path)
    raw_urls = df['target_url'].to_list()
    urls = filter_html_urls(raw_urls)
    print({
        "input_url_count": len(raw_urls),
        "html_url_count": len(urls),
        "filtered_url_count": len(raw_urls) - len(urls),
    })
    os.makedirs(output_dir, exist_ok=True)
    store = HtmlArchiveStore(output_dir)
    rendered_store = RenderedHtmlArchiveStore(rendered_output_dir)
    fetcher = HtmlFetcher()
    results = []
    rendered_links = []
    with HtmlRenderer() as renderer:
        for url in urls:
            r = fetcher.fetch_and_save(
                url,
                store,
                rendered_store=rendered_store,
                renderer=renderer,
            )
            results.append(r)
            print(r)

            rendered_html_path = r.get("rendered_html_path")
            if rendered_html_path and Path(rendered_html_path).exists():
                rendered_links.extend(extract_links_from_html(
                    source_url=url,
                    source_key=r["key"],
                    rendered_html_path=rendered_html_path,
                    allowed_url_prefix=allowed_rendered_link_prefix,
                ))

    pd.DataFrame(results).to_csv(
        f"{output_dir}/html_fetch_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(rendered_links).drop_duplicates(
        subset=["source_url", "target_url"],
    ).to_csv(
        rendered_links_path,
        index=False,
        encoding="utf-8-sig",
    )
                
        
if __name__ == "__main__":
    main()
    
    
    
