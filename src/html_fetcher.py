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
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag, NavigableString


WHITESPACE_RE = re.compile(r"\s+")
NOISE_CLASS_RE = re.compile(
    r"(nav|menu|breadcrumb|footer|header|sidebar|pager|pagination|sns|social|share|banner|ad|advert|modal|dialog)",
    re.IGNORECASE,
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
        skip_if_exists: bool = True,
    ) -> dict[str, Any]:
        if skip_if_exists and store.exists(url):
            key = store.key_from_url(url)
            return {
                "url": url,
                "key": key,
                "saved": False,
                "reason": "already_exists",
                "meta_path": store.meta_path(key),
                "html_path": store.html_path(key),
            }

        try:
            result = self.fetch(url)
        except requests.RequestException as e:
            key = store.key_from_url(url)
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
        key = store.key_from_url(url)

        return {
            "url": url,
            "key": key,
            "saved": True,
            "meta_path": paths["meta_path"],
            "html_path": paths["html_path"],
        }

    @staticmethod
    def _extract_title(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    
    
def main():
    import sys
    no = int(sys.argv[1])
    print(no)
    if no == 1:
        input_path = "./output/gooID_links.csv"
        output_dir = "./output/gooID/html"
    elif no == 2:
        input_path = "./output/docomo_faq_links.csv"
        output_dir = "./output/doconmo_faq/html"
    elif no == 3:
        input_path = "./output/ocn_support_links.csv"
        output_dir = "./output/ocn_support/html"
    elif no == 4:
        input_path = "./output/point_d_goo_links.csv"
        output_dir = "./output/point_goo/html"
        

    df = pd.read_csv(input_path)
    urls = df['target_url'].to_list()
    os.makedirs(output_dir, exist_ok=True)
    store = HtmlArchiveStore(output_dir)
    fetcher = HtmlFetcher()
    results = []
    for url in urls:
        r = fetcher.fetch_and_save(url, store)
        results.append(r)
        print(r)

    pd.DataFrame(results).to_csv(
        f"{output_dir}/html_fetch_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
                
        
if __name__ == "__main__":
    main()
    
    
    
