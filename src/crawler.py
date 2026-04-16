import csv
import os
import time
from dataclasses import dataclass, field
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

from playwright.sync_api import sync_playwright


@dataclass
class CrawlConfig:
    start_url: str
    allowed_domains: list[str]

    max_depth: int = 4
    max_links: int | None = None
    interval_sec: float = 1.0
    headless: bool = True
    debug: bool = True

    initial_wait_ms: int = 1200
    after_click_wait_ms: int = 1000
    extra_dom_wait_ms: int = 300

    block_resource_types: set[str] = field(default_factory=lambda: {"image", "font", "media"})

    enable_expand_click: bool = True
    max_click_candidates_per_page: int = 10

    click_candidate_selectors: list[str] = field(default_factory=lambda: [
        "button",
        "[role='button']",
        "summary",
        "[aria-expanded='false']",
        "input[type='button']",
        "input[type='submit']",
    ])
    
    allowed_url_prefixes: list[str] = field(default_factory=list)

    click_keywords: list[str] = field(default_factory=lambda: [
        "もっと見る", "表示", "開く", "展開", "さらに見る", "続きを見る",
        "more", "show", "expand", "open"
    ])

    require_in_viewport: bool = True
    max_viewport_y: int = 2200

    output_csv: str | None = None
    overwrite_csv: bool = False


class RateLimiter:
    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec
        self.last_time = 0.0

    # 前回のアクセスから指定秒数が経過するまで待機する。
    def wait(self):
        now = time.time()
        elapsed = now - self.last_time
        if elapsed < self.interval_sec:
            time.sleep(self.interval_sec - elapsed)
        self.last_time = time.time()


class GenericBFSCrawler:
    # クロール状態と設定を初期化する。
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.interval_sec)

        self.visited: set[str] = set()
        self.discovered: set[str] = set()
        self.results: list[dict] = []
        self.link_count = 0

    # debugが有効な場合だけログを出力する。
    def log(self, *args):
        if self.config.debug:
            print(*args)

    # hrefを絶対URL化し、許可されたURLだけに正規化する。
    def _normalize_url(self, base_url: str, href: str | None) -> str | None:
        if not href:
            return None

        href = href.strip()
        if not href:
            return None

        if href.startswith(("javascript:", "mailto:", "tel:")):
            return None

        abs_url = urljoin(base_url, href)
        abs_url, _frag = urldefrag(abs_url)

        parsed = urlparse(abs_url)
        if parsed.scheme not in ("http", "https"):
            return None

        if not any(parsed.netloc == domain or parsed.netloc.endswith(f".{domain}") for domain in self.config.allowed_domains):
            return None
        
        if self.config.allowed_url_prefixes:
            if not any(self._matches_url_prefix(abs_url, prefix) for prefix in self.config.allowed_url_prefixes):
                return None

        return abs_url

    # URLが設定されたprefixのパス配下にあるかを判定する。
    def _matches_url_prefix(self, url: str, prefix: str) -> bool:
        parsed_url = urlparse(url)
        parsed_prefix = urlparse(prefix)

        if parsed_url.scheme != parsed_prefix.scheme:
            return False

        if parsed_url.netloc != parsed_prefix.netloc:
            return False

        prefix_path = parsed_prefix.path.rstrip("/")
        if not prefix_path:
            return True

        return parsed_url.path == prefix_path or parsed_url.path.startswith(f"{prefix_path}/")

    # 最大リンク数が設定されている場合、残り何件追加できるかを返す。
    def _remaining_link_slots(self) -> int | None:
        if self.config.max_links is None:
            return None

        return max(self.config.max_links - self.link_count, 0)

    # 最大リンク数に収まるようにリンク一覧を切り詰める。
    def _limit_links(self, links: list[dict]) -> list[dict]:
        remaining = self._remaining_link_slots()
        if remaining is None:
            return links

        return links[:remaining]

    # ページタイトルを取得し、取得できない場合はog:titleを使う。
    def _get_title(self, page) -> str:
        try:
            title = (page.title() or "").strip()
            if title:
                return title
        except Exception:
            pass

        try:
            meta = page.locator("meta[property='og:title']").first.get_attribute("content")
            if meta:
                return meta.strip()
        except Exception:
            pass

        return ""

    # ページ内のリンクURLとアンカーテキストを重複を除いて収集する。
    def _collect_links_with_text(self, page) -> list[dict]:
        elements = page.query_selector_all("a[href]")
        results = []
        seen = set()

        for el in elements:
            try:
                href = el.get_attribute("href")
                norm = self._normalize_url(page.url, href)
                if not norm:
                    continue

                try:
                    text = (el.inner_text() or "").strip()
                except Exception:
                    text = ""

                key = (norm, text)
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "url": norm,
                    "text": text,
                })
            except Exception:
                continue

        return results

    # ページ内リンクのURLだけをsetで取得する。
    def _collect_link_urls(self, page) -> set[str]:
        links = self._collect_links_with_text(page)
        return {x["url"] for x in links}

    # 要素がクリックして展開を試す対象かどうかを判定する。
    def _match_click_candidate(self, item) -> bool:
        try:
            if not item.is_visible() or not item.is_enabled():
                return False
        except Exception:
            return False

        if self.config.require_in_viewport:
            try:
                box = item.bounding_box()
                if not box:
                    return False
                if box["x"] < 0 or box["y"] < 0:
                    return False
                if box["y"] > self.config.max_viewport_y:
                    return False
            except Exception:
                return False

        try:
            text = item.inner_text(timeout=300).strip().lower()
        except Exception:
            text = ""

        try:
            aria_label = (item.get_attribute("aria-label") or "").strip().lower()
        except Exception:
            aria_label = ""

        try:
            aria_expanded = item.get_attribute("aria-expanded")
        except Exception:
            aria_expanded = None

        label = f"{text} {aria_label}".strip()

        if any(k.lower() in label for k in self.config.click_keywords):
            return True

        if aria_expanded is not None:
            return True

        return False

    # ページ内から「もっと見る」などの展開クリック候補を探す。
    def _find_click_candidates(self, page):
        if not self.config.enable_expand_click:
            return []

        candidates = []
        seen = set()

        for selector in self.config.click_candidate_selectors:
            loc = page.locator(selector)
            count = min(loc.count(), self.config.max_click_candidates_per_page * 3)

            for i in range(count):
                item = loc.nth(i)

                if not self._match_click_candidate(item):
                    continue

                try:
                    key = item.evaluate(
                        """(el) => [
                            el.tagName,
                            el.id || '',
                            el.className || '',
                            el.getAttribute('aria-label') || '',
                            (el.innerText || '').trim()
                        ].join('|')"""
                    )
                except Exception:
                    key = f"{selector}:{i}"

                if key in seen:
                    continue
                seen.add(key)
                candidates.append(item)

                if len(candidates) >= self.config.max_click_candidates_per_page:
                    return candidates

        return candidates

    # 1ページ分のタイトル、通常リンク、クリック展開後リンクを抽出する。
    def _extract_page(self, page, url: str, depth: int) -> dict:
        title = self._get_title(page)

        before_links = self._collect_links_with_text(page)
        before_link_urls = {x["url"] for x in before_links}

        click_candidates = self._find_click_candidates(page)
        self.log(
            f"[PAGE] depth={depth} "
            f"links={len(before_link_urls)} "
            f"click_candidates={len(click_candidates)} "
            f"url={url}"
        )

        all_links_map = {(x["url"], x["text"]): x for x in before_links}

        for idx, locator in enumerate(click_candidates, start=1):
            try:
                before_url = page.url
                before_html = page.content()
                before_click_urls = self._collect_link_urls(page)

                self.rate_limiter.wait()
                locator.scroll_into_view_if_needed(timeout=1000)
                time.sleep(0.2)
                locator.click(timeout=2000, no_wait_after=True)

                page.wait_for_timeout(self.config.after_click_wait_ms)
                page.wait_for_timeout(self.config.extra_dom_wait_ms)

                after_url = page.url
                if after_url != before_url:
                    self.log(f"  [CLICK #{idx}] navigated -> skip")
                    continue

                after_html = page.content()
                after_links = self._collect_links_with_text(page)
                after_click_urls = {x["url"] for x in after_links}

                gained = after_click_urls - before_click_urls
                changed = (after_html != before_html)

                if gained:
                    self.log(f"  [CLICK #{idx}] gained_links={len(gained)}")
                elif changed:
                    self.log(f"  [CLICK #{idx}] dom_changed_only")
                else:
                    self.log(f"  [CLICK #{idx}] no change")

                for item in after_links:
                    all_links_map[(item["url"], item["text"])] = item

            except Exception as e:
                self.log(f"  [CLICK #{idx}] skipped: {e}")

        discovered_links = sorted(
            all_links_map.values(),
            key=lambda x: (x["url"], x["text"])
        )

        return {
            "url": url,
            "title": title,
            "depth": depth,
            "links": discovered_links,
        }

    # CSV出力が有効な場合、未作成または空ファイルにヘッダーを書き込む。
    def _write_csv_header_if_needed(self):
        if not self.config.output_csv:
            return

        output_dir = os.path.dirname(self.config.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.config.overwrite_csv:
            with open(self.config.output_csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "source_url",
                    "source_title",
                    "depth",
                    "target_url",
                    "anchor_text",
                ])
            return

        file_exists = os.path.exists(self.config.output_csv)
        if file_exists and os.path.getsize(self.config.output_csv) > 0:
            return

        with open(self.config.output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "source_url",
                "source_title",
                "depth",
                "target_url",
                "anchor_text",
            ])

    # 1ページから見つかったリンク一覧をCSVに追記する。
    def _write_csv(self, source_url: str, source_title: str, depth: int, links: list[dict]):
        if not self.config.output_csv:
            return

        with open(self.config.output_csv, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            for link in links:
                writer.writerow([
                    source_url,
                    source_title,
                    depth,
                    link["url"],
                    link["text"],
                ])

    # 開始URLからBFSでページを巡回し、結果を返す。
    def crawl(self) -> list[dict]:
        queue = deque()

        start_url = self._normalize_url(self.config.start_url, self.config.start_url)
        if not start_url:
            raise ValueError("start_url is invalid")

        queue.append((start_url, 0))
        self.discovered.add(start_url)

        self._write_csv_header_if_needed()

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=self.config.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )

            context = browser.new_context(
                locale="ja-JP",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/123.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1440, "height": 1400},
            )

            page = context.new_page()

            page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in self.config.block_resource_types
                else route.continue_()
            )

            while queue:
                if self.config.max_links is not None and self.link_count >= self.config.max_links:
                    self.log(f"[STOP] max_links reached: {self.link_count}")
                    break

                url, depth = queue.popleft()

                if url in self.visited:
                    continue

                if depth > self.config.max_depth:
                    continue

                self.visited.add(url)

                try:
                    self.log(f"[OPEN] depth={depth} {url}")

                    self.rate_limiter.wait()
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(self.config.initial_wait_ms)

                    record = self._extract_page(page, url, depth)
                    record["links"] = self._limit_links(record["links"])
                    self.link_count += len(record["links"])
                    self.results.append(record)

                    self._write_csv(
                        source_url=record["url"],
                        source_title=record["title"],
                        depth=record["depth"],
                        links=record["links"],
                    )

                    if depth < self.config.max_depth:
                        next_urls = sorted({x["url"] for x in record["links"]})
                        for next_url in next_urls:
                            if next_url not in self.discovered:
                                self.discovered.add(next_url)
                                queue.append((next_url, depth + 1))

                except Exception as e:
                    self.log(f"[ERROR] {url} -> {e}")
                    self.results.append({
                        "url": url,
                        "title": "",
                        "depth": depth,
                        "links": [],
                        "error": str(e),
                    })

            browser.close()

        return self.results

# ----

def ocn_support():
    config = CrawlConfig(
        start_url="https://support.ocn.ne.jp/",
        allowed_domains=["support.ocn.ne.jp"],
        max_depth=4,
        interval_sec=1.0,
        headless=True,
        debug=True,
        output_csv="output/ocn_support_links.csv",
    )

    crawler = GenericBFSCrawler(config)
    results = crawler.crawl()

    print(f"\nCRAWLED PAGES: {len(results)}")
    for r in results[:5]:
        print({
            "depth": r["depth"],
            "title": r["title"],
            "url": r["url"],
            "link_count": len(r["links"]),
        })
        
def docomo_faq():
    config = CrawlConfig(
         start_url="https://www.docomo.ne.jp/faq/",
         allowed_domains=["www.docomo.ne.jp"],
         allowed_url_prefixes=["https://www.docomo.ne.jp/faq"],
         max_depth=4,
         max_links=2500,
         interval_sec=1.0,
         headless=True,
         debug=True,
         output_csv="output/docomo_faq_links.csv",
         overwrite_csv=True,
     )

    crawler = GenericBFSCrawler(config)
    results = crawler.crawl()

    print(f"\nCRAWLED PAGES: {len(results)}")
    for r in results[:5]:
        print({
            "depth": r["depth"],
            "title": r["title"],
            "url": r["url"],
            "link_count": len(r["links"]),
        })
        
        
def goo_point():
    config = CrawlConfig(
        start_url="https://point.d.goo.ne.jp/",
        allowed_domains=["point.d.goo.ne.jp"],
        max_depth=4,
        interval_sec=1.0,
        headless=True,
        debug=True,
        enable_expand_click=False,
        output_csv="output/point_d_goo_links.csv",
    )
    crawler = GenericBFSCrawler(config)
    results = crawler.crawl()

    print(f"\nCRAWLED PAGES: {len(results)}")
    for r in results[:5]:
        print({
            "depth": r["depth"],
            "title": r["title"],
            "url": r["url"],
            "link_count": len(r["links"]),
        })
        
def gooID():
    # gooIDログイン画面配下をクリック展開なしでクロールする。
    config = CrawlConfig(
        start_url="https://login.mail.goo.ne.jp/id/authn/LoginStart",
        allowed_domains = ["login.mail.goo.ne.jp"],
        max_depth=4,
        interval_sec=1.0,
        headless=True,
        debug=True,
        enable_expand_click=False,
        output_csv="output/gooID_links.csv",
    )
    crawler = GenericBFSCrawler(config)
    results = crawler.crawl()

    print(f"\nCRAWLED PAGES: {len(results)}")
    for r in results[:5]:
        print({
            "depth": r["depth"],
            "title": r["title"],
            "url": r["url"],
            "link_count": len(r["links"]),
        })

if __name__ == "__main__":
    #gooID()
    #goo_point()
    #ocn_support()
    docomo_faq()
    
