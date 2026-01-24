# -*- coding: utf-8 -*-
import csv
import json
import time
import os
import shutil
import re
import itertools
import requests
from urllib.parse import quote, urlsplit, urlunsplit, parse_qsl, urlencode
from collections import OrderedDict
from bs4 import BeautifulSoup
from tqdm import tqdm
import threading
from dataclasses import dataclass, field


@dataclass
class Stats:
    success: int = 0
    failed: int = 0
    retries: int = 0
    errors: dict = field(default_factory=dict)

    def record_error(self, error_type):
        self.errors[error_type] = self.errors.get(error_type, 0) + 1

    @property
    def total(self):
        return self.success + self.failed

    @property
    def success_rate(self):
        return (self.success / self.total * 100) if self.total > 0 else 0

    def summary(self):
        return f"success={self.success:,} failed={self.failed:,} retries={self.retries:,} rate={self.success_rate:.1f}%"


class RutubeScraper:
    BASE = "https://rutube.ru"
    API = "https://rutube.ru/api"

    def __init__(self, output_dir="scraped_data/v1_vids_and_channels", target_videos=100000, rps=4, max_retries=3, report_every=1000):
        self.output_dir = output_dir
        self.target_videos = target_videos
        self.min_delay = 1.0 / rps
        self.max_retries = max_retries
        self.report_every = report_every
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "rutube-research-bot/2.0"})
        self.last_request_time = 0
        self.lock = threading.Lock()
        self.videos = {}
        self.channels = {}
        self.seen_urls = set()
        self.video_stats = Stats()
        self.channel_stats = Stats()
        self.last_report_count = 0

    def setup_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def rate_limit(self):
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            self.last_request_time = time.time()

    def request_with_retry(self, request_func, stats):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.rate_limit()
                return request_func(), True
            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code
                error_type = f"HTTP_{status}"
                if status == 429:
                    stats.retries += 1
                    time.sleep(2 ** attempt)
                    continue
                elif status in (500, 502, 503, 504):
                    stats.retries += 1
                    time.sleep(1)
                    continue
                else:
                    stats.failed += 1
                    stats.record_error(error_type)
                    return None, False
            except requests.exceptions.Timeout:
                stats.retries += 1
                stats.record_error("Timeout")
                time.sleep(1)
                continue
            except requests.exceptions.ConnectionError:
                stats.retries += 1
                stats.record_error("ConnectionError")
                time.sleep(2)
                continue
            except Exception as e:
                stats.failed += 1
                stats.record_error(type(e).__name__)
                return None, False
        stats.failed += 1
        if last_error:
            stats.record_error(f"MaxRetries_{type(last_error).__name__}")
        return None, False

    def get_json(self, url, params=None, stats=None):
        if stats is None:
            stats = self.video_stats
        def request_func():
            r = self.session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        return self.request_with_retry(request_func, stats)

    def get_html(self, url, stats=None):
        if stats is None:
            stats = self.channel_stats
        def request_func():
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
            r.encoding = 'utf-8'
            return r.text
        return self.request_with_retry(request_func, stats)

    def extract_results(self, data):
        if not data:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("results", [])
        return []

    def maybe_report(self, pbar):
        current = len(self.videos)
        if current - self.last_report_count >= self.report_every:
            self.last_report_count = current
            tqdm.write(f"[{current:,} vids] Requests: {self.video_stats.summary()} | Errors: {dict(self.video_stats.errors)}")

    def discover_categories(self):
        data, ok = self.get_json(f"{self.API}/video/category/")
        if ok:
            return [cat["id"] for cat in self.extract_results(data) if isinstance(cat, dict) and cat.get("id")]
        return list(range(1, 50))

    def discover_feeds(self):
        feeds = []
        for feed_name in ["main", "tnt", "sport", "movie", "blogger", "news"]:
            data, ok = self.get_json(f"{self.API}/feeds/{feed_name}/")
            if ok and data:
                for tab in data.get("tabs", []) if isinstance(data, dict) else []:
                    for res in tab.get("resources", []):
                        if res.get("url"):
                            feeds.append(res["url"])
        return list(set(feeds))

    def discover_tags(self):
        tags = set()
        for vid_data in list(self.videos.values())[:5000]:
            for tag in vid_data.get("tags", []):
                if isinstance(tag, dict) and tag.get("name"):
                    tags.add(tag["name"])
                elif isinstance(tag, str):
                    tags.add(tag)
        return [t for t in tags if t and len(t) > 2][:200]

    def add_qs(self, url, extra):
        p = urlsplit(url)
        q = dict(parse_qsl(p.query))
        for k, v in extra.items():
            q.setdefault(k, v)
        return urlunsplit((p.scheme, p.netloc, p.path, urlencode(q), p.fragment))

    def scrape_paginated(self, start_url, pbar, max_pages=200):
        page_url = self.add_qs(start_url, {"per_page": "100"})
        pages = 0
        while page_url and len(self.videos) < self.target_videos and pages < max_pages:
            if page_url in self.seen_urls:
                break
            self.seen_urls.add(page_url)
            data, ok = self.get_json(page_url)
            if not ok or not data:
                break
            results = self.extract_results(data)
            if not results:
                break
            self.video_stats.success += 1
            for item in results:
                vid = item.get("id") or item.get("video_id")
                if vid and vid not in self.videos:
                    self.videos[vid] = item
                    pbar.update(1)
                    pbar.set_postfix({"unique": len(self.videos), "ok": self.video_stats.success, "fail": self.video_stats.failed})
                    self.maybe_report(pbar)
                    if len(self.videos) >= self.target_videos:
                        return True
            page_url = data.get("next") if isinstance(data, dict) else None
            pages += 1
        return len(self.videos) >= self.target_videos

    def scrape_videos(self):
        pbar = tqdm(total=self.target_videos, desc="Scraping videos", unit="vid", dynamic_ncols=True)

        pbar.set_description("Scraping videos [categories]")
        categories = self.discover_categories()
        for cat_id in categories:
            if self.scrape_paginated(f"{self.API}/video/category/{cat_id}/", pbar):
                break

        if len(self.videos) < self.target_videos:
            pbar.set_description("Scraping videos [feeds]")
            for feed_url in self.discover_feeds():
                if self.scrape_paginated(feed_url, pbar):
                    break

        if len(self.videos) < self.target_videos:
            pbar.set_description("Scraping videos [search]")
            search_terms = [
                "новости", "фильм", "сериал", "музыка", "спорт", "игры", "обзор",
                "рецепт", "авто", "политика", "экономика", "технологии", "наука",
                "история", "путешествия", "юмор", "дети", "животные", "природа",
                "ремонт", "строительство", "мода", "красота", "здоровье", "фитнес"
            ]
            for term in search_terms:
                if self.scrape_paginated(f"{self.API}/video/search/?query={quote(term)}", pbar, max_pages=100):
                    break

        if len(self.videos) < self.target_videos:
            pbar.set_description("Scraping videos [tags]")
            for tag in self.discover_tags():
                if self.scrape_paginated(f"{self.API}/tags/{quote(tag)}/video/", pbar, max_pages=50):
                    break

        if len(self.videos) < self.target_videos:
            pbar.set_description("Scraping videos [channels]")
            channel_ids = list(self.extract_channel_ids())[:500]
            for ch_id in channel_ids:
                if self.scrape_paginated(f"{self.API}/video/person/{ch_id}/", pbar, max_pages=20):
                    break

        pbar.close()

    def extract_channel_ids(self):
        channel_ids = set()
        for vid_data in self.videos.values():
            author = vid_data.get("author") or {}
            if author.get("id"):
                channel_ids.add(author["id"])
        return channel_ids

    def norm_num(self, txt):
        if not txt:
            return None
        txt = txt.replace("\u00A0", " ").replace("\u202F", " ")
        m = re.search(r"(\d[\d\s]*)", txt)
        if not m:
            return None
        try:
            return int(m.group(1).replace(" ", ""))
        except Exception:
            return None

    def parse_channel_html(self, channel_id):
        url = f"{self.BASE}/channel/{channel_id}/"
        html, ok = self.get_html(url)
        if not ok or not html:
            return {"channel_id": channel_id, "url": url, "error": "Failed to fetch"}
        soup = BeautifulSoup(html, "lxml")

        title = None
        ogt = soup.find("meta", property="og:title")
        if ogt and ogt.get("content"):
            title = ogt["content"]
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        desc = None
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            desc = md["content"]
        if not desc:
            ogd = soup.find("meta", property="og:description")
            if ogd and ogd.get("content"):
                desc = ogd["content"]

        avatar = None
        ogi = soup.find("meta", property="og:image")
        if ogi and ogi.get("content"):
            avatar = ogi["content"]

        subs = None
        body_txt = soup.get_text(" ", strip=True)
        m = re.search(r"(\d[\d\s]*)\s+подписчик\w*", body_txt, re.I)
        if m:
            subs = self.norm_num(m.group(0))

        meta = {}
        for mtag in soup.find_all("meta"):
            name = mtag.get("name") or ""
            prop = mtag.get("property") or ""
            key = f"meta.name:{name}" if name else (f"meta.property:{prop}" if prop else None)
            if key and mtag.get("content") is not None:
                meta[key] = mtag["content"]

        jsonld = []
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                jsonld.append(json.loads(s.string))
            except Exception:
                pass

        self.channel_stats.success += 1
        return {
            "channel_id": channel_id,
            "url": url,
            "title": title,
            "description": desc,
            "avatar_url": avatar,
            "subscribers": subs,
            "meta": meta,
            "jsonld": jsonld
        }

    def scrape_channels(self, channel_rps=15, max_workers=4):
        channel_ids = list(self.extract_channel_ids())
        report_interval = max(1, len(channel_ids) // 10)
        channel_delay = 0.0
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_channel(ch_id):
            time.sleep(channel_delay)
            return ch_id, self.parse_channel_html(ch_id)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_channel, ch_id): ch_id for ch_id in channel_ids if ch_id not in self.channels}
            with tqdm(total=len(futures), desc="Scraping channels", unit="ch", dynamic_ncols=True) as pbar:
                for i, future in enumerate(as_completed(futures)):
                    ch_id, result = future.result()
                    self.channels[ch_id] = result
                    pbar.update(1)
                    if (i + 1) % report_interval == 0:
                        tqdm.write(f"[{i+1:,}/{len(channel_ids):,} channels] {self.channel_stats.summary()} | Errors: {dict(self.channel_stats.errors)}")
    @staticmethod
    def flatten(d, parent="", out=None):
        if out is None:
            out = {}
        if isinstance(d, list):
            for i, v in enumerate(d):
                RutubeScraper.flatten(v, f"{parent}[{i}]" if parent else f"[{i}]", out)
        elif isinstance(d, dict):
            for k, v in d.items():
                nk = f"{parent}.{k}" if parent else k
                if isinstance(v, (dict, list)):
                    RutubeScraper.flatten(v, nk, out)
                else:
                    out[nk] = v
        else:
            out[parent] = d
        return out

    def save_ndjson(self, data, filename):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def save_flat_csv(self, data, filename):
        flat_rows = [self.flatten(d) for d in data]
        all_keys = list(OrderedDict.fromkeys(itertools.chain.from_iterable(r.keys() for r in flat_rows)))
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(flat_rows)

    def save_channels_csv(self, filename):
        base_keys = ["channel_id", "url", "title", "description", "avatar_url", "subscribers"]
        all_keys = list(base_keys)
        csv_rows = []
        for ch_data in self.channels.values():
            row = {k: ch_data.get(k) for k in base_keys}
            for k, v in (ch_data.get("meta") or {}).items():
                row[k] = v
                if k not in all_keys:
                    all_keys.append(k)
            csv_rows.append(row)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(csv_rows)

    def print_final_summary(self):
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Videos collected: {len(self.videos):,}")
        print(f"Video requests:   {self.video_stats.summary()}")
        if self.video_stats.errors:
            print(f"Video errors:     {dict(self.video_stats.errors)}")
        print("-" * 60)
        print(f"Channels scraped: {len(self.channels):,}")
        print(f"Channel requests: {self.channel_stats.summary()}")
        if self.channel_stats.errors:
            print(f"Channel errors:   {dict(self.channel_stats.errors)}")
        print("-" * 60)
        failed_channels = [ch_id for ch_id, data in self.channels.items() if "error" in data]
        print(f"Channels with errors in data: {len(failed_channels):,}")
        print("=" * 60)

    def save_all(self):
        print("Saving videos...")
        self.save_ndjson(self.videos.values(), "videos.ndjson")
        self.save_flat_csv(list(self.videos.values()), "videos.csv")
        print("Saving channels...")
        self.save_ndjson(self.channels.values(), "channels.ndjson")
        self.save_channels_csv("channels.csv")

    def run(self):
        self.setup_output_dir()
        print(f"Output: {self.output_dir}")
        print(f"Target: {self.target_videos:,} videos at ~{1/self.min_delay:.1f} RPS")
        print(f"Retries: {self.max_retries}, Report every: {self.report_every:,} vids")
        print("-" * 60)
        self.scrape_videos()
        print(f"\nCollected {len(self.videos):,} unique videos")
        unique_channels = len(self.extract_channel_ids())
        print(f"Found {unique_channels:,} unique channels to scrape")
        self.scrape_channels()
        self.save_all()
        self.print_final_summary()
        print(f"\nOutput saved to: {self.output_dir}/")


if __name__ == "__main__":
    scraper = RutubeScraper(output_dir="scraped_data\\v1_vids_and_channels_100k", target_videos=100000, rps=4, max_retries=3, report_every=1000)
    scraper.run()