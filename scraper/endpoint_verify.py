# -*- coding: utf-8 -*-
import time
import json
import requests
from urllib.parse import quote, urlsplit, urlunsplit, parse_qsl, urlencode

BASE = "https://rutube.ru"
API = "https://rutube.ru/api"

FEED_NAMES = ["main", "tnt", "sport", "movie", "blogger", "news"]

SEARCH_TERMS = [
    "новости", "фильм", "сериал", "музыка", "спорт", "игры", "обзор",
    "рецепт", "авто", "политика", "экономика", "технологии", "наука",
    "история", "путешествия", "юмор", "дети", "животные", "природа",
    "ремонт", "строительство", "мода", "красота", "здоровье", "фитнес"
]

def add_qs(url, extra):
    p = urlsplit(url)
    q = dict(parse_qsl(p.query))
    for k, v in extra.items():
        q.setdefault(k, v)
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(q), p.fragment))

def short_json_shape(x):
    if isinstance(x, dict):
        keys = list(x.keys())[:12]
        return f"dict keys={keys}"
    if isinstance(x, list):
        return f"list len={len(x)}"
    return f"type={type(x).__name__}"

def probe(session, url, expect):
    out = {"url": url, "final_url": None, "status": None, "ok": False, "detail": None}
    try:
        r = session.get(url, timeout=25, allow_redirects=True)
        out["status"] = r.status_code
        out["final_url"] = r.url
        if expect == "json":
            try:
                data = r.json()
            except Exception as e:
                out["detail"] = f"json_parse_error:{type(e).__name__}"
                return out, None
            out["ok"] = 200 <= r.status_code < 400
            out["detail"] = short_json_shape(data)
            return out, data
        if expect == "html":
            txt = r.text or ""
            out["ok"] = 200 <= r.status_code < 400 and len(txt.strip()) > 0
            out["detail"] = f"html_len={len(txt)}"
            return out, txt
        out["ok"] = 200 <= r.status_code < 400
        out["detail"] = "raw"
        return out, r.text
    except Exception as e:
        out["detail"] = f"request_error:{type(e).__name__}"
        return out, None

def extract_results(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("results", [])
    return []

def safe_collect_feed_resource_urls(feed_json):
    urls = []
    if not isinstance(feed_json, dict):
        return urls
    for tab in feed_json.get("tabs", []) or []:
        for res in tab.get("resources", []) or []:
            u = res.get("url")
            if isinstance(u, str) and u:
                urls.append(u)
    seen = set()
    dedup = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup

def collect_tags_and_authors_from_results(items):
    tags = []
    author_ids = []
    for it in items:
        if not isinstance(it, dict):
            continue
        for t in it.get("tags", []) or []:
            if isinstance(t, dict) and t.get("name"):
                tags.append(t["name"])
            elif isinstance(t, str) and t:
                tags.append(t)
        a = it.get("author") or {}
        if isinstance(a, dict) and a.get("id"):
            author_ids.append(a["id"])
    def uniq(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    tags = [t for t in uniq(tags) if isinstance(t, str) and len(t) > 2]
    author_ids = [x for x in uniq(author_ids) if isinstance(x, int)]
    return tags, author_ids

def main():
    s = requests.Session()
    s.headers.update({"User-Agent": "rutube-link-probe/1.0", "Accept": "application/json,text/html;q=0.9,*/*;q=0.8"})

    checks = []
    seen = set()

    tags_pool = ["новости"]
    author_pool = [23463954]
    cat_pool = [1, 2, 3]

    def add_check(url, expect):
        if url in seen:
            return
        seen.add(url)
        checks.append((url, expect))

    add_check(f"{API}/video/category/", "json")

    for cid in cat_pool:
        add_check(add_qs(f"{API}/video/category/{cid}/", {"per_page": "100"}), "json")

    feed_jsons = []
    for name in FEED_NAMES:
        add_check(f"{API}/feeds/{name}/", "json")

    for term in SEARCH_TERMS:
        add_check(add_qs(f"{API}/video/search/?query={quote(term)}", {"per_page": "100"}), "json")

    for t in tags_pool:
        add_check(add_qs(f"{API}/tags/{quote(t)}/video/", {"per_page": "100"}), "json")

    for aid in author_pool:
        add_check(add_qs(f"{API}/video/person/{aid}/", {"per_page": "100"}), "json")

    for aid in author_pool:
        add_check(f"{BASE}/channel/{aid}/", "html")

    results = []
    discovered_resource_urls = []
    discovered_tags = []
    discovered_authors = []
    discovered_cat_ids = []

    for i, (url, expect) in enumerate(checks, 1):
        res, data = probe(s, url, expect)
        results.append(res)
        status = res["status"]
        ok = "OK" if res["ok"] else "FAIL"
        fin = res["final_url"] or ""
        det = res["detail"] or ""
        print(f"{i:03d} {ok} {status} {url} -> {fin} | {det}")
        time.sleep(0.25)

        if expect == "json" and isinstance(data, dict):
            if url.endswith("/video/category/") or url.endswith("/video/category/"):
                for c in extract_results(data):
                    if isinstance(c, dict) and isinstance(c.get("id"), int):
                        discovered_cat_ids.append(c["id"])
            if "/feeds/" in url and "/api/feeds/" in (res["final_url"] or url):
                feed_jsons.append(data)

        if expect == "json":
            items = extract_results(data) if isinstance(data, (dict, list)) else []
            tgs, aids = collect_tags_and_authors_from_results(items)
            discovered_tags.extend(tgs[:50])
            discovered_authors.extend(aids[:50])

    for fj in feed_jsons:
        discovered_resource_urls.extend(safe_collect_feed_resource_urls(fj))

    def uniq(seq):
        seen2 = set()
        out = []
        for x in seq:
            if x not in seen2:
                seen2.add(x)
                out.append(x)
        return out

    discovered_resource_urls = uniq([u for u in discovered_resource_urls if isinstance(u, str) and u.startswith("http")])
    discovered_tags = uniq([t for t in discovered_tags if isinstance(t, str) and len(t) > 2])[:10]
    discovered_authors = uniq([a for a in discovered_authors if isinstance(a, int)])[:10]
    discovered_cat_ids = uniq([c for c in discovered_cat_ids if isinstance(c, int)])[:10]

    extra_checks = []
    for u in discovered_resource_urls[:20]:
        extra_checks.append((add_qs(u, {"per_page": "100"}), "json"))

    for cid in discovered_cat_ids[:5]:
        extra_checks.append((add_qs(f"{API}/video/category/{cid}/", {"per_page": "100"}), "json"))

    for t in discovered_tags:
        extra_checks.append((add_qs(f"{API}/tags/{quote(t)}/video/", {"per_page": "100"}), "json"))

    for aid in discovered_authors:
        extra_checks.append((add_qs(f"{API}/video/person/{aid}/", {"per_page": "100"}), "json"))
        extra_checks.append((f"{BASE}/channel/{aid}/", "html"))

    extra_seen = set()
    extra_checks_dedup = []
    for u, ex in extra_checks:
        if u not in extra_seen:
            extra_seen.add(u)
            extra_checks_dedup.append((u, ex))

    if extra_checks_dedup:
        print("\n--- EXTRA (discovered from feeds/results) ---")
    for i, (url, expect) in enumerate(extra_checks_dedup, 1):
        res, _ = probe(s, url, expect)
        status = res["status"]
        ok = "OK" if res["ok"] else "FAIL"
        fin = res["final_url"] or ""
        det = res["detail"] or ""
        print(f"X{i:03d} {ok} {status} {url} -> {fin} | {det}")
        time.sleep(0.25)

if __name__ == "__main__":
    main()
