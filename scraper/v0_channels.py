import re, csv, json, time, requests
from urllib.parse import urlsplit
from bs4 import BeautifulSoup

CHANNEL_URLS = [
    "https://rutube.ru/channel/31845281/",
    "https://rutube.ru/channel/23338090/",
    "https://rutube.ru/channel/33621445/"
]
UA = {"User-Agent":"rutube-channel-crawler/1.0"}

def extract_channel_id(u):
    p = urlsplit(u)
    parts = [x for x in p.path.split("/") if x]
    return parts[1] if len(parts)>=2 and parts[0]=="channel" else ""

def norm_num(txt):
    if not txt: return None
    txt = txt.replace("\u00A0"," ").replace("\u202F"," ")
    m = re.search(r"(\d[\d\s]*)", txt)
    if not m: return None
    n = m.group(1).replace(" ","")
    try: return int(n)
    except: return None

def parse_channel(url):
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    title = None
    ogt = soup.find("meta", property="og:title")
    if ogt and ogt.get("content"): title = ogt["content"]
    if not title:
        h1 = soup.find("h1")
        if h1: title = h1.get_text(strip=True)
    desc = None
    md = soup.find("meta", attrs={"name":"description"})
    if md and md.get("content"): desc = md["content"]
    if not desc:
        ogd = soup.find("meta", property="og:description")
        if ogd and ogd.get("content"): desc = ogd["content"]
    avatar = None
    ogi = soup.find("meta", property="og:image")
    if ogi and ogi.get("content"): avatar = ogi["content"]
    subs = None
    body_txt = soup.get_text(" ", strip=True)
    m = re.search(r"(\d[\d\s]*)\s+подписчик\w*", body_txt, re.I)
    if m: subs = norm_num(m.group(0))
    meta = {}
    for mtag in soup.find_all("meta"):
        name = mtag.get("name") or ""
        prop = mtag.get("property") or ""
        key = f"meta.name:{name}" if name else (f"meta.property:{prop}" if prop else None)
        if key and mtag.get("content") is not None:
            meta[key]=mtag["content"]
    jsonld = []
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            jsonld.append(json.loads(s.string))
        except:
            pass
    return {
        "channel_id": extract_channel_id(url),
        "url": url,
        "title": title,
        "description": desc,
        "avatar_url": avatar,
        "subscribers": subs,
        "meta": meta,
        "jsonld": jsonld
    }

rows = []
for u in CHANNEL_URLS:
    try:
        rows.append(parse_channel(u))
    except Exception as e:
        rows.append({"channel_id": extract_channel_id(u), "url": u, "error": str(e)})
    time.sleep(0.35)

def get_keys(d, prefix=""):
    keys = set()
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.add(full_key)
            keys |= get_keys(v, full_key)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            keys |= get_keys(item, f"{prefix}[{i}]" if prefix else f"[{i}]")
    return keys

all_keys = {k for row in rows for k in get_keys(row)}
print(all_keys)

with open("rutube_channels.ndjson","w",encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False)+ "\n")

csv_rows = []
csv_keys = set(["channel_id","url","title","description","avatar_url","subscribers"])
for r in rows:
    base = {k:r.get(k) for k in ["channel_id","url","title","description","avatar_url","subscribers"]}
    for k,v in (r.get("meta") or {}).items():
        base[k]=v
        csv_keys.add(k)
    csv_rows.append(base)

csv_keys = list(csv_keys)
with open("rutube_channels.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=csv_keys)
    w.writeheader()
    for r in csv_rows:
        w.writerow({k:r.get(k) for k in csv_keys})

print(f"Saved {len(rows)} channels to rutube_channels.ndjson and rutube_channels.csv")
