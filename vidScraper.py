import csv, json, time, urllib.parse, requests, itertools
from collections import OrderedDict

START_FEED = "https://rutube.ru/api/feeds/tnt"
EXTRA_QS = {"origin__type": "rtb,rst", "ordering": "-created_ts"}
UA = {"User-Agent":"meta-crawler/1.0"}

def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=30, headers=UA)
    r.raise_for_status()
    return r.json()

def flatten(d, parent="", out=None):
    if out is None: out={}
    if isinstance(d, list):
        for i,v in enumerate(d):
            flatten(v, f"{parent}[{i}]" if parent else f"[{i}]", out)
    elif isinstance(d, dict):
        for k,v in d.items():
            nk = f"{parent}.{k}" if parent else k
            if isinstance(v,(dict,list)):
                flatten(v,nk,out)
            else:
                out[nk]=v
    else:
        out[parent]=d
    return out

from tqdm import tqdm
feed = get_json(START_FEED)
raw_items = []
for tab in feed.get("tabs", []):
    for res in tab.get("resources", []):
        if not res.get("url"): continue
        page_url = res["url"]
        first = True
        while page_url:
            if first:
                p = urllib.parse.urlsplit(page_url)
                q = dict(urllib.parse.parse_qsl(p.query))
                for k,v in EXTRA_QS.items():
                    q.setdefault(k, v)
                page_url = urllib.parse.urlunsplit((p.scheme,p.netloc,p.path,urllib.parse.urlencode(q),p.fragment))
                first = False
            data = get_json(page_url)
            for it in data.get("results", []):
                raw_items.append(it)
            page_url = data.get("next")
            time.sleep(0.25)
            if (len(raw_items) > 10):
                break

        if (len(raw_items) > 10):
            break

    if (len(raw_items) > 10):
        break

with open("rutube_videos.ndjson","w",encoding="utf-8") as f:
    for it in raw_items:
        f.write(json.dumps(it, ensure_ascii=False)+ "\n")

all_keys = {k for item in raw_items for k in flatten(item)}
print(list(all_keys))

flat_rows = [flatten(it) for it in raw_items]
all_keys = list(OrderedDict.fromkeys(itertools.chain.from_iterable(r.keys() for r in flat_rows)))
with open("rutube_videos.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=all_keys)
    w.writeheader()
    for r in flat_rows:
        w.writerow({k:r.get(k) for k in all_keys})

print(f"Saved {len(raw_items)} items to rutube_videos.ndjson and rutube_videos.csv")
