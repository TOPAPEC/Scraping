import json
import math
from collections import Counter, defaultdict
from tqdm import tqdm

PATH = r"scraped_data\v1_vids_and_channels_100k\videos.ndjson"
TOP_N = 200
PER_PAGE = 50

def main():
    by_cat = Counter()
    name_by_id = defaultdict(Counter)

    with open(PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading ndjson", unit="line", dynamic_ncols=True):
            item = json.loads(line)
            cat = item.get("category") or {}
            cid = cat.get("id")
            cname = cat.get("name")
            if cid is None:
                continue
            by_cat[cid] += 1
            if cname:
                name_by_id[cid][cname] += 1

    print(f"videos_total_with_category={sum(by_cat.values())}")
    print(f"unique_categories={len(by_cat)}\n")

    print("category_id\tcategory_name\tvideos\tpages_50")
    for cid, c in by_cat.most_common(TOP_N):
        cname = name_by_id[cid].most_common(1)[0][0] if name_by_id[cid] else ""
        pages = math.ceil(c / PER_PAGE)
        print(f"{cid}\t{cname}\t{c}\t{pages}")

if __name__ == "__main__":
    main()