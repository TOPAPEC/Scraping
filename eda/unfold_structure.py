import json
import pandas as pd
from collections import defaultdict
import os

def get_structure_schema(file_path, sample_size=10000):
    total_rows = 0
    stats = defaultdict(lambda: {'count': 0, 'types': set(), 'is_list': False})
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            
            try:
                data = json.loads(line)
                total_rows += 1
                
                def traverse(obj, path):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            traverse(v, f"{path}.{k}" if path else k)
                    elif isinstance(obj, list):
                        stats[path]['is_list'] = True
                        stats[path]['count'] += 1
                        stats[path]['types'].add('list')
                        if len(obj) > 0 and isinstance(obj[0], (dict, list)):
                            traverse(obj[0], f"{path}[0]")
                    else:
                        stats[path]['count'] += 1
                        if obj is None:
                            stats[path]['types'].add('NoneType')
                        else:
                            stats[path]['types'].add(type(obj).__name__)
                
                traverse(data, "")
                
            except json.JSONDecodeError:
                continue

    rows_list = []
    for path, info in stats.items():
        missing_pct = (1 - info['count'] / total_rows) * 100 if total_rows > 0 else 0
        type_str = ", ".join(sorted(info['types']))
        if info['is_list']:
            type_str = f"list (of {type_str})"
        
        rows_list.append({
            'Full Path': path,
            'Data Type': type_str,
            'Presence %': round(100 - missing_pct, 2),
            'Missing %': round(missing_pct, 2),
            'Sample Count': info['count']
        })
    
    return pd.DataFrame(rows_list).sort_values(by=['Presence %', 'Full Path'], ascending=[False, True])

def main():
    path_vid = 'scraped_data\\v1_vids_and_channels_100k\\videos.ndjson'
    path_ch = 'scraped_data\\v1_vids_and_channels_100k\\channels.ndjson'

    print(f"Analyzing: {path_vid}")
    df_vid = get_structure_schema(path_vid)
    
    print(f"\nAnalyzing: {path_ch}")
    df_ch = get_structure_schema(path_ch)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print("\n=== VIDEOS STRUCTURE ===")
        print(df_vid.to_string())
        
        print("\n\n=== CHANNELS STRUCTURE ===")
        print(df_ch.to_string())

    html_vid = df_vid.to_html(classes='table table-striped', index=False)
    html_ch = df_ch.to_html(classes='table table-striped', index=False)
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 12px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .table-striped tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Dataset Structure & Dtypes Analysis</h1>
        <h2>Videos (videos.ndjson)</h2>
        {html_vid}
        <h2>Channels (channels.ndjson)</h2>
        {html_ch}
    </body>
    </html>
    """
    
    with open("data_structure_analysis.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("\nAnalysis saved to data_structure_analysis.html")

if __name__ == "__main__":
    main()