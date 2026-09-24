import os, glob
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage
import pandas as pd
import json

# 1. Load CSVs
csv_nz23 = pd.read_csv(r'd:\CFRD\New folder\data_NZ2023.csv')
csv_nz24 = pd.read_csv(r'd:\CFRD\New folder\data_NZ2024.csv')
csv_uk24 = pd.read_csv(r'd:\CFRD\New folder\data_UK2024.csv')

extracted_root = Path(r'd:\CFRD\New folder\extracted')
sample_dirs = sorted([d for d in extracted_root.rglob('202*') if d.is_dir() and not d.name.startswith('results') and not d.name.startswith('capture') and not d.name.startswith('metadata')])

records = []

for sdir in sample_dirs:
    folder_name = sdir.name
    batch_name = sdir.parent.name
    
    # Read XML
    desc = ""
    for xml_p in sdir.glob('metadata/*.xml'):
        try:
            tree = ET.parse(xml_p)
            for elem in tree.iter('key'):
                if elem.get('field') == 'description':
                    desc = elem.text.strip()
        except: pass
        
    # Find image (preview PNG or results PNG)
    png_path = sdir / f"{folder_name}.png"
    if not png_path.exists():
        pngs = list(sdir.glob("*.png"))
        if pngs: png_path = pngs[0]
        else: continue
        
    # Segment blobs
    im = Image.open(png_path).convert('RGB')
    arr = np.array(im)
    gray = arr.mean(axis=2)
    mask = gray > 25
    labeled, num_features = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)
    
    apple_boxes = []
    for i, obj in enumerate(objects):
        sz = np.sum(labeled[obj] == (i + 1))
        y_slice, x_slice = obj
        if sz > 2000 and y_slice.start > 50:
            apple_boxes.append({
                'x_start': int(x_slice.start),
                'x_stop': int(x_slice.stop),
                'x_center': int((x_slice.start + x_slice.stop) / 2),
                'y_start': int(y_slice.start),
                'y_stop': int(y_slice.stop),
                'size_px': int(sz)
            })
            
    apple_boxes.sort(key=lambda b: b['x_center'])
    
    # Query CSV
    # Search across all 3 CSVs for this folder name in any Image_folder column
    matched_rows = []
    csv_source = ""
    for name, df in [('NZ2024', csv_nz24), ('NZ2023', csv_nz23), ('UK2024', csv_uk24)]:
        cols = [c for c in ['Image_folder_A', 'Image_folder_B', 'Image_folder_C', 'Image_folder_D'] if c in df.columns]
        for c in cols:
            m = df[df[c].astype(str).str.contains(folder_name, na=False)]
            if len(m) > 0:
                # filter to exact batch / date if possible
                matched_rows = m
                csv_source = name
                break
        if len(matched_rows) > 0:
            break
            
    row_info = []
    if len(matched_rows) > 0:
        for idx, r in matched_rows.iterrows():
            row_info.append({
                'csv_idx': int(r.get('index', idx)),
                'ID': str(r.get('ID', '')),
                'Number': int(r.get('Number', -1)) if pd.notna(r.get('Number')) else -1,
                'Brix': float(r.get('Brix', 0.0)),
                'Avg_firmness': float(r.get('Avg_firmness', 0.0)),
                'Starch': float(r.get('Starch', 0.0))
            })
            
    records.append({
        'batch': batch_name,
        'folder': folder_name,
        'xml_desc': desc,
        'csv_source': csv_source,
        'num_blobs': len(apple_boxes),
        'blob_positions': apple_boxes,
        'csv_num_rows': len(row_info),
        'csv_rows': row_info
    })

with open('records_audit.json', 'w') as f:
    json.dump(records, f, indent=2)

print(f"Processed {len(records)} sample frames.")
for r in records:
    print(f"\nFolder: {r['folder']} ({r['batch']}) | Blobs detected: {r['num_blobs']} | CSV rows: {r['csv_num_rows']}")
    print(f"  XML Desc: {repr(r['xml_desc'])}")
    if r['csv_rows']:
        print(f"  CSV IDs: {[x['ID'] for x in r['csv_rows']]}")
        print(f"  CSV Numbers: {[x['Number'] for x in r['csv_rows']]}")
        print(f"  CSV Brix: {[x['Brix'] for x in r['csv_rows']]}")
    print(f"  Blob X-centers: {[b['x_center'] for b in r['blob_positions']]}")
