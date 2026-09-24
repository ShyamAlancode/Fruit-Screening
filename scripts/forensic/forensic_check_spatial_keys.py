import os, glob
import xml.etree.ElementTree as ET
import pandas as pd

print("=== 1. CHECKING ALL XML KEYS IN ALL EXTRACTED DIRECTORIES ===")
extracted_dir = r"d:\CFRD\New folder\extracted"
xml_keys = set()
for root, dirs, files in os.walk(extracted_dir):
    for f in files:
        if f.endswith('.xml'):
            p = os.path.join(root, f)
            try:
                tree = ET.parse(p)
                for elem in tree.iter():
                    xml_keys.add((elem.tag, str(elem.attrib)))
            except: pass
for k in sorted(list(xml_keys)):
    print("XML tag/attrib:", k)

print("\n=== 2. CHECKING ALL HDR HEADERS FOR NON-STANDARD FIELDS ===")
hdr_fields = set()
for root, dirs, files in os.walk(extracted_dir):
    for f in files:
        if f.endswith('.hdr'):
            p = os.path.join(root, f)
            try:
                with open(p, 'r') as hf:
                    for line in hf:
                        if '=' in line:
                            field = line.split('=')[0].strip()
                            hdr_fields.add(field)
            except: pass
print("HDR fields:", sorted(list(hdr_fields)))

print("\n=== 3. CHECKING ALL COLUMNS IN ALL 3 CSVS ===")
for name in ['data_NZ2023.csv', 'data_NZ2024.csv', 'data_UK2024.csv']:
    p = os.path.join(r"d:\CFRD\New folder", name)
    df = pd.read_csv(p)
    print(name, ":", df.columns.tolist())
