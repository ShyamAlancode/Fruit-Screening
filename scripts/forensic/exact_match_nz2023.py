import pandas as pd

df = pd.read_csv(r'd:\CFRD\New folder\data_NZ2023.csv')

folders_to_check = [
    ('cox_batch19', '2023-03-03_004'),
    ('cox_batch19', '2023-03-03_005'),
    ('cox_batch7', '2023-03-03_036'),
    ('cox_batch7', '2023-03-03_037'),
    ('gala_batch13', '2023-03-03_053'),
    ('gala_batch13', '2023-03-03_054'),
    ('gala_batch13', '2023-03-03_055'),
    ('gala_batch13', '2023-03-04_013'),
    ('gala_batch13', '2023-03-04_014'),
]

cols = ['Image_folder_A', 'Image_folder_B', 'Image_folder_C', 'Image_folder_D']

print("=== EXACT MATCHES IN data_NZ2023.csv ===")
for batch, f in folders_to_check:
    cultivar_hint = "Cox" if "cox" in batch else "Gala"
    found = False
    for c in cols:
        m = df[df[c].astype(str).str.contains(f) & df['Cultivar'].str.contains(cultivar_hint)]
        if len(m) > 0:
            found = True
            print(f"\nFolder {f} ({batch}) matched in column {c}: {len(m)} rows")
            for idx, r in m.iterrows():
                print(f"  Row {idx}: ID={r['ID']}, Number={r['Number']}, Cultivar={r['Cultivar']}, Brix={r['Brix']}, Firmness={r['Avg_firmness']}")
                print(f"    Paths: A={r['Image_folder_A'].split('/')[-2] if pd.notna(r['Image_folder_A']) else 'None'}, D={r['Image_folder_D'].split('/')[-2] if pd.notna(r['Image_folder_D']) else 'None'}")
            break
    if not found:
        print(f"\nFolder {f} ({batch}): NO MATCH with cultivar {cultivar_hint}!")
