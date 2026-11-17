import pymupdf

doc = pymupdf.open('research-data/2604.12352v1.pdf')
page = doc[19] # page 20
tabs = page.find_tables()
print("P06 p.20 tables count:", len(tabs.tables))
for i, tab in enumerate(tabs.tables):
    print(f"Table {i} bbox: {tab.bbox}")
    data = tab.extract()
    print(f"  Rows: {len(data)}, Cols: {len(data[0]) if data else 0}")
    print(f"  First row: {data[0] if data else []}")
blocks = [b for b in page.get_text('blocks') if b[6] == 0]
for b in blocks:
    if 'Table' in b[4]:
        print(f"Caption block at {b[:4]}: {b[4][:60]}")
