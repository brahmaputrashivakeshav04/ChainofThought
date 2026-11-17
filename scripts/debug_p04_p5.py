import pymupdf

doc = pymupdf.open("research-data/2603.24925v3.pdf")
page = doc[4] # page 5
tabs = page.find_tables()
print("P04 p.5 find_tables count:", len(tabs.tables))
for i, t in enumerate(tabs.tables):
    print(f"Table {i} bbox: {t.bbox}")
    for row in t.extract()[:5]:
        print("  ", row)
blocks = [b for b in page.get_text('blocks') if b[6] == 0]
for b in blocks:
    if 'Table' in b[4]:
        print("Block at", b[:4], "->", repr(b[4][:50]))
