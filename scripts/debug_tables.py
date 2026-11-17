import pymupdf

def check_p04():
    doc = pymupdf.open('research-data/2603.24925v3.pdf')
    for pidx, page in enumerate(doc):
        tabs = page.find_tables()
        if tabs and tabs.tables:
            print(f"P04 p.{pidx+1} has {len(tabs.tables)} tables")
        blocks = [b for b in page.get_text('blocks') if b[6] == 0]
        for b in blocks:
            if 'Table' in b[4]:
                print(f"  P04 p.{pidx+1} text: {b[4][:50]}")

def check_p06():
    doc = pymupdf.open('research-data/2604.12352v1.pdf')
    page = doc[19] # page 20
    tabs = page.find_tables()
    print(f"\nP06 p.20 tables found: {len(tabs.tables) if tabs else 0}")
    blocks = [b for b in page.get_text('blocks') if b[6] == 0]
    for b in blocks:
        if 'Table 16' in b[4]:
            print(f"  P06 p.20: {b[4][:80]}")

if __name__ == '__main__':
    check_p04()
    check_p06()
