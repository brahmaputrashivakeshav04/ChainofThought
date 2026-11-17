import pymupdf

doc = pymupdf.open("research-data/2603.24925v3.pdf")
page = doc[4]
blocks = page.get_text('blocks')
for b in sorted(blocks, key=lambda x: x[1]):
    if b[1] < 160:
        print(f"y0={b[1]:.1f}, y1={b[3]:.1f}: {repr(b[4][:80])}")
