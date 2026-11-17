import pymupdf

doc = pymupdf.open('research-data/2606.18075v1.pdf')
for pidx, p in enumerate(doc):
    blocks = [b for b in p.get_text('blocks') if b[6] == 0]
    fig_blocks = [b for b in blocks if 'Fig' in b[4] or 'Table' in b[4]]
    if fig_blocks:
        print(f"P07 p.{pidx+1}:")
        for b in fig_blocks:
            print(f"   text: {repr(b[4][:60])}")
