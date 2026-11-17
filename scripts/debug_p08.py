import pymupdf

doc = pymupdf.open('research-data/s44163-026-02023-3.pdf')
for pidx, p in enumerate(doc):
    imgs = p.get_image_info(xrefs=True)
    blocks = [b for b in p.get_text('blocks') if b[6] == 0]
    fig_blocks = [b for b in blocks if 'Fig' in b[4] or 'fig' in b[4]]
    if imgs or fig_blocks:
        print(f"P08 p.{pidx+1}: {len(imgs)} imgs, {len(fig_blocks)} fig text blocks")
        for b in fig_blocks:
            print(f"   text: {repr(b[4][:60])}")
