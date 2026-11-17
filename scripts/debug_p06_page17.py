import pymupdf

doc = pymupdf.open("research-data/2604.12352v1.pdf")
page = doc[16] # page 17
imgs = page.get_image_info(xrefs=True)
print("P06 p.17 imgs:", len(imgs))
for i, img in enumerate(imgs):
    print(f"  img {i}: {img.get('bbox')}")
blocks = [b for b in page.get_text('blocks') if b[6] == 0]
for b in blocks:
    if 'Fig' in b[4] or 'fig' in b[4]:
        print(f"  caption: {b[:4]} -> {b[4][:50]}")
