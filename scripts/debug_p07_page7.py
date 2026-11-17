import pymupdf

doc = pymupdf.open("research-data/2606.18075v1.pdf")
page = doc[6]
imgs = page.get_image_info(xrefs=True)
print("P07 p.7 imgs:", len(imgs))
for i, img in enumerate(imgs):
    print(f"  img {i}: {img.get('bbox')}")
blocks = [b for b in page.get_text('blocks') if b[6] == 0]
for b in blocks:
    if 'Figure' in b[4]:
        print(f"  caption: {b[:4]} -> {b[4][:50]}")
