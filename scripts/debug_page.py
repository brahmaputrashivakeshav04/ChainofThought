import pymupdf

def debug_p05():
    doc = pymupdf.open('research-data/2604.01733v1.pdf')
    page = doc[5] # page 6
    imgs = page.get_image_info(xrefs=True)
    print("P05 p.6 images count:", len(imgs))
    for i, img in enumerate(imgs):
        print(f"  img {i}: bbox={img['bbox']}, size=({img['width']}, {img['height']})")
    blocks = [b for b in page.get_text('blocks') if b[6] == 0]
    for b in blocks:
        if 'Fig' in b[4]:
            print(f"  caption at ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}): {b[4][:50]}")

def debug_p03():
    doc = pymupdf.open('research-data/2602.10271v3.pdf')
    for pidx in [6, 7]:
        page = doc[pidx]
        print(f"\nP03 p.{pidx+1}:")
        imgs = page.get_image_info(xrefs=True)
        print(f"  images count: {len(imgs)}")
        for i, img in enumerate(imgs):
            print(f"    img {i}: bbox={img['bbox']}, size=({img['width']}, {img['height']})")
        blocks = [b for b in page.get_text('blocks') if b[6] == 0]
        for b in blocks:
            if 'Fig' in b[4]:
                print(f"    caption at ({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}): {b[4][:50]}")

if __name__ == '__main__':
    debug_p05()
    debug_p03()
