from PIL import Image
import os
path = os.path.join(r"c:\\Users\\sevan\\Documents\\GitHub\\vlmocr", "test.png")
img = Image.open(path).convert("RGB")
pixels = img.getdata()
total = img.width * img.height
non_white = sum(1 for p in pixels if p != (255,255,255))
percentage = non_white / total * 100
print(f"total_pixels={total}")
print(f"non_white_pixels={non_white}")
print(f"non_white_percentage={percentage:.6f}")