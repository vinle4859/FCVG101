"""Dữ liệu tổng hợp tối thiểu để kiểm thử quy trình CHỨ KHÔNG PHẢI dataset coursework.
KHÔNG sử dụng accuracy từ dữ liệu này để báo cáo chất lượng nhận dạng thực tế.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random

r = random.Random(2026)
for label in ['apple', 'banana', 'orange']:
    dest = Path('data/raw') / label
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        im = Image.new('RGB', (150, 150), (r.randint(230, 255),) * 3)
        draw = ImageDraw.Draw(im)
        x, y = r.randint(30, 50), r.randint(30, 48)
        if label == 'apple':
            draw.ellipse((x, y, x+70, y+75), fill=(190+r.randint(0,35), 35, 35))
            draw.line((x+35,y-8,x+38,y+8), fill=(55,85,20), width=5)
        elif label == 'orange':
            draw.ellipse((x, y, x+74, y+74), fill=(245, 115+r.randint(0,30), 30))
            draw.ellipse((x+30,y-6,x+48,y+6), fill=(35,130,65))
        else:
            draw.arc((x-15,y-10,x+90,y+100), start=10, end=150,
                     fill=(235,195,30), width=19)
            draw.arc((x-10,y-10,x+88,y+98), start=13, end=140,
                     fill=(225,175,25), width=8)
        im = im.filter(ImageFilter.GaussianBlur(radius=r.choice([0,0.3,0.6])))
        im.save(dest / f'obj{i:03d}__view1.jpg')
print('Đã tạo 45 ảnh TỔNG HỢP để smoke-test; thay bằng dữ liệu thật khi làm coursework.')
