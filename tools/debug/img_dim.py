from PIL import Image

img = Image.open("/home/introai21/mmtracking/data/CEUS/Data/fold_1/aug/FNH/16045645_2012/AP/ser005img00071_aug.png")
print(img.size)   # (width, height)
print(img.mode)   # 'L', 'RGB' 등