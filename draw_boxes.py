import cv2
import random
import pandas as pd
import collections
im = cv2.imread('104182.jpg')

dh, dw, _ = im.shape
class_to_display = "D13"
classes = ["A1", "A2", "A3", "B4", "B5", "C6", "C7", "C8", "C9", "C10", "C11", "C12",
           "D13", "D14", "D15", "E16", "E17", "E18", "F19", "F20", "G21", "G22", "X", "Y"]
color_list = [random.choices(range(256), k=3) for i in range(len(classes))]

with open('104182.txt') as f:
    a = f.read().splitlines()

df = pd.read_csv('104182.txt', sep=" ", header=None)
df.columns = ["class", "x", "y", "w", "h", "conf"]
df = df[['class', 'conf']]
df = df.sort_values('class')
total_chromosomes = len(df)
ch13 = (df['class'] == 12).sum()
ch18 = (df['class'] == 17).sum()
ch21 = (df['class'] == 20).sum()
ch23 = (df['class'] == 22).sum()
ch24 = (df['class'] == 23).sum()

if total_chromosomes == 47 and ch13 >= 3:
    print('Trysomy 13')
elif total_chromosomes == 47 and ch18 >= 3:
    print('Trysomy 18')
elif total_chromosomes == 47 and ch21 >= 3:
    print('Trysomy 21')
elif total_chromosomes == 47 and ch23 >= 2 and ch24 >= 1:
    print('Trisomy XXY')
elif total_chromosomes == 45 and ch23 == 1 and ch24 == 0:
    print('Monosomy X')
else:
    print('Normal')
print(ch13, ch18, ch21, ch23, ch24)


for dt in a:
    a = dt.split(' ')
    if class_to_display == "all":
        class_index = int(a[0])
        x, y, w, h = float(a[1]), float(a[2]), float(a[3]), float(a[4])
        conf = round(float(a[5])*100, 2)
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        color = random.choices(range(256), k=3)
        cv2.rectangle(im, (l, t), (r, b), color_list[class_index], 2)
        cv2.putText(im, f'{classes[class_index]}: {conf}%', (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[class_index], 2)
    else:
        if int(a[0]) == classes.index(class_to_display):
            x, y, w, h = float(a[1]), float(a[2]), float(a[3]), float(a[4])
            conf = round(float(a[5])*100, 2)
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)
            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1

            cv2.rectangle(im, (l, t), (r, b), (36, 255, 12), 2)
            cv2.putText(im, f'{class_to_display}: {conf}%', (l, t-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
cv2.imshow('im', im)
cv2.waitKey(0)
