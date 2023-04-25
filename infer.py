import os
import cv2
import random
im = '104182.jpg'

os.system(f'python tools/infer.py --yaml chromosome.yaml --weights best_model.pt --source {im} --save-txt')

classes = ["A1", "A2", "A3", "B4", "B5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "D13", "D14", "D15", "E16", "E17", "E18", "F19", "F20", "G21", "G22", "X", "Y"]
color_list = [random.choices(range(256), k=3) for i in range(len(classes))]

def draw_boxes(image_path, model, class_to_display):
    image = cv2.imread(image_path)
    h, w = image.shape
    return None
