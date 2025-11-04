from color_utils import *
import numpy as np

def convert_color_scale(color_01_scale):
    for rgb_element in color_01_scale:
        print(rgb_element * 255)


a = np.array([[[0.8, 0.2, 0.2]]], dtype=np.float32)  # light red
b = np.array([[[0.8, 0.25, 0.2]]], dtype=np.float32) # light red variant

# convert_color_scale([0.8, 0.2, 0.2])
# convert_color_scale([0.8, 0.25, 0.2])

a_lab = rgb_to_lab(a); b_lab = rgb_to_lab(b)
for i in deltaE_2000(a_lab, b_lab):
    print(i[0])
