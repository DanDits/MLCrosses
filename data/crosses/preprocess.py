import os
from PIL import Image
base_path = "/home/daniel/PycharmProjects/machinelearning/data/crosses/type_"
tar_base_path = "/home/daniel/PycharmProjects/machinelearning/data/crosses/work_type_"
pixel_dimension = 40 # for width and height of output working data

for cross_type in ["empty", "crossed", "cancelled"]:
    for file in os.listdir(base_path + cross_type):
        img = Image.open(base_path + cross_type + "/" + file)
        img = img.resize((pixel_dimension, pixel_dimension))
        img.save(tar_base_path + cross_type + "/" + file)