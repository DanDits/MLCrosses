import os
from PIL import Image
import numpy as np
import random


def load_data(path="/home/daniel/PycharmProjects/machinelearning/data/crosses/", use_cancelled=False):
    types = ["empty", "crossed"]
    if use_cancelled:
        types.append("cancelled")
    type_files = {cross_type: os.listdir(path + "work_type_" + cross_type) for cross_type in types}
    # we got 21867 empty, 5575 crossed and only 66 cancelled in default dataset
    type_label_value = {cross_type: types.index(cross_type) for cross_type in types}
    type_label = {cross_type: _vectorized_label(type_label_value[cross_type], len(types)) for cross_type in types}

    # by default load all available files and convert to array
    type_data = {}
    for cross_type in types:
        label = type_label[cross_type]
        data = []
        type_data[cross_type] = data
        for file in type_files[cross_type]:
            array = _load_image_as_array(path + "work_type_" + cross_type + "/" + file)
            data.append((array, label))

    test_data_fraction = 0.25
    training_data = []
    test_data = []

    # shuffle type data to not have fixed test data
    for cross_type in types:
        random.shuffle(type_data[cross_type])
    #type_data["empty"] = type_data["empty"][:int(len(type_data["crossed"]) * 1.2)]
    for cross_type in types:
        data = type_data[cross_type]
        sep = int(test_data_fraction * len(data))
        test_data.extend((data[:sep]))
        training_data.extend(data[sep:])

    # Shuffle training data so that it is not sorted by types
    random.shuffle(training_data)
    random.shuffle(test_data)

    print("Trainingdata size:", len(training_data), "Test data size:", len(test_data))
    print("Training data empty samples:", sum(1 if np.argmax(data[1],axis=0) == type_label_value["empty"] else 0 for data in training_data))
    print("Training data crossed samples:",
          sum(1 if np.argmax(data[1],axis=0) == type_label_value["crossed"] else 0 for data in training_data))
    print("Test data empty samples:",
          sum(1 if np.argmax(data[1],axis=0) == type_label_value["empty"] else 0 for data in test_data))
    print("Test data crossed samples:",
          sum(1 if np.argmax(data[1],axis=0) == type_label_value["crossed"] else 0 for data in test_data))

    return training_data, test_data

def _load_image_as_array(image_path):
    img = Image.open(image_path)
    pixels = img.load()
    width, height = img.size
    pixel_list = []
    for i in range(width):
        for j in range(height):
            pixel_list.append(sum(pixels[i, j]) / (3. * 255.)) # as the images are RGB and not completely in shades of grey

    array = np.array(pixel_list, dtype="float64", copy=False)
    return array.reshape((len(pixel_list), 1))


def _vectorized_label(j, size):
    e = np.zeros((size, 1))
    e[j] = 1.0
    return e