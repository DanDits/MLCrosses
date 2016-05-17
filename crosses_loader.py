from PIL import Image
import numpy as np
import random
import tarfile

def load_data(path="/home/daniel/PycharmProjects/machinelearning/data/crosses/crosses.tar.gz",
              grey_transform=None,
              use_cancelled=False):

    # Types are the classes of crosses we want to classify
    # we got 21867 empty, 5575 crossed and only 66 cancelled in default dataset
    types = ["empty", "crossed"]
    if use_cancelled:
        types.append("cancelled")

    # Build mappings of these type names to the label values, to the vectorized labels and the actual data
    type_label_value = {cross_type: types.index(cross_type) for cross_type in types}
    type_label = {cross_type: _vectorized_label(type_label_value[cross_type], len(types)) for cross_type in types}

    # if data files are decompressed in the given path, much slower, needs "import os"
    #    in this case type_data would need to be read in another way
    # type_files = {cross_type: os.listdir(path + "work_type_" + cross_type) for cross_type in types}

    # Read the actual data, build tuples of (data, label, filename)
    tar = tarfile.open(path, mode="r:gz")
    type_data = {cross_type:[] for cross_type in types}
    for member in tar.getmembers():
        if member.isfile():
            f = tar.extractfile(member)
            for cross_type in types:
                if member.name.startswith("work_type_" + cross_type):
                    label = type_label[cross_type]
                    array = _load_image_as_array(f, grey_transform)
                    type_data[cross_type].append((array, label, member.name))
                    break
    tar.close()

    # Now randomly draw a fraction of the data to use for testing the trained network
    test_data_fraction = 0.25
    training_data = []
    test_data = []

    # shuffle type data to not have fixed test data
    for cross_type in types:
        random.shuffle(type_data[cross_type])
    for cross_type in types:
        data = type_data[cross_type]
        sep = int(test_data_fraction * len(data))
        test_data.extend((data[:sep]))
        training_data.extend(data[sep:])

    # Shuffle training data so that it is not sorted by types
    random.shuffle(training_data)
    random.shuffle(test_data)

    # Some debug information about the size of training and test data
    print("Trainingdata size:", len(training_data), "Test data size:", len(test_data))
    print("Training data empty samples:", sum(1 if np.argmax(data[1],axis=0) == type_label_value["empty"] else 0 for data in training_data))
    print("Training data crossed samples:",
          sum(1 if np.argmax(data[1],axis=0) == type_label_value["crossed"] else 0 for data in training_data))
    print("Test data empty samples:",
          sum(1 if np.argmax(data[1],axis=0) == type_label_value["empty"] else 0 for data in test_data))
    print("Test data crossed samples:",
          sum(1 if np.argmax(data[1],axis=0) == type_label_value["crossed"] else 0 for data in test_data))

    return training_data, test_data


def _load_image_as_array(image_file, grey_transform):
    img = Image.open(image_file)
    pixels = img.load()
    width, height = img.size
    pixel_list = []
    for i in range(width):
        for j in range(height):
            # Important! Normalize pixel data to range [0,1] by dividing by 255.
            value = sum(pixels[i, j]) / (3. * 255.) # as the images are RGB and not completely in shades of grey
            if grey_transform:
                value = grey_transform(value)
            pixel_list.append(value)
    array = np.array(pixel_list, dtype="float64", copy=False)
    return array.reshape((len(pixel_list), 1))


def _vectorized_label(j, size):
    e = np.zeros((size, 1))
    e[j] = 1.0
    return e