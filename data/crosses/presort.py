import os, re
files = filter(lambda file: file.endswith(".png"), os.listdir())
file_value = {file: float(re.findall("_(.[^_]+)___", file)[0]) for file in files}

for file, value in file_value.items():
    if value <= 50.:
        os.rename(file, "/home/daniel/PycharmProjects/machinelearning/data/crosses/type_empty/" + file)
    elif value <= 750:
        os.rename(file, "/home/daniel/PycharmProjects/machinelearning/data/crosses/type_crossed/" + file)
    else:        
        os.rename(file, "/home/daniel/PycharmProjects/machinelearning/data/crosses/type_cancelled/" + file)