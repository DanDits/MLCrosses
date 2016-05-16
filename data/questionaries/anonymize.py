# -*- coding: utf-8 -*-
"""
Created on Thu May  5 18:12:44 2016

@author: daniel
"""
from PIL import Image
import os
base_path = "/home/daniel/PycharmProjects/machinelearning/data/questionaries/"
for file in filter(lambda file: file.endswith(".jpg"), 
                   os.listdir(base_path)):
    img = Image.open(base_path + file)
    width, height = img.size
    pixels = img.load()
    
    # anonymize top left corner containing some company and course info
    anonym_shade = 0
    for y in range(610):
        for x in range(1000):
            pixels[x, y] = anonym_shade
            
    # anonymize handwritten answers on the bottom by cropping
    img = img.crop((0, 0, width, 2830))
    img.save(base_path + "anonym/" + file)