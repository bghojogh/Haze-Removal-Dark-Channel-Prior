import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm
import numpy as np
from PIL import Image
import colorsys
from MyImage_class import *
import math
import os, os.path


class RainSnowDetector:

    def __init__(self):
        pass

    def divide_image_into_boxes(self, vertical_edge_map, myImage_object, size_of_box=10, save_path='./', format_of_save='jpg'):
        number_of_rows = vertical_edge_map.shape[0]
        number_of_columns = vertical_edge_map.shape[1]
        for row in range(0, (math.floor(number_of_rows/size_of_box))*size_of_box, size_of_box):   # ignore the last box if it is fractional
            for column in range(0, (math.floor(number_of_columns/size_of_box))*size_of_box, size_of_box):   # ignore the last box if it is fractional
                box = vertical_edge_map[row:row+size_of_box, column:column+size_of_box]
                box_showable = box * 255
                box_image = myImage_object.numpyArray2pilImage(array=box_showable)
                number_of_saved_boxes_in_folder = myImage_object.count_number_of_files_in_folder(folder_path=save_path)
                name_of_box = str(number_of_saved_boxes_in_folder)
                myImage_object.save_image(image=box_image, name=name_of_box, save_path=save_path, format_of_save=format_of_save)

    def local_binary_pattern(self, image, myImage_object):
        if isinstance(image,Image.Image):
            image = myImage_object.rgb2gray_anImage(image)
            image = myImage_object.pilImage2numpyArray(image)
        else:
            image = myImage_object.numpyArray2pilImage(image)
            image = myImage_object.rgb2gray_anImage(image)
            image = myImage_object.pilImage2numpyArray(image)
        number_of_rows = image.shape[0]
        number_of_columns = image.shape[1]
        feature_map = np.zeros((number_of_rows-2, number_of_columns-2))  # excluding boundary pixels
        for row in range(0+1, number_of_rows-1):
            for column in range(0+1, number_of_columns-1):
                pattern = np.zeros(8)
                g_center = image[row, column]
                g_neighbor = image[row-1, column-1]; pattern[0] = self.s(g_neighbor=g_neighbor, g_center=g_center)
                g_neighbor = image[row-1, column  ]; pattern[1] = self.s(g_neighbor=g_neighbor, g_center=g_center)
                g_neighbor = image[row-1, column+1]; pattern[2] = self.s(g_neighbor=g_neighbor, g_center=g_center)
                g_neighbor = image[row  , column-1]; pattern[3] = self.s(g_neighbor=g_neighbor, g_center=g_center)
                g_neighbor = image[row  , column+1]; pattern[4] = self.s(g_neighbor=g_neighbor, g_center=g_center)
                g_neighbor = image[row+1, column-1]; pattern[5] = self.s(g_neighbor=g_neighbor, g_center=g_center)
                g_neighbor = image[row+1, column  ]; pattern[6] = self.s(g_neighbor=g_neighbor, g_center=g_center)
                g_neighbor = image[row+1, column+1]; pattern[7] = self.s(g_neighbor=g_neighbor, g_center=g_center)
                # calculating U:
                U = 0
                for neighbor_index in range(0+1, 8):
                    U += abs(pattern[neighbor_index] - pattern[neighbor_index-1])
                U += abs(pattern[7] - pattern[0])
                # calculating feature of the pixel:
                if U <= 2:
                    feature_map[row-1, column-1] = sum(pattern)  #--> -1: because row adn column are starting from 1 (excluding boundary of image)
                else:
                    feature_map[row-1, column-1] = 8+1
        feature_vector = feature_map.ravel()
        return feature_vector

    def s(self, g_neighbor, g_center):
        # --> s(x) function in LBP:
        # print(g_neighbor); print(g_center); print('********')
        if int(g_neighbor) - int(g_center) >= 0:
            return 1
        else:
            return 0