import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm
import numpy as np
from PIL import Image
import colorsys
from scipy import ndimage as ndi
from skimage import feature
import math
import os, os.path
import os, shutil
from scipy import signal
import re
import pickle
import shutil


class MyImage:
    image_list = []
    images_address = []

    def __init__(self, _path=None, _image_type=None):
        self.imagesPath = _path
        self.imagesType = 'jpg' if _image_type is None else _image_type

    def read_images(self, folder_path=None):
        self.image_list = []
        if folder_path is None:
            folder_path = self.imagesPath
        self.images_address = folder_path + '*.' + self.imagesType
        for filename in self.natsort(list_=glob.glob(self.images_address)):
            im = Image.open(filename)    # similar to: im = plt.imread(filename)
            #arr = np.array(im)
            self.image_list.append(im)
        return self.image_list

    def natsort(self, list_):
        """ for sorting names of files in human-sense """
        # http://code.activestate.com/recipes/285264-natural-string-sorting/  ---> comment of r8qyfhp02
        # decorate
        tmp = [ (int(re.search('\d+', i).group(0)), i) for i in list_ ]
        tmp.sort()
        # undecorate
        return [ i[1] for i in tmp ]

    def numbering_images(self, folder_path=None, format_of_save='jpg', folder_path_save='./'):
        """ Numbering images cleanly starting from 0 """
        self.image_list = []
        if folder_path is None:
            folder_path = self.imagesPath
        self.images_address = folder_path + '*.' + self.imagesType
        counter = 0
        for filename in self.natsort(list_=glob.glob(self.images_address)):
            im = Image.open(filename)    # similar to: im = plt.imread(filename)
            self.save_image(image=im, name=str(counter), save_path=folder_path_save, format_of_save=format_of_save)
            counter += 1

    def show_pilImage_fromFolder(self, index_of_showing_image, _image_list=None):
        __image_list = self.image_list if _image_list is None else _image_list
        try:
            im = __image_list[index_of_showing_image]
            self.show_pilImage_fromInput(im)
        except Exception as e:
            print(e)
            print('The index of image is beyond the range of number of images!')

    def show_pilImage_fromInput(self, image):
        if self.is_image_gray_scale(image):
            plt.imshow(image, cmap=matplotlib.cm.Greys_r)
            # https://stackoverflow.com/questions/30419722/in-pil-why-isnt-convertl-turning-image-grayscale
        else:
            plt.imshow(image)
        plt.show()

    def is_image_gray_scale(self, image):
        image_as_numpy_array = np.asarray(image)   # convert from PIL image to numpy array
        if len(image_as_numpy_array.shape) == 2:  # if it is gray-scale (if it is 2D and not 3D)
            return True
        else:                  # if it is RGB (if it is 3D)
            return False

    def show_all_images(self, pause_in_second=0.1):
        for filename in glob.glob(self.images_address):
            im = plt.imread(filename)
            if self.is_image_gray_scale(im):
                plt.imshow(im, cmap=matplotlib.cm.Greys_r)
                # https://stackoverflow.com/questions/30419722/in-pil-why-isnt-convertl-turning-image-grayscale
            else:
                plt.imshow(im)
            plt.draw()     # use plt.draw() instead of plt.show() in order not to stop for closing figure
            plt.pause(pause_in_second)
            plt.close()

    def rgb2gray_anImage(self, rgb_image):
        if isinstance(rgb_image,Image.Image):
            gray_image = rgb_image.convert('L')   # should use rgb_image.convert('L') instead of rgb_image.convert('LA') for jpg images.
            return gray_image
        else:
            rgb_image = self.numpyArray2pilImage(array=rgb_image)
            gray_image = rgb_image.convert('L')   # should use rgb_image.convert('L') instead of rgb_image.convert('LA') for jpg images.
            gray_image = self.pilImage2numpyArray(img=gray_image)
            return gray_image

    def rgb2gray_allImages(self, save_path='./', format_of_save='jpg'):
        counter = 0
        for filename in glob.glob(self.images_address):
            im = Image.open(filename)
            gray_image = self.rgb2gray_anImage(im)
            counter += 1
            self.save_image(image=gray_image, name=str(counter), save_path=save_path, format_of_save='jpg')

    def save_image(self, image, name='temp', save_path='./', format_of_save='jpg'):
        if not os.path.exists(save_path):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(save_path)
        is_image_in_pil_format = isinstance(image,Image.Image)
        if is_image_in_pil_format is False:  # if is numpy array
            pil_Image = self.numpyArray2pilImage(array=image)
        else:
            pil_Image = image
        try:
            pil_Image.save(save_path + name + '.' + format_of_save)
        except Exception as e:
            print(e)

    def save_plot(self, figure, name='temp', save_path='./', format_of_save='jpg'):
        # do these before calling this function:
        # 1- fig = plt.figure()
        # 2- plt.plot(...) or plt.hist(...) or plt.scatter(...) or ...
        # 3- pass fig as figure to function
        if not os.path.exists(save_path):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(save_path)
        figure.savefig(save_path + name + '.' + format_of_save, dpi=300)  # if don't want borders: bbox_inches='tight'
        plt.close(figure)

    def pilImage2numpyArray(self, img):
        # https://stackoverflow.com/questions/384759/pil-and-numpy
        if isinstance(img,Image.Image):
            img_arr = np.array(img)
            return img_arr
        else:
            return None

    def numpyArray2pilImage(self, array):
        # https://stackoverflow.com/questions/384759/pil-and-numpy
        img_arr = Image.fromarray(np.uint8(array))
        return img_arr

    def rgb2hsv_anImage(self, img):
        # see also: https://stackoverflow.com/questions/22236956/rgb-to-hsv-via-pil-and-colorsys
        if isinstance(img,Image.Image):
            img_arr = self.pilImage2numpyArray(img)
        else:
            img_arr = img
        number_of_rows = img_arr.shape[0]
        number_of_columns = img_arr.shape[1]
        #r,g,b = img.split()
        r = img_arr[:,:,0]
        g = img_arr[:,:,1]
        b = img_arr[:,:,2]
        image_hsv = np.zeros((number_of_rows,number_of_columns,3))
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                h, s, v = self.__rgb2hsv_aPixel(r[row,column],g[row,column],b[row,column])
                image_hsv[row,column,0] = h
                image_hsv[row,column,1] = s
                image_hsv[row,column,2] = v
        return image_hsv

    def rgb2hls_anImage(self, img):
        if isinstance(img,Image.Image):
            img_arr = self.pilImage2numpyArray(img)
        else:
            img_arr = img
        number_of_rows = img_arr.shape[0]
        number_of_columns = img_arr.shape[1]
        #r,g,b = img.split()
        r = img_arr[:,:,0]
        g = img_arr[:,:,1]
        b = img_arr[:,:,2]
        image_hls = np.zeros((number_of_rows,number_of_columns,3))
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                h, l, s = self.__rgb2hls_aPixel(r[row,column],g[row,column],b[row,column])
                image_hls[row,column,0] = h
                image_hls[row,column,1] = l
                image_hls[row,column,2] = s
        return image_hls

    def hsv2rgb_anImage(self, img):
        if isinstance(img,Image.Image):
            img = self.pilImage2numpyArray(img)
        else:
            img = img
        number_of_rows = img.shape[0]
        number_of_columns = img.shape[1]
        h = img[:,:,0]
        s = img[:,:,1]
        v = img[:,:,2]
        image_rgb = np.zeros((number_of_rows,number_of_columns,3))
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                r, g, b = self.__hsv2rgb_aPixel(h[row,column],s[row,column],v[row,column])
                image_rgb[row,column,0] = r
                image_rgb[row,column,1] = g
                image_rgb[row,column,2] = b
        return image_rgb

    def hls2rgb_anImage(self, img):
        if isinstance(img,Image.Image):
            img = self.pilImage2numpyArray(img)
        else:
            img = img
        number_of_rows = img.shape[0]
        number_of_columns = img.shape[1]
        h = img[:,:,0]
        l = img[:,:,1]
        s = img[:,:,2]
        image_rgb = np.zeros((number_of_rows,number_of_columns,3))
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                r, g, b = self.__hls2rgb_aPixel(h[row,column],l[row,column],s[row,column])
                image_rgb[row,column,0] = r
                image_rgb[row,column,1] = g
                image_rgb[row,column,2] = b
        return image_rgb

    # private method:
    def __rgb2hsv_aPixel(self, r, g, b):
        # https://docs.python.org/2/library/colorsys.html
        r, g, b = r/255.0, g/255.0, b/255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h, s, v

    # private method:
    def __rgb2hls_aPixel(self, r, g, b):
        # https://docs.python.org/2/library/colorsys.html
        r, g, b = r/255.0, g/255.0, b/255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return h, l, s

    # private method:
    def __hsv2rgb_aPixel(self, h, s, v):
        # https://docs.python.org/2/library/colorsys.html
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        r *= 255; g *= 255; b *= 255
        return r, g, b

    # private method:
    def __hls2rgb_aPixel(self, h, l, s):
        # https://docs.python.org/2/library/colorsys.html
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        r *= 255; g *= 255; b *= 255
        return r, g, b

    def cannyEdgeDetection(self, image_numpy_array, sigma=3):
        # very good web: http://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html
        im = ndi.gaussian_filter(image_numpy_array, 4)
        edges = feature.canny(im, sigma=sigma)
        return edges

    def sobelEdgeDetection_manual(self, image, do_threshold=True, threshold=None):
        if isinstance(image,Image.Image):
            image = self.rgb2gray_anImage(image)
            img_arr = self.pilImage2numpyArray(image)
        else:
            image_pil = self.numpyArray2pilImage(image)
            image = self.rgb2gray_anImage(image_pil)
            img_arr = self.pilImage2numpyArray(image)
        G_x = [[1,0,-1],[2,0,-2],[1,0,-1]]
        G_x_opposite = [[-1,0,1],[-2,0,2],[-1,0,1]]
        G_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
        G_y_opposite = [[-1,-2,-1],[0,0,0],[1,2,1]]
        G_x_applied = np.zeros(img_arr.shape)
        G_x_opposite_applied = np.zeros(img_arr.shape)
        G_y_applied = np.zeros(img_arr.shape)
        G_y_opposite_applied = np.zeros(img_arr.shape)
        G_total = np.zeros(img_arr.shape)
        phase_sobel = np.zeros(img_arr.shape)
        number_of_rows = img_arr.shape[0]
        number_of_columns = img_arr.shape[1]
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                if row != 0 and row != number_of_rows-1 and column != 0 and column != number_of_columns-1:
                    patch_of_image = [[img_arr[row-1,column-1],img_arr[row,column-1],img_arr[row+1,column-1]],
                                      [img_arr[row-1,column],img_arr[row,column],img_arr[row+1,column]],
                                      [img_arr[row-1,column+1],img_arr[row,column+1],img_arr[row+1,column+1]]]
                    G_x_applied[row,column] = np.sum(np.multiply(G_x, patch_of_image))
                    G_x_opposite_applied[row,column] = np.sum(np.multiply(G_x_opposite, patch_of_image))
                    G_y_applied[row,column] = np.sum(np.multiply(G_y, patch_of_image))
                    G_y_opposite_applied[row,column] = np.sum(np.multiply(G_y_opposite, patch_of_image))
                G_total[row,column] = (G_x_applied[row,column]**2 + G_y_applied[row,column]**2)**(1/2)
                phase_sobel[row,column] = math.atan2(G_y_applied[row,column], G_x_applied[row,column]) * (180/math.pi)
        if do_threshold:
            G_x_applied[G_x_applied < threshold] = 0; G_x_applied[G_x_applied >= threshold] = 1
            G_x_opposite_applied[G_x_opposite_applied < threshold] = 0; G_x_opposite_applied[G_x_opposite_applied >= threshold] = 1
            G_y_applied[G_y_applied < threshold] = 0; G_y_applied[G_y_applied >= threshold] = 1
            G_y_opposite_applied[G_y_opposite_applied < threshold] = 0; G_y_opposite_applied[G_y_opposite_applied >= threshold] = 1
            G_total[G_total < threshold] = 0; G_total[G_total >= threshold] = 1
        return G_total, G_x_applied, G_y_applied, phase_sobel, G_x_opposite_applied, G_y_opposite_applied

    def count_number_of_files_in_folder(self, folder_path='./'):
        if not os.path.exists(folder_path):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            return 0
        # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
        number_of_files_in_folder = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        return number_of_files_in_folder

    def delete_files_in_folder(self, folder_path, do_delete_sub_directories_too=False):
        # https://stackoverflow.com/questions/185936/delete-folder-contents-in-python
        for the_file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, the_file)
            try:
                if do_delete_sub_directories_too is False:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                else:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def check_if_an_element_exists_in_array(self, array, element):
        # https://stackoverflow.com/questions/7571635/fastest-way-to-check-if-a-value-exist-in-a-list
        try:
            index = array.tolist().index(element)
        except ValueError:
            index = None   # does not exist in array
        return index

    def apply_median_filter(self, image, kernel_size):
        if isinstance(image,Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        return signal.medfilt(volume=image_array, kernel_size=kernel_size)

    def map_2d_array_to_image_range(self, array):
        number_of_rows = array.shape[0]
        number_of_columns = array.shape[1]
        mapped_image = np.zeros(array.shape)
        if len(array.shape) == 2:  # if it is gray-scale (if it is 2D and not 3D)
            mapped_image = self.translate(array=array, leftMin=min(array.flatten()), leftMax=max(array.flatten()), rightMin=0, rightMax=255)
        else:                  # if it is RGB (if it is 3D)
            for channel in range(3):
                mapped_image[:,:,channel] = self.translate(array=array[:,:,channel], leftMin=min(array[:,:,channel].flatten()), leftMax=max(array[:,:,channel].flatten()), rightMin=0, rightMax=255)
        return mapped_image

    def translate(self, array, leftMin, leftMax, rightMin, rightMax):
        # https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        # Convert the left range into a 0-1 range (float)
        arrayScaled = (array - leftMin) / (leftSpan)
        # Convert the 0-1 range into a value in the right range.
        return np.floor(rightMin + (arrayScaled * rightSpan))

    def save_file(self, file, filename, path_save):
        if not os.path.exists(path_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_save)
        # https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
        with open(path_save + filename + '.pkl', 'wb') as output:
            pickle.dump(file, output, pickle.HIGHEST_PROTOCOL)

    def load_file(self, filename, path_folder):
        # https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
        with open(path_folder + filename + '.pkl', 'rb') as input:
            file = pickle.load(input)
        return file

    def remove_folder(self, folder_path):
        """ removes a folder """
        # https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty-with-python
        shutil.rmtree(folder_path, ignore_errors=True)

    def store_sth_in_text_file(self, sth_in_a_line='something...', path_of_test_file='./text.txt', overwrite_file_if_already_exists=False):
        # https://stackoverflow.com/questions/4706499/how-do-you-append-to-a-file
        # https://stackoverflow.com/questions/21839803/how-to-append-new-data-onto-a-new-line
        # https://stackoverflow.com/questions/82831/how-do-i-check-whether-a-file-exists-using-python
        # https://stackoverflow.com/questions/35807605/create-a-file-if-it-doesnt-exist
        # https://www.tutorialspoint.com/python/file_close.htm
        if not os.path.isfile(path_of_test_file): #--> if the file does not exist already
            file = open(path_of_test_file, 'w')
        else:
            if overwrite_file_if_already_exists:
                file = open(path_of_test_file, 'w')
            else:
                file = open(path_of_test_file, 'a')
        file.write(sth_in_a_line)
        file.write('\n')
        file.close()