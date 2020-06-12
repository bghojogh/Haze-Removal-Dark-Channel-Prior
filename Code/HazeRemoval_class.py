from MyImage_class import *
from PIL import Image

class HazeRemoval:

    def __init__(self):
        pass

    def find_dark_channel(self, image, myImage_object, patch_size=15):
        if isinstance(image,Image.Image):
            image_array = MyImage.pilImage2numpyArray(myImage_object,image)
        else:
            image_array = image
        number_of_rows = image_array.shape[0]
        number_of_columns = image_array.shape[1]
        r = image_array[:,:,0]
        g = image_array[:,:,1]
        b = image_array[:,:,2]
        dark_channel = np.zeros([number_of_rows, number_of_columns])
        # ----- find min of channels:
        min_of_channels = np.zeros([number_of_rows, number_of_columns])
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                min_of_channels[row,column] = min(r[row,column], g[row,column], b[row,column])
        # ----- find min in patches:
        t = int((patch_size-1)/2)
        for row in range(0, number_of_rows):
            for column in range(0, number_of_columns):
                minimum = 10**4  # infinite number
                # iteration on neighbors of pixel in patch:
                for i in range(row-t, row+t+1):
                    for j in range(column-t, column+t+1):
                        if (i >= 0) and (i < number_of_rows) and (j >= 0) and (j < number_of_columns):  # if the pixel is in the range of image
                            if min_of_channels[i,j] < minimum:
                                minimum = min_of_channels[i,j]
                dark_channel[row,column] = minimum
        return dark_channel

    def find_atmospheric_light(self, image, myImage_object, dark_channel, threshold=0.1/100):
        if isinstance(image,Image.Image):
            image_array = MyImage.pilImage2numpyArray(myImage_object,image)
        else:
            image_array = image
        number_of_rows = image_array.shape[0]
        number_of_columns = image_array.shape[1]
        # ------ find brightest in dark channel:
        dark_channel_reshaped = dark_channel.ravel()
        dark_channel_reshaped.sort() # sort from smallest to largest
        dark_channel_reshaped = dark_channel_reshaped[::-1]  # sort from largest to smallest
        brightest_in_dark_channel = dark_channel_reshaped[0]
        # ------ pick the 'threshold' number of brightes pixels:
        n = int(threshold * (number_of_rows * number_of_columns))
        indices_of_top_brightest_pixels_in_dark_channel = (-dark_channel_reshaped).argsort()[:n]  # https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
        counter = 0
        bright_pixels_in_dark_channel = np.zeros([indices_of_top_brightest_pixels_in_dark_channel.shape[0],2])
        for i in indices_of_top_brightest_pixels_in_dark_channel:
            row_of_pixel = int(i / number_of_columns)
            column_of_pixel = int(i % number_of_columns)
            bright_pixels_in_dark_channel[counter,0] = row_of_pixel
            bright_pixels_in_dark_channel[counter,1] = column_of_pixel
            counter += 1
        # ------ find the highest intensities of bright_pixels_in_dark_channel in the input image:
        atmospheric_light = np.zeros(3)  # has 3 channels
        max_in_channel_red = 0; max_in_channel_green = 0; max_in_channel_blue = 0
        for pixel in range(0, bright_pixels_in_dark_channel.shape[0]):
            row_of_pixel = int(bright_pixels_in_dark_channel[pixel,0])
            column_of_pixel = int(bright_pixels_in_dark_channel[pixel,1])
            # channel red:
            if image_array[row_of_pixel, column_of_pixel, 0] > max_in_channel_red:
                max_in_channel_red = image_array[row_of_pixel, column_of_pixel, 0]
                atmospheric_light[0] = max_in_channel_red
            # channel green:
            if image_array[row_of_pixel, column_of_pixel, 1] > max_in_channel_green:
                max_in_channel_green = image_array[row_of_pixel, column_of_pixel, 1]
                atmospheric_light[1] = max_in_channel_green
            # channel blue:
            if image_array[row_of_pixel, column_of_pixel, 2] > max_in_channel_blue:
                max_in_channel_blue = image_array[row_of_pixel, column_of_pixel, 2]
                atmospheric_light[2] = max_in_channel_blue
        return atmospheric_light

    def find_transmission(self, image, atmospheric_light, myImage_object, weight=0.95, patch_size=15):
        if isinstance(image,Image.Image):
            image_array = MyImage.pilImage2numpyArray(myImage_object,image)
        else:
            image_array = image
        # --- normalizing input image with atmospheric_light in each channel (r, g, and b):
        image_array_normalized = np.zeros(image_array.shape)
        image_array_normalized[:,:,0] = image_array[:,:,0] / atmospheric_light[0]
        image_array_normalized[:,:,1] = image_array[:,:,1] / atmospheric_light[1]
        image_array_normalized[:,:,2] = image_array[:,:,2] / atmospheric_light[2]
        dark_channel_of_normalized_hazy_image = self.find_dark_channel(image=image_array_normalized, myImage_object=myImage_object, patch_size=patch_size)
        # --- find the transmission map:
        transmission_map = 1 - (weight * dark_channel_of_normalized_hazy_image)
        return transmission_map

    def remove_haze(self, image, atmospheric_light, transmission_map, myImage_object, t_0=0.1):
        if isinstance(image,Image.Image):
            image_array = MyImage.pilImage2numpyArray(myImage_object,image)
        else:
            image_array = image
        number_of_rows = image_array.shape[0]
        number_of_columns = image_array.shape[1]
        recovered_image = np.zeros(image_array.shape)
        for channel in range(3):
            for row in range(0, number_of_rows):
                for column in range(0, number_of_columns):
                    recovered_image[row, column, channel] = ((image_array[row,column,channel] - atmospheric_light[channel]) / max(transmission_map[row,column],t_0)) + atmospheric_light[channel]
        return recovered_image
