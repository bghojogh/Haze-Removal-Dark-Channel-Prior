from MyImage_class import *
from PIL import Image


class RainSnowRemoval:

    def __init__(self):
        pass

    def remove_RainSnow(self, image, myImage_object, vertical_edge_map):
        if isinstance(image, Image.Image):
            image = MyImage.pilImage2numpyArray(myImage_object, image)
        image_hsv = myImage_object.rgb2hsv_anImage(image)   # convert image to hsv
        number_of_rows = image_hsv.shape[0]
        number_of_columns = image_hsv.shape[1]
        recovered_image_hsv = np.empty_like(image_hsv); recovered_image_hsv[:] = image_hsv   # copy array by value (not by reference): https://stackoverflow.com/questions/6431973/how-to-copy-data-from-a-numpy-array-to-another
        print(vertical_edge_map.shape)
        vertical_edge_map_backup = np.empty_like(vertical_edge_map); vertical_edge_map_backup[:] = vertical_edge_map
        for channel in range(1,3):  # iteration on channels: S, V
            vertical_edge_map = np.empty_like(vertical_edge_map_backup); vertical_edge_map[:] = vertical_edge_map_backup
            for row in range(number_of_rows):
                for column in range(number_of_columns):
                    if int(vertical_edge_map[row,column]) is 1:  # it is a vertical edge (with a good chance, it is rain/snow streak)
                        streak_pixels_coordinates = np.array([[row,column]])
                        radius = 0
                        still_streak_exists = 1
                        # find the pixels of streak (search for the streak around edge pixel):
                        while still_streak_exists is 1:
                            radius += 1
                            still_streak_exists = 0
                            # iterate on square around the pixel, with the radius:
                            for i in range(row-radius,row+radius+1):
                                if (i is row-radius) or (i is row+radius):
                                    neighbors_around_pixel_with_radius = range(column-radius,column+radius+1)
                                else:
                                    neighbors_around_pixel_with_radius = [column-radius, column+radius]
                                for j in neighbors_around_pixel_with_radius:
                                    # check for pixels in streak:
                                    if (i >= 0) and (i < number_of_rows) and (j >= 0) and (j < number_of_columns):  # if the pixel is in the range of image
                                        if int(vertical_edge_map[i,j]) is 1:  # if it is a vertical edge pixel
                                            # check if it is neighbor of an existing pixel of streak (if it is connected to the streak or is in another streak):
                                            is_neighbor_of_an_existing_pixel_of_streak = self.is_the_pixel_neighbor_of_streak_pixel(myImage_object=myImage_object, streak_pixels_coordinates=streak_pixels_coordinates, pixel_coordinate=[i,j], number_of_rows=number_of_rows, number_of_columns=number_of_columns)
                                            if is_neighbor_of_an_existing_pixel_of_streak is True:
                                                still_streak_exists = 1
                                                streak_pixels_coordinates = np.vstack([streak_pixels_coordinates, [i,j]])  # add the new streak pixels as a row of streak_pixels_coordinates
                        # find the rectangular region covering streak:
                        up_of_region_of_streak = max((streak_pixels_coordinates[:,0].min()) - 1, 0)  # min of rows of streak pixels - 1 pixel
                        down_of_region_of_streak = min((streak_pixels_coordinates[:,0].max()) + 1, number_of_rows-1)  # max of rows of streak pixels + 1 pixel
                        left_of_region_of_streak = max((streak_pixels_coordinates[:,1].min()) - 1, 0)  # min of columns of streak pixels - 1 pixel
                        right_of_region_of_streak = min((streak_pixels_coordinates[:,1].max()) + 1, number_of_columns-1)  # max of columns of streak pixels + 1 pixel
                        # find the surrounding background pixels (one pixel around streak pixels) among pixels of the rectangular region, and average them:
                        sum_of_surrounding_background_pixels = 0
                        counter = 0
                        for i in range(up_of_region_of_streak, down_of_region_of_streak+1):
                            for j in range(left_of_region_of_streak, right_of_region_of_streak+1):
                                if MyImage.check_if_an_element_exists_in_array(self=myImage_object, array=streak_pixels_coordinates, element=[i,j]) == None:  # if the pixel is not in streak itself (because background is not in streak)
                                    # extract the surrounding pixels out of rectangular region (check if this pixel is neighbor of streak pixel or not):
                                    this_pixel_is_surrounding_pixel = self.is_the_pixel_neighbor_of_streak_pixel(myImage_object=myImage_object, streak_pixels_coordinates=streak_pixels_coordinates, pixel_coordinate=[i,j], number_of_rows=number_of_rows, number_of_columns=number_of_columns)
                                    if this_pixel_is_surrounding_pixel is True:
                                        # the pixel is neighbor of streak (is in surrounding background)
                                        sum_of_surrounding_background_pixels += image_hsv[i,j,channel]
                                        counter += 1
                        average_of_surrounding_background = sum_of_surrounding_background_pixels/counter
                        # replace (inpaint) the streak pixels with the average of surrounding background pixels:
                        number_of_streak_pixels_in_this_streak = streak_pixels_coordinates.shape[0]
                        for streak_pixel_index in range(number_of_streak_pixels_in_this_streak):
                            row_of_streak_pixel_in_image = streak_pixels_coordinates[streak_pixel_index,0]
                            column_of_streak_pixel_in_image = streak_pixels_coordinates[streak_pixel_index,1]
                            # inpaint in image:
                            recovered_image_hsv[row_of_streak_pixel_in_image,column_of_streak_pixel_in_image,channel] = average_of_surrounding_background
                            # recovered_image_hsv[row_of_streak_pixel_in_image,column_of_streak_pixel_in_image,0]  = 0
                            # recovered_image_hsv[row_of_streak_pixel_in_image,column_of_streak_pixel_in_image,1]  = 0
                            # recovered_image_hsv[row_of_streak_pixel_in_image,column_of_streak_pixel_in_image,2]  = 0
                            # print(recovered_image_hsv[row_of_streak_pixel_in_image,column_of_streak_pixel_in_image,channel])
                            # print(image_hsv[row_of_streak_pixel_in_image,column_of_streak_pixel_in_image,channel])
                            # print('***********')
                            # remove the pixels of that streak from vertical_edge_map (map of streaks):
                            vertical_edge_map[row_of_streak_pixel_in_image, column_of_streak_pixel_in_image] = 0
        # print(recovered_image_hsv[10,10,0], recovered_image_hsv[10,10,1], recovered_image_hsv[10,10,2])
        recovered_image = myImage_object.hsv2rgb_anImage(recovered_image_hsv)   # convert recovered_image_hsv to rgb
        # print(colorsys.hsv_to_rgb(0, 0, 0))
        # print(recovered_image[10,10,0], recovered_image[10,10,1], recovered_image[10,10,2])
        return recovered_image

    def is_the_pixel_neighbor_of_streak_pixel(self, myImage_object, streak_pixels_coordinates, pixel_coordinate, number_of_rows, number_of_columns):
        i = pixel_coordinate[0]  # row of pixel
        j = pixel_coordinate[1]  # column of pixel
        is_neighbor_of_pixel_of_streak = False
        for ii in range(i-1,i+1+1):   # iteration on neighbors of pixel
            for jj in range(j-1,j+1+1):
                if (ii >= 0) and (ii < number_of_rows) and (jj >= 0) and (jj < number_of_columns):  # if the pixel is in the range of image
                    if (is_neighbor_of_pixel_of_streak is False) and (MyImage.check_if_an_element_exists_in_array(self=myImage_object, array=streak_pixels_coordinates, element=[ii,jj]) != None):
                        is_neighbor_of_pixel_of_streak = True
        return is_neighbor_of_pixel_of_streak