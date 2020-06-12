from MyImage_class import *
from PIL import Image

class HazeDetection:

    def __init__(self):
        pass

    def extract_histogram_features(self, image, myImage_object, image_index, path_histogram='./', format_of_save='jpg'):
        if isinstance(image,Image.Image):
            image_array = myImage_object.pilImage2numpyArray(img=image)
        else:
            image_array = image
        # gray-scaling image:
        if not myImage_object.is_image_gray_scale(image=image_array):
            image_array = myImage_object.rgb2gray_anImage(rgb_image=image_array)
        # find histogram:
        image_reshaped = image_array.ravel()
        fig = plt.figure()
        bins = [x for x in range(0,256)]  # bins in range: 0:1:255
        # bin_counts, bin_edges = np.histogram(image_reshaped, bins)     # https://stackoverflow.com/questions/29864330/histogram-of-gray-scale-values-in-numpy-image
        bin_counts, bin_edges, patches = plt.hist(image_reshaped, bins)  # https://stackoverflow.com/questions/9141732/how-does-numpy-histogram-work
        threshold = (max(bin_counts)) / 50
        plt.plot([0,255], [threshold,threshold])
        #plt.show()
        number_of_images_in_folder = myImage_object.count_number_of_files_in_folder(folder_path=path_histogram)
        name_of_histogram = str(number_of_images_in_folder)
        myImage_object.save_plot(figure=fig, name=name_of_histogram, save_path=path_histogram, format_of_save='jpg')
        # threshold histogram:
        feature_vector = bin_counts
        feature_vector[feature_vector <= threshold] = 0
        feature_vector[feature_vector > threshold] = 1
        return feature_vector

    def divide_image_into_boxes(self, image, myImage_object, number_of_boxes_in_each_dimension=5, save_path='./', format_of_save='jpg'):
        if isinstance(image,Image.Image):
            image_array = myImage_object.pilImage2numpyArray(img=image)
        else:
            image_array = image
        # gray-scaling image:
        if not myImage_object.is_image_gray_scale(image=image_array):
            image_array = myImage_object.rgb2gray_anImage(rgb_image=image_array)
        number_of_rows = image_array.shape[0]
        number_of_columns = image_array.shape[1]
        vertical_size_of_box = math.floor(number_of_rows/number_of_boxes_in_each_dimension)
        horizontal_size_of_box = math.floor(number_of_columns/number_of_boxes_in_each_dimension)
        for row in range(0, vertical_size_of_box*number_of_boxes_in_each_dimension, vertical_size_of_box):   # ignore the last box if it is fractional
            for column in range(0, horizontal_size_of_box*number_of_boxes_in_each_dimension, horizontal_size_of_box):   # ignore the last box if it is fractional
                box = image_array[row:row+vertical_size_of_box, column:column+horizontal_size_of_box]
                number_of_saved_boxes_in_folder = myImage_object.count_number_of_files_in_folder(folder_path=save_path)
                name_of_box = str(number_of_saved_boxes_in_folder)
                myImage_object.save_image(image=box, name=name_of_box, save_path=save_path, format_of_save=format_of_save)