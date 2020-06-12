from MyImage_class import *
from RainSnowDetector_class import *
from RainSnowRemoval_class import *
from HazeRemoval_class import *
from HazeDetection_class import *
from LDA_class import *
import time
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


def main():
    # ------ Settings:
    path_dataset = './input_images/'
    imagesType = 'jpg'
    # ------ iteration on all images of dataset:
    images_address = path_dataset + '*.' + imagesType
    list_of_image_addresses = glob.glob(images_address)
    number_of_images_in_dataset = len(list_of_image_addresses)
    for image_index in range(number_of_images_in_dataset):
        # ------ https://stackoverflow.com/questions/678236/how-to-get-the-filename-without-the-extension-from-a-path-in-python
        address_of_image = list_of_image_addresses[image_index]
        image_name_with_extension = os.path.basename(address_of_image)
        image_name_pure = os.path.splitext(image_name_with_extension)[0]
        print('=====> Processing image #' + str(image_index+1) + ' (' + image_name_with_extension + ')' + ' out of ' + str(number_of_images_in_dataset) + ' images...')
        # ------ read image:
        image = Image.open(address_of_image)
        # ------ calling weather module:
        weather_module(image=image, image_name=image_name_pure, path_saving_enhanced_dataset='./exnhanced_dataset/', format_of_save=imagesType, path_saving_condition='./condition_of_images/')

# ------ Weather Module:
def weather_module(image, image_name, path_saving_enhanced_dataset='./exnhanced_dataset/', format_of_save='jpg', path_saving_condition='./condition_of_images/'):
    # ------ instantiate MyImage class:
    myImage = MyImage(_image_type='jpg', _path='./')
    # ------ paths of savings:
    path_intermediate_files = './intermediate_saved_files/'
    path_haze_detection = path_intermediate_files + 'haze_detection/'
    path_rain_detection = path_intermediate_files + 'rain_detection/'
    path_haze_removal = path_intermediate_files + 'haze_removal/'
    path_save_condition = path_saving_condition + 'conditions_of_images.txt'
    # ------ load trained files:
    # -> haze detection files:
    e_vecs = myImage.load_file(filename='e_vecs', path_folder=path_haze_detection+'saved_train_files/')
    projected_training_data = myImage.load_file(filename='projected_training_data', path_folder=path_haze_detection+'saved_train_files/')
    # -> rain detection files:
    lda = myImage.load_file(filename='lda', path_folder=path_rain_detection+'saved_train_files/')
    pca = myImage.load_file(filename='pca', path_folder=path_rain_detection+'saved_train_files/')
    # ------ detect luminance:
    print('luminance amount estimation...')
    light_factor = detect_light_condition(image=image, myImage_object=myImage)
    # ------ detect haze:
    print('haze amount estimation...')
    haze_factor, estimated_classes_of_boxes_HAZE = Detect_Haze(image=image, myImage_object=myImage, projected_training_data=projected_training_data, e_vecs=e_vecs, path_haze_detection=path_haze_detection)
    # ------ detect rain:
    print('rain amount estimation...')
    rain_factor, estimated_classes_of_boxes_RAIN = Detect_Rain(image=image, myImage_object=myImage, lda=lda, pca=pca, path_rain_detection=path_rain_detection, save_intermediate_images=False)
    # ------ luminance enhancement (if necessary):
    print('luminance enhancement (if necessary)...')
    is_luminance_enhanced, image = enhance_luminance(image=image, myImage_object=myImage)
    # ------ haze removal (if necessary):
    print('haze enhancement (if necessary)...')
    if not (light_factor == 5):   #--> if it not night
        if haze_factor == 4 or haze_factor == 5:
            image = Remove_Haze(image=image, myImage_object=myImage, patch_size=3, save_intermediate_images=False, path_haze_removal=path_haze_removal)
            is_haze_removed = True
        else:
            is_haze_removed = False
    else:
        is_haze_removed = False
    # ------ save the condition of image in text file:
    line_to_store = str(light_factor) + '\t' + str(haze_factor) + '\t' + str(rain_factor) + '\t' + str(is_luminance_enhanced) + '\t' + str(is_haze_removed)
    if not os.path.exists(path_saving_condition):  #--> if the folder does not exist, create it
        os.makedirs(path_saving_condition)
    myImage.store_sth_in_text_file(sth_in_a_line=line_to_store, path_of_test_file=path_save_condition, overwrite_file_if_already_exists=False)
    # ------ save the [recovered] image:
    print('saving results...')
    myImage.save_image(image=image, name=image_name, save_path=path_saving_enhanced_dataset, format_of_save=format_of_save)

# ------ functions used in main:
def enhance_luminance(image, myImage_object):
    image_hls = myImage_object.rgb2hls_anImage(img=image)
    luminance = image_hls[:,:,1]
    luminance_average = luminance.mean()
    if luminance_average < 0.4:
        is_luminance_enhanced = True
        difference = 0.4 - luminance_average
        luminance = luminance + difference
        image_hls[:,:,1] = luminance
    else:
        is_luminance_enhanced = False
    image = myImage_object.hls2rgb_anImage(img=image_hls)
    return is_luminance_enhanced, image

def detect_light_condition(image, myImage_object):
    image = myImage_object.rgb2gray_anImage(rgb_image=image)
    image_array = myImage_object.pilImage2numpyArray(img=image)
    # number_of_rows = image_array.shape[0]
    # number_of_columns = image_array.shape[1]
    # number_of_pixels = number_of_rows * number_of_columns
    # average_of_intensities = image_array.mean()
    # mode = stats.mode(image_array.ravel())
    # mode_of_intensities = int(mode[0])
    # frequency_of_mode = int(mode[1])
    # if frequency_of_mode >= number_of_pixels/2:
    #     aa = (mode_of_intensities / 255) * 5
    # else:
    #     aa = (average_of_intensities / 255) * 5
    # print(aa)
    # # aa = (average_of_intensities * 0.6) + (mode_of_intensities * 0.4)
    # # print(aa * 5)
    # light_factor = aa  #--> range: [0, 5]
    # light_factor = int(np.ceil(light_factor))  #--> quantizing light_factor --> 1, 2, 3, 4, 5 --> 1: too dark, 5: too light
    im = image_array.ravel()
    number_of_pixels_in_biggest_range = -1
    range_length = int(np.floor(255/5))
    for range_index in range(5):
        number_of_pixels_in_range = sum((im >= range_index*range_length) & (im < range_index*range_length + range_length))
        if number_of_pixels_in_range > number_of_pixels_in_biggest_range:
            number_of_pixels_in_biggest_range = number_of_pixels_in_range
            biggest_range_index = range_index
    light_factor = biggest_range_index + 1  #--> 1, 2, 3, 4, or 5 --> 1: too dark, 5: too light
    # reversing the light_factor: #--> 1, 2, 3, 4, or 5 --> 1: too light, 5: too dark
    if light_factor == 1:
        light_factor = 5
    elif light_factor == 2:
        light_factor = 4
    elif light_factor == 3:
        light_factor = 3
    elif light_factor == 4:
        light_factor = 2
    elif light_factor == 5:
        light_factor = 1
    return light_factor

def train_haze_condition(path_dataset_train_hazy, path_dataset_train_not_hazy, path_dataset_train, myImage_object):
    path_dataset_train_backup = path_dataset_train
    number_of_classes = 2  #---> 2 classes: hazy and not hazy
    training_data = [None] * number_of_classes
    for class_index in range(number_of_classes):
        if class_index == 0:
            print('----- Class ' + str(class_index) + ' (hazy)...')
            path_dataset_train = path_dataset_train_hazy
        else:
            print('----- Class ' + str(class_index) + ' (not hazy)...')
            path_dataset_train = path_dataset_train_not_hazy
        # ------ Cleanly number images:
        clean_numbered_path = path_dataset_train + 'clean_numbered/'
        myImage_object.numbering_images(folder_path=path_dataset_train, format_of_save='jpg', folder_path_save=clean_numbered_path)
        path_dataset_train = clean_numbered_path
        # ------ reading images:
        number_of_train_images = myImage_object.count_number_of_files_in_folder(folder_path=path_dataset_train)
        image_list = myImage_object.read_images(folder_path=path_dataset_train)
        # ------ instantiate class:
        hazeDetection = HazeDetection()
        # ------ divide images into boxes:
        print('----- Dividing images into boxes...')
        path_boxes = path_dataset_train + 'boxes/'
        myImage_object.remove_folder(folder_path=path_boxes)  # removing previous folder
        for image_index in range(number_of_train_images):
            image = image_list[image_index]
            hazeDetection.divide_image_into_boxes(image=image, myImage_object=myImage_object, number_of_boxes_in_each_dimension=5, save_path=path_boxes, format_of_save='jpg')
        # ------ training images:
        print('----- Extracting features...')
        image_list = myImage_object.read_images(folder_path=path_boxes)
        number_of_train_images = myImage_object.count_number_of_files_in_folder(folder_path=path_boxes)
        path_histogram = path_dataset_train + 'histograms/'
        myImage_object.remove_folder(folder_path=path_histogram)  # removing previous folder
        training_data_class = np.empty((0, 255))  #--> 255 because there are 255 bins in histogram
        for image_index in range(number_of_train_images):
            image = image_list[image_index]
            feature_vector = hazeDetection.extract_histogram_features(image=image, myImage_object=myImage_object, image_index=image_index, path_histogram=path_histogram, format_of_save='jpg')
            training_data_class = np.vstack([training_data_class, feature_vector])
        training_data[class_index] = training_data_class
    # ------ train:
    print('----- Train...')
    classifier_LDA = Classifier_LDA()
    e_vecs, e_vals, within_scatter, between_scatter, projected_training_data = classifier_LDA.Fisher_LDA_train(training_data=training_data)
    # ------ saving training data:
    print('----- Saving training data...')
    path_dataset_train = path_dataset_train_backup
    myImage_object.save_file(file=training_data, filename='training_data', path_save=path_dataset_train+'saved_train_files/')
    myImage_object.save_file(file=e_vecs, filename='e_vecs', path_save=path_dataset_train+'saved_train_files/')
    myImage_object.save_file(file=projected_training_data, filename='projected_training_data', path_save=path_dataset_train+'saved_train_files/')
    # ------ returns:
    return projected_training_data, e_vecs

def Detect_Haze(image, myImage_object, projected_training_data, e_vecs, path_haze_detection):
    # ------ instantiate class:
    hazeDetection = HazeDetection()
    # ------ divide images into boxes:
    path_boxes = path_haze_detection + 'boxes/'
    myImage_object.remove_folder(folder_path=path_boxes)  # removing previous folder
    number_of_boxes_in_each_dimension = 5
    hazeDetection.divide_image_into_boxes(image=image, myImage_object=myImage_object, number_of_boxes_in_each_dimension=number_of_boxes_in_each_dimension, save_path=path_boxes, format_of_save='jpg')
    # ------ Extracting features:
    box_list = myImage_object.read_images(folder_path=path_boxes)
    number_of_boxes = myImage_object.count_number_of_files_in_folder(folder_path=path_boxes)
    path_histogram = path_haze_detection + 'histograms/'
    myImage_object.remove_folder(folder_path=path_histogram)  # removing previous folder
    test_data_boxes = np.empty((0, 255))  #--> 255 because there are 255 bins in histogram
    for image_index in range(number_of_boxes):
        box = box_list[image_index]
        feature_vector = hazeDetection.extract_histogram_features(image=box, myImage_object=myImage_object, image_index=image_index, path_histogram=path_histogram, format_of_save='jpg')
        test_data_boxes = np.vstack([test_data_boxes, feature_vector])
    # ------ test (estimation of boxes):
    classifier_LDA = Classifier_LDA()
    estimated_classes_of_boxes = classifier_LDA.Fisher_LDA_test(test_data=test_data_boxes, projected_training_data=projected_training_data, e_vecs=e_vecs)
    # ------ test (voting):
    number_of_boxes_per_test_image = number_of_boxes_in_each_dimension * number_of_boxes_in_each_dimension
    number_of_votes_for_not_haze = int(sum(estimated_classes_of_boxes))  # class 1: not haze
    number_of_votes_for_haze = number_of_boxes_per_test_image - number_of_votes_for_not_haze   # class 0: haze
    haze_factor = (number_of_votes_for_haze / number_of_boxes_per_test_image) * 5 #--> range: [0, 5]
    haze_factor = int(np.ceil(haze_factor))  #--> quantizing haze_factor --> 1, 2, 3, 4, 5 --> 1: normal, 5: too hazy
    if haze_factor == 0:
        haze_factor = 1
    # ------ returns:
    return haze_factor, estimated_classes_of_boxes

def Remove_Haze(image, myImage_object, patch_size=15, save_intermediate_images=False, path_haze_removal='./'):
    # ------ instantiate class:
    hazeRemoval = HazeRemoval()
    # ------ extract dark channel:
    dark_channel = hazeRemoval.find_dark_channel(image=image, myImage_object=myImage_object, patch_size=patch_size)
    # ------ save results:
    if save_intermediate_images:
        path_dark_channel = path_haze_removal + 'dark_channel/'
        myImage_object.save_image(image=dark_channel, name='DarkChannel', save_path=path_dark_channel, format_of_save='jpg')
    # ------ find atmospheric light:
    atmospheric_light = hazeRemoval.find_atmospheric_light(image=image, myImage_object=myImage_object, dark_channel=dark_channel, threshold=0.1/100)
    # ------ find transmission (t_tilda):
    transmission_map = hazeRemoval.find_transmission(image=image, atmospheric_light=atmospheric_light, myImage_object=myImage_object, weight=0.95, patch_size=patch_size)
    # ------ save results:
    if save_intermediate_images:
        path_transmission_map = path_haze_removal + 'transmission_map/'
        transmission_map = transmission_map * 255
        myImage_object.save_image(image=transmission_map, name='TransmissionMap', save_path=path_transmission_map, format_of_save='jpg')
    # ------ removing haze:
    recovered_image = hazeRemoval.remove_haze(image=image, atmospheric_light=atmospheric_light, transmission_map=transmission_map, myImage_object=myImage_object, t_0=0.1)
    recovered_image = myImage_object.map_2d_array_to_image_range(recovered_image)
    # ------ save results:
    return recovered_image

def Remove_RainSnow(image, image_index, myImage_object, vertical_edge_map, vertical_opposite_edge_map, save_intermediate_images=False, path_RainSnow_removed='./'):
    # ------ instantiate class:
    rainSnowRemoval = RainSnowRemoval()
    # ------ Recover image (phase 1):
    recovered_image = rainSnowRemoval.remove_RainSnow(image=image, myImage_object=myImage_object, vertical_edge_map=vertical_edge_map)
    # ------ save results:
    if save_intermediate_images:
        myImage_object.save_image(image=recovered_image, name='recovered'+str(image_index), save_path=path_RainSnow_removed, format_of_save='jpg')
    # ------ Recover image (phase 2):
    recovered_image_PIL = myImage_object.numpyArray2pilImage(array=recovered_image)
    recovered_image2 = rainSnowRemoval.remove_RainSnow(image=recovered_image_PIL, myImage_object=myImage_object, vertical_edge_map=vertical_opposite_edge_map)
    # ------ save results:
    if save_intermediate_images:
        myImage_object.save_image(image=recovered_image2, name='recovered'+str(image_index), save_path=path_RainSnow_removed, format_of_save='jpg')
    # ------ Median Filter:
    recovered_image3 = myImage_object.apply_median_filter(image=recovered_image2, kernel_size=3)
    # ------ save results:
    if save_intermediate_images:
        myImage_object.save_image(image=recovered_image3, name='recovered'+str(image_index), save_path=path_RainSnow_removed, format_of_save='jpg')

def Train_rain_condition(myImage_object, save_intermediate_images=False, path_dataset_train='./', path_dataset_train_rainy='./', path_dataset_train_not_rainy='./', find_edge_map_again=True):
    rainSnowDetector = RainSnowDetector()  # ------ instantiate class
    path_dataset_train_backup = path_dataset_train
    size_of_box = 100
    number_of_classes = 2  #---> 2 classes: rainy / not-rainy
    training_data = [None] * number_of_classes
    for class_index in range(number_of_classes):
        if class_index == 0:
            print('----- Class ' + str(class_index) + ' (rainy)...')
            path_dataset_train = path_dataset_train_rainy
        else:
            print('----- Class ' + str(class_index) + ' (not rainy)...')
            path_dataset_train = path_dataset_train_not_rainy
        # ------ Cleanly number images:
        clean_numbered_path = path_dataset_train + 'clean_numbered/'
        myImage_object.numbering_images(folder_path=path_dataset_train, format_of_save='jpg', folder_path_save=clean_numbered_path)
        path_dataset_train = clean_numbered_path
        if find_edge_map_again:
            # ------ reading images:
            number_of_train_images = myImage_object.count_number_of_files_in_folder(folder_path=path_dataset_train)
            image_list = myImage_object.read_images(folder_path=path_dataset_train)
            # ------ Finding vertical edge maps + Dividing into boxes (patches):
            print('----- Finding vertical edge maps + Dividing into boxes (patches)...')
            for image_index in range(number_of_train_images):
                image = image_list[image_index]
                # ------ Sobel edge detection (vertical edges) on image:
                edge_map, horizontal_edge_map, vertical_edge_map, phase_sobel, horizontal_opposite_edge_map, vertical_opposite_edge_map = myImage_object.sobelEdgeDetection_manual(image=image, do_threshold=True, threshold=255/6)
                if save_intermediate_images:
                    path_vertical_edge_map = path_dataset_train + 'vertical_edge_maps/'
                    vertical_edge_map_showable = vertical_edge_map * 255  #--> change range from [0,1] to [0,255]
                    myImage_object.save_image(image=vertical_edge_map_showable, name='edge_map'+str(image_index), save_path=path_vertical_edge_map, format_of_save='jpg')
                    vertical_opposite_edge_map_showable = vertical_opposite_edge_map * 255
                    vertical_opposite_edge_map_showable = myImage_object.numpyArray2pilImage(array=vertical_opposite_edge_map_showable)
                    myImage_object.save_image(image=vertical_opposite_edge_map_showable, name='edge_map'+str(image_index)+'_opposite', save_path=path_vertical_edge_map, format_of_save='jpg')
                # ----- divide image into several boxes, and then save the boxes:
                path_boxes = path_dataset_train + 'boxes/'
                if image_index is 0:
                    myImage_object.remove_folder(folder_path=path_boxes)
                rainSnowDetector.divide_image_into_boxes(vertical_edge_map=vertical_edge_map, myImage_object=myImage_object, size_of_box=size_of_box, save_path=path_boxes, format_of_save='jpg')
        # ------ extract features:
        print('----- Extracting features...')
        path_boxes = path_dataset_train + 'boxes/'
        image_list = myImage_object.read_images(folder_path=path_boxes)
        number_of_train_images = myImage_object.count_number_of_files_in_folder(folder_path=path_boxes)
        training_data_class = np.empty((0, (size_of_box-2)**2))  #--> -2: because excluding boundary pixels
        for image_index in range(number_of_train_images):
            image = image_list[image_index]
            feature_vector = rainSnowDetector.local_binary_pattern(image=image, myImage_object=myImage_object)
            training_data_class = np.vstack([training_data_class, feature_vector])
        training_data[class_index] = training_data_class
    # ------ train:
    X = training_data[0]
    X = np.vstack([X, training_data[1]])
    y_class0 = np.zeros((training_data[0].shape[0], 1))
    y_class1 = np.ones((training_data[1].shape[0], 1))
    y = np.vstack([y_class0, y_class1])
    y = y.ravel()
    print('----- PCA...')
    # http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html
    pca = PCA(n_components=80)
    pca = pca.fit(X)
    X = pca.transform(X)
    print('----- LDA...')
    # http://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html
    # https://stackoverflow.com/questions/31107945/how-to-perform-prediction-with-lda-linear-discriminant-in-scikit-learn
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    lda = lda.fit(X, y) #learning the projection matrix
    # ------ saving training data:
    print('----- Saving training data...')
    path_dataset_train = path_dataset_train_backup
    myImage_object.save_file(file=training_data, filename='training_data', path_save=path_dataset_train+'saved_train_files/')
    myImage_object.save_file(file=lda, filename='lda', path_save=path_dataset_train+'saved_train_files/')
    myImage_object.save_file(file=pca, filename='pca', path_save=path_dataset_train+'saved_train_files/')
    # ------ returns:
    return lda, pca

def Detect_Rain(image, myImage_object, lda, pca, path_rain_detection, save_intermediate_images=False):
    # ------ some settings:
    size_of_box = 100
    # ------ instantiate class:
    rainSnowDetector = RainSnowDetector()
    # ------ Sobel edge detection (vertical edges) on image:
    edge_map, horizontal_edge_map, vertical_edge_map, phase_sobel, horizontal_opposite_edge_map, vertical_opposite_edge_map = myImage_object.sobelEdgeDetection_manual(image=image, do_threshold=True, threshold=255/6)
    if save_intermediate_images:
        path_vertical_edge_map = path_rain_detection + 'vertical_edge_maps/'
        myImage_object.remove_folder(folder_path=path_vertical_edge_map)
        vertical_edge_map_showable = vertical_edge_map * 255  #--> change range from [0,1] to [0,255]
        myImage_object.save_image(image=vertical_edge_map_showable, name='edge_map', save_path=path_vertical_edge_map, format_of_save='jpg')
        vertical_opposite_edge_map_showable = vertical_opposite_edge_map * 255
        vertical_opposite_edge_map_showable = myImage_object.numpyArray2pilImage(array=vertical_opposite_edge_map_showable)
        myImage_object.save_image(image=vertical_opposite_edge_map_showable, name='edge_map_opposite', save_path=path_vertical_edge_map, format_of_save='jpg')
    # ----- divide image into several boxes, and then save the boxes:
    path_boxes = path_rain_detection + 'boxes/'
    myImage_object.remove_folder(folder_path=path_boxes)
    rainSnowDetector.divide_image_into_boxes(vertical_edge_map=vertical_edge_map, myImage_object=myImage_object, size_of_box=size_of_box, save_path=path_boxes, format_of_save='jpg')
    # ------ extract features:
    box_list = myImage_object.read_images(folder_path=path_boxes)
    number_of_boxes = myImage_object.count_number_of_files_in_folder(folder_path=path_boxes)
    test_data_boxes = np.empty((0, (size_of_box-2)**2))  #--> -2: because excluding boundary pixels
    for image_index in range(number_of_boxes):
        box = box_list[image_index]
        feature_vector = rainSnowDetector.local_binary_pattern(image=box, myImage_object=myImage_object)
        test_data_boxes = np.vstack([test_data_boxes, feature_vector])
    # ------ test (estimation of boxes):
    X = test_data_boxes
    # PCA:
    # http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html
    X = pca.transform(X)
    # LDA:
    # http://scikit-learn.sourceforge.net/0.6/auto_examples/plot_lda_vs_qda.html
    # http://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html
    # https://stackoverflow.com/questions/31107945/how-to-perform-prediction-with-lda-linear-discriminant-in-scikit-learn
    y_pred = lda.predict(X)
    estimated_classes_of_boxes = y_pred
    # ------ test (voting):
    image = myImage_object.pilImage2numpyArray(img=image)
    number_of_rows = image.shape[0]
    number_of_columns = image.shape[1]
    number_of_boxes_in_row = math.floor(number_of_rows/size_of_box)
    number_of_boxes_in_column = math.floor(number_of_columns/size_of_box)
    number_of_boxes_per_test_image = number_of_boxes_in_row * number_of_boxes_in_column
    # -----
    number_of_votes_for_not_rain = int(sum(estimated_classes_of_boxes))  # class 1: not rain
    number_of_votes_for_rain = number_of_boxes_per_test_image - number_of_votes_for_not_rain   # class 0: rain
    # -----
    rain_factor = (number_of_votes_for_rain / number_of_boxes_per_test_image) * 5 #--> range: [0, 5]
    rain_factor = int(np.ceil(rain_factor))  #--> quantizing rain_factor --> 1, 2, 3, 4, 5 --> 1: normal, 5: too rainy
    if rain_factor == 0:
        rain_factor = 1
    # ------ returns:
    return rain_factor, estimated_classes_of_boxes

if __name__ == '__main__':
    main()
