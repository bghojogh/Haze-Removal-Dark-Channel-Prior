format of saving condition of images in text file:

light_factor + [TAB] + haze_factor + [TAB] + rain_factor + [TAB] + is_luminance_enhanced? + [TAB] + is_haze_removed?

=================== 

range of parameters:

light_factor --> 1 (normal = luminant), 2, 3, 4, 5 (dark)

haze_factor --> 1 (normal = not hazy), 2, 3, 4, 5 (too hazy)

rain_factor --> 1 (normal = not rainy), 2, 3, 4, 5 (too rainy)

is_luminance_enhanced? --> True, False

is_haze_removed? --> True, False

===================

Notice: 

if (is_luminance_enhanced?==True) or (is_haze_removed?==True):
     the original image of dataset has changed in folder "enhanced_dataset"
else:
     the original image of dataset has been copied to folder "enhanced_dataset" without any change

===================

Important note:

Before running code (starting on the dataset), delete this text file (otherwise you will append the results of previous run):
./condition_of_images/condition_of_images.txt