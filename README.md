# Haze-Removal-Dark-Channel-Prior
The code for haze removal using dark channel prior, which was a part of the self-driving car project

This is my code implementation of the following paper:

He, Kaiming, Jian Sun, and Xiaoou Tang. "Single image haze removal using dark channel prior." IEEE transactions on pattern analysis and machine intelligence 33, no. 12 (2010): 2341-2353.

Note:
This project was for a vehicle project. In a part of this project, I implemented the dark channel prior method. If you run the whole project, you should put some images in the directory './input_images/'. Then, the function "weather_module()" in main.py tries to detect the levels of haze, light (luminance), and rain in the image. If haze removal is required, it calls the function "Remove_Haze()" in main.py. In that function, we instantiate the class "HazeRemoval" which is the implementation of dark channel prior method for haze removal. If you want to just use the module "HazeRemoval", you can slightly edit the function "Remove_Haze()" in main.py as desired and input the image(s) to that function directly.
