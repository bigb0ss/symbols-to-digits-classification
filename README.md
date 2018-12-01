# symbol-digit-detection


Dataset: 
The dataset consists of five classes (1-5) ,each class containing 1000 image samples each. The dataset set is prepared manually by using OpenCV 2 toolkit . The images are converted into hsv ,for easy scaling , and its made to scale the images to detect the skin tone. The dataset is read and stored as image_data.npy file for easy access.

The Model is trained using Keras deep learning framework , using the convolutional layers.The model is trained with an accuracy of 98% which is a acceptable metric in standards of deep learning.

And the model is deployed using OpenCV ,on a live cam feed to continously detect the symbols that are captured by the webcam.

main.py is the executable file ,which has the executable module