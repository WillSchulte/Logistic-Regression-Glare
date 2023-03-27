# Logistic-Regression-Glare
Demonstration of a logistic regression-based machine learning that recognizes sun glare in a set of images.

The file 'preprocess.py' processes the images into flat arrays of RGB data, then uploads them to their respective .csv files.

The file 'main.py' loads the processed data, then runs a logistic regression model on a set of training images, then tests itself on a
subset of test images. A graph is then displayed (figure.png) that shows the cost v iterations and the train accuracy v iterations. The final accuracy on a set of test images is also displayed.
