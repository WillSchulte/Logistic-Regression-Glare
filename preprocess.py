import os
import numpy as np
import csv
from PIL import Image

train_set_dir = ('C:/Users/Will Schulte/Desktop/train_data')
test_set_dir = ('C:/Users/Will Schulte/Desktop/test_data')
os.chdir(train_set_dir)

#image_resolution = [167,167]
#image_resolution = [100,100]
image_resolution = [600,316]

#4d array containing the R,G, and B values for each pixel of each image in the dataset
train_set_ary = []
test_set_ary = []

order = []
order2 = []

#loop through all images in the dataset folder
for folder in os.listdir(train_set_dir):
    if not (folder.endswith(".csv")):
        order.append(folder)
        for file in enumerate(os.listdir(folder)):
            if file[0]%2 == 0:
                continue
            if file[0]%3 == 0:
                continue
            #open image with PIL
            image = Image.open(train_set_dir + '/' + folder + '/' + file[1])
            #resize image for standardization
            image = image.resize(image_resolution)
            #load 2d array of all pixels in image
            pixel_ary = image.load()
            #make three array for each of the 3 color values of each image
            red = []
            green = []
            blue = []

            #loop though all of the pixels in image
            for x in range(image.size[0]):
                #for each new row, add a row on to the red, green, and blue arrays
                red.append([])
                green.append([])
                blue.append([])
                for y in range(image.size[1]):
                    #for each pixel, add R,G,or B value to the corrosponding array
                    red[x].append(pixel_ary[x, y][0])
                    green[x].append(pixel_ary[x, y][1])
                    blue[x].append(pixel_ary[x, y][2])

            #combine R,G, and B arrays in to one array and add them to the dataset array
            rgb = [red, green, blue]
            train_set_ary.append(rgb)
            print("done: " + file[1])

os.chdir(test_set_dir)
for folder in os.listdir(test_set_dir):
    if not (folder.endswith(".csv")):
        order2.append(folder)
        for file in os.listdir(folder):
            image = Image.open(test_set_dir + '/' + folder + '/' + file)
            image = image.resize(image_resolution)
            pixel_ary = image.load()
            red = []
            green = []
            blue = []
            for x in range(image.size[0]):
                red.append([])
                green.append([])
                blue.append([])
                for y in range(image.size[1]):
                    red[x].append(pixel_ary[x, y][0])
                    green[x].append(pixel_ary[x, y][1])
                    blue[x].append(pixel_ary[x, y][2])
            rgb = [red, green, blue]
            test_set_ary.append(rgb)

#flatten 4d array in to 2d array
x = np.array(train_set_ary)
train_ary_flat = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
y = np.array(test_set_ary)
test_ary_flat = y.reshape(y.shape[0], y.shape[1]*y.shape[2]*y.shape[3])

print("train size: ", len(train_ary_flat[0]))
print("test size: ", len(test_ary_flat[0]))

# standardize data
train_ary_flat = train_ary_flat / 255
test_ary_flat = test_ary_flat / 255

#save data to csv + transpose to make it fit better in the csv
os.chdir(train_set_dir)
np.savetxt("training_data.csv", train_ary_flat.T, delimiter=',', fmt='%f')
order_np = np.array(order).T
np.savetxt("order.csv", [order_np], delimiter=',', fmt='%s')

os.chdir(test_set_dir)
np.savetxt("test_data.csv", test_ary_flat.T, delimiter=',', fmt='%f')
order2_np = np.array(order2).T
np.savetxt("order.csv", [order2_np], delimiter=',', fmt='%s')