import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


def main():
    training_data = []
    data = np.load("A_new_training_data.npy")
    for dat in data:
        x = dat[0].reshape((1,) + dat[0].shape)
        print("Shape:", np.shape(x))
        y  = dat[1]
        training_data.append(data)
        i=0
        for x_batch in datagen.flow(x, batch_size=1, shuffle=False):
            training_data.append([x_batch[0], y])
            i += 1
            if i > 9:
                break
    print("ADded shape",np.shape(training_data[-1][0]))

    np.save("A_new_training_data_1.npy", training_data)
    print("Saved")


main()