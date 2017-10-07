from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from random import shuffle

WIDTH = 160
HEIGHT = 160
def main():

    training_data=[]
    data = np.load("A_new_training_data_1.npy")
    for dat in data:
        training_data.append(dat)
    data = np.load("B_new_training_data_1.npy")
    for dat in data:
        training_data.append(dat)
    data = np.load("C_new_training_data_1.npy")
    for dat in data:
        training_data.append(dat)
    data = np.load("V_new_training_data_1.npy")
    for dat in data:
        training_data.append(dat)

    print("Training_data length", len(training_data))
    shuffle(training_data)
    shuffle(training_data)
    shuffle(training_data)

    t = int(0.9*len(training_data))
    
    train = training_data[:t]
    print(len(train))
    test = training_data[t:]
    print(len(test))
    training_data = []
    print("Length of i[0]", np.shape(train[0][0]))
    X_train = np.array([i[0] for i in train])
    print("X_Trainiing shape", np.shape(X_train))
    Y_train = [i[1] for i in train]
    X_valid = np.array([i[0] for i in test])
    Y_valid = [i[1] for i in test]

    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(WIDTH, HEIGHT, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #  let's add a fully-connected layer 
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(4, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print("Compiled")
    print("Starting training")
    model.fit(model.fit(X_train, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid)))
    model.save("sign_model.h5")
    del model
    print("model saved")

main()