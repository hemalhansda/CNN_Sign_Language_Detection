import numpy as np
import cv2
from keras.models import load_model
cap = cv2.VideoCapture(0)

#Check the labels in the labels array and modify them accordingly if they are not same.
labels=["A", "B", "C", "D", "E"]
def main():

    model = load_model("sign_model.h5")
    while True:
        
        ret, image = cap.read()
        cv2.imshow("Test", cv2.resize(image,(800,640)))
        image = cv2.resize(image, (160, 160))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((1,) + image.shape)
        output = model.predict(image, batch_size=1, verbose=1)
        
        output = (np.argmax(output[0]))
        if(output<=len(labels)):
            print(labels[output])
        else:
            print("Cannot Predict")

        if (cv2.waitKey(25) & 0xFF=='q') :
            # np.save("{}-data.npy".format(label), training_data)
            cv2.destroyAllWindows()
            break

main()