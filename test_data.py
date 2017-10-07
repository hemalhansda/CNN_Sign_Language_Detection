import numpy as np
import cv2
from keras.models import load_model
cap = cv2.VideoCapture(0)

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
        
        if(output==0):
            print("A")
        elif(output==1):
            print("B")
        elif(output==2):
            print("C")
        elif(output==3):
            print("V")
        else:
            print("MODEL DOESNT WORK :P")

        if (cv2.waitKey(25) & 0xFF=='q') :
            # np.save("{}-data.npy".format(label), training_data)
            cv2.destroyAllWindows()
            break

main()