import numpy as np
import cv2

cap = cv2.VideoCapture(0)

label = 'A'

def main():
    count = 1
    training_data = []
    while True:
        print(count)
        ret, image = cap.read()
        cv2.imshow("Test", cv2.resize(image,(800,640)))
        image = cv2.resize(image, (160, 160))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        training_data.append(image)
        count = count+1
        if (cv2.waitKey(25) & 0xFF=='q') or count == 50 :
            np.save("{}-data.npy".format(label), training_data)
            cv2.destroyAllWindows()
            break

main()