import numpy as np
import cv2

cap = cv2.VideoCapture(0)

label = 'A'
training_data = []
def main():
    count = 0
    while True:
        print(count)
        ret, image = cap.read()
        cv2.imshow("Test", cv2.resize(image,(800,600)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        training_data.append(image)
        count = count+1
        if (cv2.waitKey(25) & 0xFF=='q') or count==101 :
            np.save("{}-data".format(label), training_data)
            cv2.destroyAllWindows()
            break

main()