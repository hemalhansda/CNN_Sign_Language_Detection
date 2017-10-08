import numpy as np

training_data=[]

#Check the labels in the labels array and modify them accordingly if they are not same. And also change the respective label in Elif conditions below.
labels=["A", "B", "C", "D", "E"]

def main():

    for label in labels:    
        image_data = np.load("{}-data.npy".format(label))
        #Change labels in the elif conditions beow if they are not the same.
        for data in image_data:
            if(label == "A"):
                output = [1,0,0,0,0,0,0,0,0,0]
            elif(label == "B"):
                output = [0,1,0,0,0,0,0,0,0,0]
            elif(label == "C"):
                output = [0,0,1,0,0,0,0,0,0,0]
            elif(label == "D"):
                output = [0,0,0,1,0,0,0,0,0,0]
            elif(label == "E"):
                output = [0,0,0,0,1,0,0,0,0,0]
            else:
                print("Please use correct Label")
                break
            training_data.append([data, output])
        
        np.save("{}-new_data.npy".format(label), training_data)
        print("Saved new {} data".format(label))
    print("Output appended for all labels")

main()