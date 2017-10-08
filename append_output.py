import numpy as np

training_data=[]
labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def main():

    for label in labels:    
        image_data = np.load("{}-data.npy".format(label))
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
            elif(label == "F"):
                output = [0,0,0,0,0,1,0,0,0,0]
            elif(label == "G"):
                output = [0,0,0,0,0,0,1,0,0,0]
            elif(label == "H"):
                output = [0,0,0,0,0,0,0,1,0,0]
            elif(label == "I"):
                output = [0,0,0,0,0,0,0,0,1,0]
            elif(label == "J"):
                output = [0,0,0,0,0,0,0,0,0,1]
            else:
                print("Please use correct Label")
                break
            training_data.append([data, output])
        
        np.save("{}-new_data.npy".format(label), training_data)
        print("Saved new {} data".format(label))
    print("Output appended for all labels")

main()