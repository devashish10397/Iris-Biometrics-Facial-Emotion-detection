import socket
import cv2
import os
import shutil
import random
import numpy as np
#from pygame import mixer
#from time import sleep


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
fishface = cv2.face.FisherFaceRecognizer_create() #Init fisher face classifier
#lbpface = cv2.face.LBPHFaceRecognizer_create() # Init LBPH classifier
data = {}


def connect(message):
    s = socket.socket()
    s.connect(('192.168.2.20',7443))
    #while True:
    str = message
    s.send(str.encode());
        #if(str == "Bye" or str == "bye"):
        #    break
        #print ("Received Message:",s.recv(1024).decode())
    s.close()




def captureLiveImage(counter):
    cap = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('F:\\MajorProject\\Emotion_Dataset\\Testing\\{index}.png'.format(index=counter),frame)
            break

    cap.release()
    cv2.destroyAllWindows()






def extract_Face():

    faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
    files = os.listdir("F:\\MajorProject\\Emotion_Dataset\\Testing\\")
    filenumber = 0


    for f in files:
        frame = cv2.imread("F:\\MajorProject\\Emotion_Dataset\\Testing\\"+f)
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)


        if len(face) == 1:
            facefeatures = face
                
        elif len(face_two) == 1:
            facefeatures = face_two
                
        elif len(face_three) == 1:
            facefeatures = face_three
                    
        elif len(face_four) == 1:
            facefeatures = face_four
                    
        else:
            facefeatures = ""


        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print ("Face Detected in File : ",f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size

            newImg = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
            cv2.imwrite("F:\\MajorProject\\Emotion_Dataset\\TestingImages\\"+str(filenumber)+".png", newImg)


        filenumber += 1







def get_files(emotion): #split 80-20

    files = os.listdir("F:\\MajorProject\\Emotion_Dataset\\"+emotion+"\\")
    files2 = os.listdir("F:\\MajorProject\\Emotion_Dataset\\TestingImages\\")

    random.shuffle(files)
    random.shuffle(files2)

    training = files[:int(len(files)*0.8)]
    prediction = files2[:int(len(files2)*1)]
    #prediction = files[-int(len(files)*0.2):]

    return training, prediction






def make_sets():
    
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    
    for emotion in emotions:
        training, prediction = get_files(emotion)

        #Data appended to the training list, and generates labels 0-7
        for item in training:
            image = cv2.imread("F:\\MajorProject\\Emotion_Dataset\\"+emotion+"\\"+item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))

        #Data appended to the prediction list, and generates labels 0-7
        for item in prediction:
            #print(item)
            image = cv2.imread("F:\\MajorProject\\Emotion_Dataset\\TestingImages\\"+item)
            #image = cv2.imread("D:\\MajorProject\\Emotion_Dataset\\"+emotion+"\\"+item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels







def start_recog():

    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    print("# Sets Created Sucessfully!!")
    print ("\n\n----> Training Fisher Face Classifier <----")
    print ("> The Size of Training Set : ",len(training_labels)," Images")

    #print(training_data)
    fishface.train(training_data, np.asarray(training_labels))
    
    print ("\n> Let's Predict Classification Set")
    cnt = 0
    correct = 0
    incorrect = 0
    
    for image in prediction_data:
        pred, conf = fishface.predict(image)
        
        if pred == prediction_labels[cnt]:
            print("\n> Prediction : ",emotions[pred])
            correct += 1
            cnt += 1

            if(emotions[pred]=="happy"):
                connect(emotions[pred]);
            elif(emotions[pred]=="sad"):
                connect(emotions[pred]);
            elif(emotions[pred]=="neutral"):
                connect(emotions[pred]);
            elif(emotions[pred]=="anger"):
                connect(emotions[pred]);
            else:
                connect(emotions[pred]);

        else:
            incorrect += 1
            cnt += 1

    return ((100*correct)/(correct + incorrect))





# PHD's Main Function

Score = []

for i in range(0,1):
    print("\n\n#-----------------------------------------------------------#\n\n")
    
    for counter in range(0,1):
        captureLiveImage(counter)
        print("#> Image Captured..")

    print("\n")
    
    extract_Face()
    correct = start_recog()
    

