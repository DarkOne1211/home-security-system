from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from picamera.array import PiRGBArray
from picamera import PiCamera
from collections import Counter
from sklearn import metrics
from imutils import paths
from time import sleep
import numpy as np
import cv2 as cv
import imutils
import pickle
import json
import time
import dlib
import os

#######################################################################
# function name: initialize
# INFO: Initialization function
######################################################################
def initialize():
    # Setting up folder structure
    if not (os.path.exists('recorded_images')):
        os.mkdir('recorded_images')
    if not (os.path.exists('verify_images')):
        os.mkdir('verify_images')
        os.mkdir('verify_images/test_images/')
    if not (os.path.exists('person_features')):
        os.mkdir('person_features')
    if not (os.path.exists('person_classifier')):
        os.mkdir('person_classifier')
    if not (os.path.exists('settings')):
        os.mkdir('settings')
    data = {}
    data['users'] = []
    with open('settings/settings.json', 'w') as outfile:
        json.dump(data, outfile,indent=4)
    outfile.close()
    

#######################################################################
# function name: add_user
# return codes: 
#   1     - sucessful
#  -1     - failed, none of the images had faces
#   0     - failed, More than one person detected in images
# INFO: Initialization function for a new user
######################################################################
def add_user(name):
    data = {}
    if not (os.path.exists('recorded_images/' + name)):
        os.mkdir('recorded_images/' + name)
    if not (os.path.exists('person_features/' + name)):
        os.mkdir('person_features/' + name)
    with open('settings/settings.json', 'r') as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    with open('settings/settings.json', 'w') as jsonFile:
        data['users'].append(name)
        data[name] = []
        json.dump(data, jsonFile, indent=4)
    jsonFile.close()
    captureImages('recorded_images', name, 10)
    f = extract_align_faces('recorded_images', name)
    if f != 1:
        return f
    extract_features(name)

#######################################################################
# function name: captureImages
# arguements:
#   folder - high level folder to store all images
#   name   - name of the person who's images are being stores
#   t      - time t that the program takes pictures 
# INFO: This function uses the camera to take pictures for t seconds
######################################################################

def captureImages(folder, name, t):
    #Set up scheduler to take a picture every 0.25 seconds
    #Start video capture
    captureNumber = 0
    camera = PiCamera()
    camera.resolution = (640,480)
    camera.start_preview()
    sleep(0.5)
    cap = PiRGBArray(camera)
    tEnd = time.monotonic() + t
    while(time.monotonic() < tEnd):
        #capture frame-by-frame
        camera.capture(cap, format="rgb")
        frame = cap.array

        #convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #display the resulting frame
        cv.imwrite(folder + '/' + name + '/' + 'testImage{0}.png'.format(str(captureNumber)),gray)
        if(cv.waitKey(1) & 0xFF == ord('q')):
            break
        captureNumber += 1
        cap.truncate()
        cap.seek(0)
    camera.stop_preview()
    cv.destroyAllWindows()

##############################################################
# function name: extract_align_faces
# agruements:
#   folder - folder where all images are stored
#   name   - name of the specific person
# return codes: 
#   1     - sucessful
#  -1     - failed, none of the images had faces
#   0     - failed, More than one person detected in images
# INFO: This function goes through all the existing photos
#       of one user and extracts and aligns their faces
##############################################################

def extract_align_faces(folder, name):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_classifier/shape_predictor_68_face_landmarks.dat')
    face_align = FaceAligner(predictor,desiredFaceWidth=256)

    # load images and convert it to grayscale
    for filename in os.listdir(folder + '/' + name):

        #Load all recorded images and convert to grayscale
        path = folder + '/' + name + '/' + filename
        image = cv.imread(path)
        image = imutils.resize(image, width=800)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        #detect faces
        ##rects = detector(gray, 2)
        rects = detector(gray,0)
        #display rectangles over the faces
        if(len(rects) < 1):
            os.remove(path)
            continue
        if(len(rects) > 1):
            print("Only one person can be in the frame at a time")
            return 0
        for rect in rects:
            (x,y,w,h) = rect_to_bb(rect)
            face_aligned = face_align.align(image,gray,rect)
            cv.imwrite(path, face_aligned)

    if len(os.listdir(folder + '/' + name)) == 0:
        return -1
    else:
        return 1
    return 1

##############################################################
# function name: extract_features
# agruements:
#   name   - name of the specific person
# INFO: This function goes through all the existing photos
#       of one user and turns them into 128 x 1 numpy vectors
#       and stores in a .pickle file
##############################################################

def extract_features(name):
    #Load the siamese nueral network
    feature_extractor = cv.dnn.readNetFromTorch('face_classifier/nn4.small2.v1.t7')
    names = []
    features = []

    #Extract the features for all images
    for filename in os.listdir('recorded_images' + '/' + name):
        path = 'recorded_images/' +  name + '/' + filename
        image = cv.imread(path)
        faceBlob = cv.dnn.blobFromImage(image, 1.0/255,
                    (96,96),(0,0,0),swapRB=True,crop=False)
        feature_extractor.setInput(faceBlob)
        vec = feature_extractor.forward()
        names.append(name)
        features.append(vec.flatten())
        
    #Writing all features to disk
    data = {"Features": features, "Names": names}
    f = open('person_features/' + name + '/' + '{}.pickle'.format(names[0]),"wb")
    f.write(pickle.dumps(data))
    f.close()

##############################################################
# function name: train_knn
# agruements:
#   name   - name of the specific person
# INFO: This function goes through all the existing feature
#       vector and trains a knn classfier using it and stores
#       it a .pickle file
##############################################################

def train_knn():
    with open('settings/settings.json', 'r') as jsonFile:
        data = json.load(jsonFile)
    jsonFile.close()
    
    # stores all the availble user labels
    usernames = []
    userFeatures = []
    for name in data['users']:
        #load image feature data
        data = pickle.loads(open('person_features' + '/' + name + '/' + name + '.pickle',"rb").read())
        usernames += data['Names']
        userFeatures += data["Features"]
    # create label encoder
    le = LabelEncoder()
    labels = le.fit_transform(usernames)

    allScores = []
    for k in range(1, int(len(usernames)/2)):
        # create and train a knn model using the feature vector
        X_train, X_test, y_train, y_test = train_test_split(
            userFeatures, 
            labels, 
            test_size=0.33,
            random_state = 42
        )

        recog = KNeighborsClassifier(n_neighbors = k)
        recog.fit(X_train, y_train)
        y_pred = recog.predict(X_test)
        scores = metrics.accuracy_score(y_test, y_pred)
        allScores.append(scores)

    highestK = np.argmax(allScores)
    recognizer = KNeighborsClassifier(n_neighbors = highestK + 1)
    recognizer.fit(userFeatures, labels)

    # writing the face recognition model to disk
    f = open('person_classifier/knn_model.pickle',"wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # writing a label encoder to disk
    f = open('person_classifier/le.pickle',"wb")
    f.write(pickle.dumps(le))
    f.close()

##############################################################
# function name: recognize_face
# return value:
#   <user_at_camera> - returns a string of the most probable
#                      face
#   
# INFO: This function return the most likely person in front
#       of the camera. Returns unknown if the user is unknown
##############################################################
def recognize_face():
    captureImages('verify_images', 'test_images', 1)
    extract_align_faces('verify_images','test_images')

    # load the siamese network model
    feature_extractor = cv.dnn.readNetFromTorch('face_classifier/nn4.small2.v1.t7')
    # load the knn model
    recognizer = pickle.loads(open('person_classifier/knn_model.pickle',"rb").read())

    #load the label encoder
    le = pickle.loads(open('person_classifier/le.pickle',"rb").read())
    features = []
    name = []
    totalPics = 0
    #Extract the features for all images
    for filename in os.listdir('verify_images/test_images'):
        totalPics += 1
        path = 'verify_images/test_images/' + filename
        image = cv.imread(path)
        faceBlob = cv.dnn.blobFromImage(image, 1.0/255,
                    (96,96),(0,0,0),swapRB=True,crop=False)
        
        # put the picture through the siamese network
        feature_extractor.setInput(faceBlob)
        vec = feature_extractor.forward()
        features.append(vec.flatten())

    user_at_camera = []

    # put the resulting vectors the knn classifier
    for feature in features:
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        probability = preds[j]
        if(probability < 0.8):
            continue
        user_at_camera.append(le.classes_[j])

    if(len(user_at_camera) > 0):
        data = Counter(user_at_camera)
        mostCommon = data.most_common(1)[0]
        if(mostCommon[1] < totalPics * 0.5):
            return 'unknown'
        else:
            return mostCommon[0]
    else:
        return 'unknown'

    