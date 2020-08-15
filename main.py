# import tensorflow.keras as keras
import cv2
from facenet_pytorch import MTCNN
from tensorflow.keras.models import model_from_json
import numpy as np

categories = ['with_mask' , 'without_mask']
#loading the json_file

def load_model():
    print('[INFO] loading the model')
    load_json = open('mask_detection.json' , 'r')
    load_facemask_detection_model = load_json.read()
    facemask_detection_model = model_from_json(load_facemask_detection_model)
    facemask_detection_model.load_weights('mask_detection.h5')
    print('[INFO] model loaded')
    return facemask_detection_model

def webcam():
    #load the face mask model
    facemask_detection_model = load_model()

    #load the face detection model
    print('[INFO] loading the face detectiom model')
    mtcnn = MTCNN( keep_all = True , post_process = False , image_size=224)
    print('[INFO] loading face detectiom model')


    cap = cv2.VideoCapture(0)
 
    while cap:
        ret , frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            faces = mtcnn(frame)
            for face in faces:
                detected_face = face.permute(1,2,0).int().numpy()
                output = facemask_detection_model(detected_face.reshape(1 , 224 , 224 , 3))
                print(categories[np.argmax(output)])
                cv2.imshow('frame' , frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        except:
            continue
        # cap.release()
        # cv2.destroyAllWindows()

webcam()
