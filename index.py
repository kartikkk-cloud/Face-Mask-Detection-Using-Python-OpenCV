import tkinter as tk
import cv2 
import tensorflow
from PIL import ImageTk, Image
from tkinter import font
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import winsound


class madsk:
    
    def __init__(self):
        root = tk.Tk()
        root.geometry("1920x1080")
        frame = tk.Frame(root, bg= "blue")



        root.title("Mask Detection GUI")

        image = tk.PhotoImage(file="C:\CAPSTONE1\maskk12.png") 
        label = tk.Label(root, image=image)
        label.place(relwidth=1, relheight=1)


        label1 = tk.Label(root, text="Scan for Mask Detection",font=font.Font(size=14))
        label1.place(x=1135, y=650)
        imaged= tk.PhotoImage(file="C:\CAPSTONE1\person.png") 
        imaged = imaged.subsample(5,5)

                
        button = tk.Button(root, image=imaged, compound="center",command=self.mask_key)
        button.place(x=1150, y=450)

        root.mainloop()


    def mask_key(self):
            
            
            def detect_and_predict_mask(frame, faceNet, maskNet):
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
                faceNet.setInput(blob)
                detections = faceNet.forward()
                print(detections.shape)

                
                faces = []
                locs = []
                preds = []


                for i in range(0, detections.shape[2]):
                    
                    confidence = detections[0, 0, i, 2]

                    
                    if confidence > 0.5:
                        
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        
                        (startX, startY) = (max(0, startX), max(0, startY))
                        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                        
                        face = frame[startY:endY, startX:endX]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = cv2.resize(face, (224, 224))
                        face = img_to_array(face)
                        face = preprocess_input(face)

                        
                        faces.append(face)
                        locs.append((startX, startY, endX, endY))

                
                if len(faces) > 0:
                    
                    faces = np.array(faces, dtype="float32")
                    preds = maskNet.predict(faces, batch_size=32)

                
                return (locs, preds)


            prototxtPath = r"C:\CAPSTONE1\face_detector\deploy.prototxt"
            weightsPath = r"C:\CAPSTONE1\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
            faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

            maskNet = load_model("mask_detector.model")


            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()


            while True:
                
                frame = vs.read()
                frame = imutils.resize(frame, width=700)

                
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                
                for (box, pred) in zip(locs, preds):
                
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred

                    
                    label = "Mask" if mask > withoutMask else "No Mask"
                    if label == 'No Mask':
                        winsound.Beep(1000, 50)
                    
                

                    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                    
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                    
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


                cv2.imshow("MASK DETECTION SYSTEM", frame)
                key = cv2.waitKey(1) & 0xFF

                
                if key == ord("q"):
                    break

            cv2.destroyAllWindows()
            vs.stop()

    

obj = madsk()
