import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
import pyttsx3


track_history = defaultdict(lambda: [])
model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture(0) 
assert cap.isOpened(), "Error opening webcam"

w, h, fps = 320, 240, 30

# result = cv2.VideoWriter("object_tracking.avi",
                    #    cv2.VideoWriter_fourcc(*'mp4v'),
                    #    fps,
                    #    (w, h))

tts_engine = pyttsx3.init()
speech_rate = tts_engine.getProperty('rate') 
new_speech_rate = 100  
tts_engine.setProperty('rate', new_speech_rate)
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[1].id)

while cap.isOpened():
    success, frame = cap.read()
    if success:
       
        frame = cv2.resize(frame, (w, h))

        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:
           
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

           
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

               
                object_name = names[int(cls)]
                tts_engine.say('there is a' + object_name+" in front  ")
                tts_engine.runAndWait()

               
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))

        # result.write(frame)
        cv2.imshow("Object Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# result.release()
cap.release()
cv2.destroyAllWindows()
