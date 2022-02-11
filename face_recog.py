import numpy as np
import face_recognition as fr
import cv2
from PIL import Image
import os

video_capture = cv2.VideoCapture(0)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = []
images = []
for filename in os.listdir(BASE_DIR):
    img = cv2.imread(os.path.join(BASE_DIR,filename))
    if img is not None:
        images.append(img)

file_name = []
# bg_image = fr.load_image_file("BillGates.jpg")
# print(bg_image)
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg"):
            path.append(os.path.join(root, file)) #Images 's path
            file_name.append(file)

print(path) 
bg_image = []
bg_face_encoding = []
known_face_encondings = []
known_face_names = []
print(file_name) 
for i in range(len(path)):
    bg_image.append(fr.load_image_file(path[i]))
    bg_face_encoding.append(fr.face_encodings(bg_image[i])[0])
    known_face_encondings.append([bg_face_encoding[i]])
    known_face_names.append(file_name[i])

print(known_face_encondings[0])
while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
    matches = []
    face_distances  =[0]*len(path)
    print(face_distances)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        name = "Unknown"
        for j in range(len(path)):
            matches.append(fr.compare_faces(known_face_encondings[j], face_encoding, tolerance = 0.545))
            #print(matches)

            face_distances[j] = fr.face_distance(known_face_encondings[j], face_encoding)
            #print(face_distances)
        
        print(matches)
        print(face_distances)
        best_match_index = np.argmin(face_distances)
        print(best_match_index)
        if matches[best_match_index] == [True]:
            name = known_face_names[best_match_index]    
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()