import cv2
import urllib.request
import numpy as np

import face_recognition
from os import walk

# Ket noi stream url
print("Connecting to url...")
stream_url = 'http://192.168.4.1:81/stream'
stream = urllib.request.urlopen(stream_url)
print("Success connect to "+ stream_url)
# print(type(stream))

# Tai thu muc chua anh
mypath = "C:/Users/pbhhe/OneDrive/Documents/Python/learning/image"
f = []
for root, dirs, files in walk(mypath):
    # print (root)
    # print (dirs)
    # print (files)
    f.extend(files)
    break

known_face_encodings = []
known_face_names = []

print("Loading image data...")
for img in f:
    # print (img)
    image = face_recognition.load_image_file(mypath+'/'+img)

    image_face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.extend([image_face_encoding])

    image_name = img.split(".")[0]
    known_face_names.extend([image_name])

print("Image data loaded.")
print("Stream start.")
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

bytes = b''
while True:
    bytes += stream.read(1024)
    # print(bytes)    
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        # print('1')
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        if jpg != b'':
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        #TODO
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        ##
        cv2.imshow('http://192.168.4.1:81/stream', frame)
        if cv2.waitKey(1) == 27:
            exit(0)
cv2.destroyAllWindows()