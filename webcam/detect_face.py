import cv2
import face_recognition
import pickle
import os

ENCODINGS_FILE = "encodings.pkl"

def main():

    if not os.path.exists(ENCODINGS_FILE):
        print("No saved faces. Run save_face.py first.")
        return

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    cam = cv2.VideoCapture(0)
    print("Starting face detection... Press 'q' to exit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(small_rgb)
        face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

        for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):

            matches = face_recognition.compare_faces(known_encodings, face_enc)
            name = "Unknown"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]

            # Scale back to original size
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
