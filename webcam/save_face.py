import cv2
import face_recognition
import os
import pickle
import time

ENCODINGS_FILE = "encodings.pkl"

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_encodings(data):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

def main():
    data = load_encodings()

    name = input("Enter name to register: ").strip()

    print("\nOpening camera... Look directly into the camera.\n")
    cam = cv2.VideoCapture(0)

    captured_face = None
    face_encoding = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Camera - Please look at the screen", frame)
        cv2.waitKey(1)

        # Detect face
        face_locations = face_recognition.face_locations(frame)

        if len(face_locations) == 1:
            print("Face detected! Capturing...")

            top, right, bottom, left = face_locations[0]
            captured_face = frame[top:bottom, left:right]

            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

            # Show captured face for 10 seconds
            for i in range(5, 0, -1):
                msg_img = captured_face.copy()
                cv2.putText(msg_img, f"{i}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Captured Face", msg_img)

                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    print("Rejected capture. Trying again...")
                    captured_face = None
                    break
            break

        elif len(face_locations) > 1:
            print("Multiple faces detected! Please stay alone in frame.")

    cam.release()
    cv2.destroyAllWindows()

    if captured_face is None:
        print("No valid face captured. Exiting.")
        return

    # Ask to save
    save_decision = input(f"Do you want to save face for '{name}'? (y/n): ").strip().lower()
    if save_decision != 'y':
        print("Face not saved.")
        return

    # Save data
    data["encodings"].append(face_encoding)
    data["names"].append(name)
    save_encodings(data)


    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    img_path = os.path.join(SCRIPT_DIR, f"{name}.jpg")
    cv2.imwrite(img_path, captured_face)

    print(f"Face data saved for: {name}")

if __name__ == "__main__":
    main()
