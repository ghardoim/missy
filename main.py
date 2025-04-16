from utils.known_face import *
import cv2

def tag_face(frame, name:str, location:tuple) -> None:
    (top, right, bottom, left) = location

    cv2.rectangle(frame, (left - 20, top - 20), (right + 20, bottom + 20), (0, 0, 0), 2)
    cv2.putText(frame, f"Seja Bem Vindo {name}!", (left - 50, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

known_faces = load_known_faces()

def main():
    cap = cv2.VideoCapture(0)
    process_frame = True

    while True:
        ret, frame = cap.read()
        if not ret: break

        if process_frame: name, location = is_known_face(frame, known_faces)
        if name and location: tag_face(frame, name, location)

        cv2.imshow("", frame)
        process_frame = not process_frame

        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if "__main__" == __name__: main()