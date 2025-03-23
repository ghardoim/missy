from os.path import join, splitext
import face_recognition as fr
from os import listdir
import cv2

def load_familiar_faces(path:str="imgs"):
    return [ {"name": splitext(img)[0], "face": has_face[0]} for img in listdir(path)
        if img.endswith((".png", ".jpg", ".jpeg")) and (has_face := fr.face_encodings(fr.load_image_file(join(path, img)))) ]

def is_known_face(frame, familiar_faces:list[dict]):
    if not (detected_faces := fr.face_encodings(frame)): return None
    known_faces = [ f["face"] for f in familiar_faces ]

    for face in detected_faces:
        results = fr.compare_faces(known_faces, face)

        for r, result in enumerate(results):
            if result: return familiar_faces[r]

    return None

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        if not (face := is_known_face(frame, load_familiar_faces())):
            cv2.imwrite("imgs/new_face.png", frame)
            print("new face detected")

        else:
            (top, right, bottom, left) = fr.face_locations(frame)[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
            cv2.putText(frame, face["name"], (left + 10, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if "__main__" == __name__: main()