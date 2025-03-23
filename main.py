from os.path import join, splitext, getmtime
from datetime import datetime as dt
import face_recognition as fr
from os import listdir
import cv2

def load_familiar_faces(path:str="imgs") -> list[dict]:
    return [ {"name": splitext(img)[0], "face": has_face[0], "img": fullpath, "when": dt.fromtimestamp(getmtime(fullpath))} for img in listdir(path)
        if img.endswith((".png", ".jpg", ".jpeg")) and (has_face := fr.face_encodings(fr.load_image_file(fullpath:=join(path, img)))) ]

def is_known_face(frame, familiar_faces:list[dict]) -> dict:
    if not (detected_faces := fr.face_encodings(frame)): return None
    known_faces = [ f["face"] for f in familiar_faces ]

    for face in detected_faces:
        results = fr.compare_faces(known_faces, face)

        for r, result in enumerate(results):
            if result: return familiar_faces[r]

    return {"name": "isnew", "face": detected_faces[0]}

def tag_face(frame, imgpath:str, label:str) -> None:
    cv2.imwrite(imgpath, frame)

    (top, right, bottom, left) = fr.face_locations(frame)[0]
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 2)
    cv2.putText(frame, label, (left - 50, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        if not (face := is_known_face(frame, load_familiar_faces())):
            cv2.putText(frame, "nenhum rosto detectado", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        else:
            tag_face(frame, *["imgs/newperson.png", "new face"] if "isnew" == face["name"] else [ 
                face["img"], f'{face["name"]}, visto por ultimo dia {face["when"].day}' ])

        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if "__main__" == __name__: main()