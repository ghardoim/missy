from os.path import join, splitext, getmtime
from datetime import datetime as dt
import face_recognition as fr
from os import listdir

def load_known_faces(path:str="imgs") -> list[dict]:
    known_faces = []

    for img in listdir(path):
        if not img.lower().endswith((".png", ".jpg", ".jpeg")): continue
        encodings = fr.face_encodings(fr.load_image_file(fullpath := join(path, img)))

        if encodings:
            known_faces.append({
                "name": splitext(img)[0], "encodings": encodings[0], "img": fullpath,
                "when": dt.fromtimestamp(getmtime(fullpath)), "isnew": False
            })
    return known_faces

def is_known_face(frame, known_faces:list[dict]) -> tuple:
    if not (detected_faces := fr.face_encodings(frame)): return (None, None)

    results = fr.compare_faces([ f["encodings"] for f in known_faces ], detected_faces[0])
    location = fr.face_locations(frame)[0]

    for r, result in enumerate(results):
        if result: return known_faces[r]["name"], location

    return "Visitante", location