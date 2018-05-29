import face_recognition
import dlib
import numpy as np

predictor_model = "./shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()

def vectorize_photo(photo):
	detected_faces = face_detector(photo, 1)

	vectors = []

	for face_shape in detected_faces:
		face_photo = photo[face_shape.top():face_shape.bottom(),face_shape.left():face_shape.right()]
		vector = face_recognition.face_encodings(face_photo)[0]

		vectors.append(vector)

	return vectors