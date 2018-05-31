import os
import face_recognition
from annoy import AnnoyIndex
from skimage import io
from tqdm import tqdm
import json

vec_len = 128
photos = os.listdir("photos")

index = 0
index2photo = dict()
annoy = AnnoyIndex(vec_len)

for photo in tqdm(photos):
	try:
		image = io.imread("photos/"+photo)
	except Exception:
		print("error:", photo)
		continue

	for vec in face_recognition.face_encodings(image):
		index2photo[index] = photo
		annoy.add_item(index, vec)
		index+=1

print("Build annoy")
annoy.build(10)
print("Save annoy")
annoy.save('index.ann')

print("Save index2photo.json")
with open("index2photo.json", "w") as f:
	f.write(json.dumps(index2photo))

print("Success!")