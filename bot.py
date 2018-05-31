import telebot
import json
import face_recognition
from annoy import AnnoyIndex
from skimage import io

vec_len = 128
annoy = AnnoyIndex(vec_len)
index2iphoto = json.loads(open("index2photo.json").read())
config = json.loads(open("config.json").read())


annoy.load("index.ann")
bot = telebot.TeleBot(config["token"])


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
	bot.send_message(message.chat.id, "Привет, я бот, который поможет тебе найти свои фоточки с 🎉 Datafest.\nДля этого просто отправь свое селфи 🤳.")


@bot.message_handler(content_types=["photo"])
def process_selfie(message):
	bot.reply_to(message, "⏳ Обрабатываю твой селфач.")


	fileID = message.photo[-1].file_id
	file_info = bot.get_file(fileID)
	downloaded_file = bot.download_file(file_info.file_path)


	image = io.imread(downloaded_file, plugin='imageio')
	vectors = face_recognition.face_encodings(image)
	
	if not vectors:
		bot.reply_to(message, "Тут нет ни одной рожи")
		return 

	indexes, distances = annoy.get_nns_by_vector(vectors[0], 15, include_distances=True)

	if distances[0] < 0.35:
		bot.reply_to(message, "Йоу, зацени что нашел 👀")
		for index, distance in zip(indexes, distances):
			if distance >= 0.35:
				break

			photo_path = "./photos/" + index2iphoto[str(index)]
			
			with open(photo_path, "rb") as photo:
				try:
					bot.send_photo(message.chat.id, photo)
				except Exception:
					print("Error on ", photo_path)
					continue

		bot.send_message(message.chat.id, "Это все 🤷‍♀️.")
	else:
		bot.reply_to(message, "Кажется, я ничего не нашел 🤔")


bot.polling()