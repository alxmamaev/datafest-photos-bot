import telebot
import json
import face_recognition
from skimage import io
from scipy.spatial.distance import cosine

face_vectors = []
index2name = dict()


states = dict()
user_data = dict()

config = json.loads(open("config.json").read())


bot = telebot.TeleBot(config["token"])


def get_state(message):
	return index2name.get(message.chat.id, "get-photo")

def set_state(message, state):
	index2name[message.chat.id] = state


def get_nn_vector_index(vector):
    distance = 2
    index = None
    for i, v in enumerate(face_vectors):
        d = cosine(v, vector)
        
        if d<distance:
            distance = d
            index = i
            
    return index, distance




@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
	bot.send_message(message.chat.id, "Привет, я бот, я могу распознавать людей по лицам.\nДля этого просто отправь свое селфи 🤳.")


@bot.message_handler(content_types=["photo"], func=lambda m: get_state(m) == "get-photo")
def process_selfie(message):
	bot.reply_to(message, "⏳ Обрабатываю твой селфач.")


	fileID = message.photo[-1].file_id
	file_info = bot.get_file(fileID)
	downloaded_file = bot.download_file(file_info.file_path)


	image = io.imread(downloaded_file, plugin='imageio')
	vectors = face_recognition.face_encodings(image)
	
	if not vectors:
		bot.reply_to(message, "Тут нет ни одного лица 😢")
		return 
	elif len(vectors) > 1:
		bot.reply_to(message, "Слишком много лиц на фото 😨")
		return

	index, distance = get_nn_vector_index(vectors[0])

	if distance < 0.071:
		bot.reply_to(message, "Я тебя знаю, привет %s 👀" % index2name[index])
	else:
		bot.reply_to(message, "Кажется, я тебя не знаю 🤔, напиши свое имя.")
		user_data[message.chat.id] = len(face_vectors)
		face_vectors.append(vectors[0])
		set_state(message, "get-name")

@bot.message_handler(content_types=["text"], func=lambda m: get_state(m) == "get-name")
def process_name(message):
	bot.reply_to(message, "👌 Я тебя запомнил")
	index2name[user_data[message.chat.id]] = message.text
	set_state(message, "get-photo")


bot.polling()