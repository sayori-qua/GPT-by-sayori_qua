from logging import exception
import telebot
import numpy as np
import io
from pydub import AudioSegment
from qa_txt_voice import generate_answer
from generate_text_en_ru import generate_text
from langdetect import detect, LangDetectException
import re
from pydub.utils import which
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

bot = telebot.TeleBot("7681243585:AAH6RVknILVYnvpft2PZj46vT0jvi6WkG8w")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

en_voice_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
en_voice_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Хранилище состояний пользователей
user_states = {}

@bot.message_handler(commands=['start'])
def send_welcome(message):
    user_lang = message.from_user.language_code
    if user_lang == 'ru':
        welcome_text = (
            f"Здравствуйте, {message.from_user.username}!\n\n"
            "<b>/ask</b> - Задать вопрос.\n\n"
            "<b>/generate</b> - Сгенерирую текст для вас.\n\n"
            "<b>/help</b> — Если вам нужна помощь.")
        bot.send_message(message.chat.id, welcome_text, parse_mode='html')
    else:
        welcome_text = (
            f"Hi, {message.from_user.username}!\n"
            "<b>/ask</b> - Ask a question.\n\n"
            "<b>/generate</b> I will generate text for you.\n\n"
            "<b>/help</b> — If you need help.")
        bot.send_message(message.chat.id, welcome_text, parse_mode='html')

@bot.message_handler(commands=['help'])
def send_help(message):
    if message.from_user.language_code == 'ru':
        bot.reply_to(message, "Используйте: <b>/generate</b> для генерации текста \n\n"
                                   "Используйте: <b>/ask</b> для ответа на ваш вопрос", parse_mode='html')
    else:
        bot.reply_to(message, "Use: <b>/generate</b> to generate text \n\n"
                                   "Use: <b>/ask</b> to answer your question", parse_mode='html')


@bot.message_handler(commands=['generate'])
def handle_generate(message):
    user_id = message.from_user.id
    user_lang = message.from_user.language_code
    user_states[user_id] = 'awaiting_generate'

    if user_lang == 'ru':
        bot.reply_to(message, "Начните предложение для генерации текста:")
    else:
        bot.reply_to(message, "Start a sentence to generate text:")


@bot.message_handler(commands=['ask'])
def handle_ask(message):
    user_id = message.from_user.id
    user_lang = message.from_user.language_code
    user_states[user_id] = 'awaiting_ask'

    if user_lang == 'ru':
        bot.reply_to(message, "Отправьте голосовое сообщение или текстовый вопрос:")
    else:
        bot.reply_to(message, "Send a voice message or write your question:")


@bot.message_handler(content_types=['text'])
def handle_text_input(message):
    user_id = message.from_user.id
    user_state = user_states.get(user_id)
    if user_state == 'awaiting_generate':
        try:
            prompt = message.text.strip()
            response = generate_text(prompt)
            bot.reply_to(message, response)
        except Exception as e:
            bot.reply_to(message, f"Error: {str(e)}")
        finally:
            user_states[user_id] = None  # Сброс состояния

    elif user_state == 'awaiting_ask':
        try:
            answer = generate_answer(message.text)
            bot.reply_to(message, answer)
        except Exception as e:
            bot.reply_to(message, f"Error: {str(e)}")
        finally:
            user_states[user_id] = None
    else:
        user_lang = message.from_user.language_code
        if user_lang == 'ru':
            bot.reply_to(message, "Пожалуйста, используйте команды:\n\n"
                                  "<b>/generate</b> — для генерации текста\n\n"
                                  "<b>/ask</b> — чтобы задать вопрос\n\n"
                                  "<b>/help</b> — помощь", parse_mode='html')
        else:
            bot.reply_to(message, "Please use the commands:\n\n"
                                  "<b>/generate</b> — to generate text\n\n"
                                  "<b>/ask</b> — to ask a question\n\n"
                                  "<b>/help</b> — help", parse_mode='html')

def convert_voice_to_float32(voice_bytes, sampling_rate=16000):
    audio = AudioSegment.from_file(io.BytesIO(voice_bytes), format="ogg")
    audio = audio.set_frame_rate(sampling_rate).set_channels(1).set_sample_width(2)
    raw_data = np.array(audio.get_array_of_samples())
    return raw_data.astype(np.float32) / 32768.0

def detect_lang(prompt):
    try:
        lang = detect(prompt)
    except LangDetectException:
        lang = "unknown"
    if re.search(r'[а-яА-Я]', prompt):
        return "ru"
    return lang if lang in ["ru", "en"] else "en"

@bot.message_handler(content_types=['voice'])
def handle_voice_input(message):
    user_id = message.from_user.id
    user_state = user_states.get(user_id)

    if user_state == 'awaiting_ask':
        try:
            file_info = bot.get_file(message.voice.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            AudioSegment.converter = which("D:/ffmpeg/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe")
            AudioSegment.ffprobe = which("D:/ffmpeg/ffmpeg-7.1.1-essentials_build/bin/ffprobe.exe")
            audio = AudioSegment.from_file(io.BytesIO(downloaded_file), format="ogg")
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            raw_data = np.array(audio.get_array_of_samples())
            audio_np = raw_data.astype(np.float32) / 32768.0
            input_features = en_voice_processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(en_voice_model.device)
            predicted_ids = en_voice_model.generate(input_features, task='transcribe')
            transcribed_text = en_voice_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"Распознанный текст: {transcribed_text}")
            lang = detect_lang(transcribed_text)
            print(f"Определённый язык: {lang}")
            answer = generate_answer(transcribed_text, lang=lang)
            print(f"Ответ: {answer}")
            bot.reply_to(message, answer)
        except Exception as e:
            bot.reply_to(message, f"Error: {str(e)}")
        finally:
            user_states[user_id] = None
    else:
        user_lang = message.from_user.language_code
        if user_lang == 'ru':
            bot.reply_to(message, "Пожалуйста, используйте команды:\n\n"
                                  "<b>/generate</b> — для генерации текста\n\n"
                                  "<b>/ask</b> — чтобы задать вопрос\n\n"
                                  "<b>/help</b> — помощь", parse_mode='html')
        else:
            bot.reply_to(message, "Please use the commands:\n\n"
                                  "<b>/generate</b> — to generate text\n\n"
                                  "<b>/ask</b> — to ask a question\n\n"
                                  "<b>/help</b> — help", parse_mode='html')

@bot.message_handler(func=lambda message: True)
def handle_unknown_message(message):
    user_lang = message.from_user.language_code
    if user_lang == 'ru':
        bot.reply_to(message, "Извините, я не понимаю. Используйте:\n\n"
                              "<b>/generate</b> — для генерации текста\n\n"
                              "<b>/ask</b> — чтобы задать вопрос\n\n"
                              "<b>/help</b> — помощь", parse_mode='html')
    else:
        bot.reply_to(message, "Sorry, I don't understand. Use:\n\n"
                              "<b>/generate</b> — to generate text\n\n"
                              "<b>/ask</b> — to ask a question\n\n"
                              "<b>/help</b> — help", parse_mode='html')

if __name__ == '__main__':
    print("Бот запущен...")
    bot.infinity_polling()