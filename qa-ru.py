import random
import re
import os
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import NllbTokenizer
from transformers import M2M100ForConditionalGeneration
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

src_lang = "rus_Cyrl"
tgt_lang = "eng_Latn"
tgt_lang_back = "rus_Cyrl"

translator_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_safetensors=False).to(device)
print("Доступные языки ")

print(translator_tokenizer.lang_code_to_id.keys())
en_qa_model_path = "C:/Users/user/PycharmProjects/GPT_2MODELS/results/en_qa"
en_qa_model = AutoModelForSeq2SeqLM.from_pretrained(en_qa_model_path).to(device)
en_qa_tokenizer = AutoTokenizer.from_pretrained(en_qa_model_path)


def translate(text, src_lang, tgt_lang):
    translator_tokenizer.src_lang = src_lang
    inputs = translator_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    generated_tokens = translator_model.generate(
        **inputs,
        forced_bos_token_id=translator_tokenizer.lang_code_to_id[tgt_lang],
        max_length=256,
    )
    return translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def detect_lang(prompt):
    flag_words = {'где', 'когда', 'почему', 'зачем', 'как', 'какой', 'он', 'хочу', 'сделай',
                  'она', 'ты', 'ответь', 'спроси', 'оно', 'хотел', 'что', 'такой', 'можно',
                  'объясни', 'найди', 'помоги', 'исправь', 'подскажи', 'перепиши', 'есть', 'кто', 'какая'}
    prompt_lower = prompt.lower()
    words = prompt_lower.split()
    has_rus_word = any(word in flag_words for word in words)
    try:
        lang = detect(prompt)
    except LangDetectException:
        return 'ru' if has_rus_word else 'en'
    return 'ru' if lang == 'ru' or has_rus_word else 'en'

def clean_response(text):
    text = re.sub(r'Вопрос:', "", text, flags=re.IGNORECASE)
    text = re.sub(r'Ответ:', "" , text, flags=re.IGNORECASE)
    if not text:
        return "Я не могу ответить на этот вопрос."
    text = text[0].upper() + text[1:]
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'
    return text

def generate_text(prompt):
    if not prompt.strip():
        return "Error: пустой запрос."
    lang = detect_lang(prompt)
    if lang == 'ru':
        text_en = translate(prompt, src_lang, tgt_lang)
        inputs = en_qa_tokenizer(text_en, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = en_qa_model.generate(
                **inputs,
                max_new_tokens=96,
                length_penalty=1.0,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=1.2, #0.8
                repetition_penalty=1.3,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=en_qa_tokenizer.pad_token_id,
                eos_token_id=en_qa_tokenizer.eos_token_id,
                bad_words_ids=[[en_qa_tokenizer.unk_token_id]],
            )

        response = en_qa_tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response_ru = translate(response, tgt_lang, tgt_lang_back)
        cleaned_response_ru = clean_response(response_ru)
        return cleaned_response_ru
    else:
        return "Вопрос должен быть на русском."

prompts_qa_ru = ["Можно ли действительно доверять своей памяти?", "Программирование полезно для мозга?",
    "Что происходит после того, как мы уходим отсюда?", "Правда ли, что тебя создал человек?",
    "Почему он всегда встречается с ней ночью?", "Где ты родился?", "Самая быстрая машина в мире - это?",
    "Самый ли GPT лучший в понимании людей?", "Кто живет в лесу?", "Кто такой Достоевский?",
    "Что живет глубоко в лесу?", "Сколько сейчас времени?", "Что такое интернет?",
    "Сделал ли я достаточно?", "можешь ли ты говорить по-русски?",
    "Почему всё стало таким скучным в последнее время?", "Кто был со мной вместе вчера?"
    "Как долго может длиться мое терпение?", "Исскуственный интеллект - это?",
    "Почему так много людей любят кошек больше, чем людей?", "Где ты учишься?",
    "Как твои дела?", "Математика или Биология?", "Что ты сейчас делаешь?",
    "У тебя есть чувства?", "Ты завтра идешь со мной в университет?", "Можешь ли ты создать аватар для моего бота?"
]

for prompt_qa in prompts_qa_ru:
    print(f"Вопрос: {prompt_qa}")
    answer = generate_text(f"{prompt_qa}")
    print(f"Ответ: {answer}")
    print("-" * 100)