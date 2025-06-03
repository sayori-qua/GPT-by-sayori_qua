import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langdetect import detect, LangDetectException
import re
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect

# кастомизация параметров
max_new_tokens = 50
epochs = 5
batch_size = 6
temperature = 0.8
top_k = 50
num_beams = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

en_tokenizer = AutoTokenizer.from_pretrained("C:/Users/user/PycharmProjects/GPT_2MODELS/results/gpt2_final")
en_model = AutoModelForCausalLM.from_pretrained("C:/Users/user/PycharmProjects/GPT_2MODELS/results/gpt2_final").to(device)

ru_tokenizer = AutoTokenizer.from_pretrained("C:/Users/user/PycharmProjects/GPT_2MODELS/results/ru_gpt2")
ru_model = AutoModelForCausalLM.from_pretrained("C:/Users/user/PycharmProjects/GPT_2MODELS/results/ru_gpt2").to(device)

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
        if has_rus_word:
            return 'ru'
        else:
            return 'en'
    if lang == 'ru' or has_rus_word:
        return 'ru'
    else:
        return 'en'

#постобработка generate_en
def postprocess_en(text, prompt):
    text = text.replace(" n t", " not").replace("'m", " am").replace("'ve", " have")
    if text.endswith(' moustache'):
        text = ' '.join(text.split()[:-1])
    prompt = prompt.strip()
    text = text.strip()
    if text == prompt:
        return "I'm not sure how to continue that."
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    return text

def postprocess_ru(text, prompt):
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    return text

#генерация
def generate_text(prompt):
    if not prompt.strip():
        return "Ошибка: пустой запрос."

    lang = detect_lang(prompt)
    if lang == 'en':
        inputs = en_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(en_model.device)
        with torch.no_grad():
            output = en_model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams = 4,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=1.25,
                repetition_penalty=1.3,
                top_p=0.95,
                top_k=45,
                do_sample=True,
                pad_token_id=en_tokenizer.pad_token_id,
                eos_token_id=en_tokenizer.eos_token_id
            )

        generated_text = en_tokenizer.decode(output[0], skip_special_tokens=True)
        return postprocess_en(generated_text, prompt)
    elif lang == 'ru':
        inputs = ru_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(ru_model.device)
        with torch.no_grad():
            output = ru_model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=5,
                length_penalty=1.5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=1.5,
                repetition_penalty=1.35,
                top_p=0.85,
                top_k=40,
                do_sample=True,
                pad_token_id=ru_tokenizer.pad_token_id,
                eos_token_id=ru_tokenizer.eos_token_id
            )
        generated_text = ru_tokenizer.decode(output[0], skip_special_tokens=True)
        return postprocess_ru(generated_text, prompt)
