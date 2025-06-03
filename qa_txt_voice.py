from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
import re
import random
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect
from scipy.io import wavfile
import numpy as np
import io
from pydub import AudioSegment
import tempfile
import os
from pydub.utils import which
from transformers import NllbTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

en_voice_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
en_voice_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

en_qa_model = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/user/PycharmProjects/GPT_2MODELS/results/en_qa").to(device)
en_qa_tokenizer = AutoTokenizer.from_pretrained("C:/Users/user/PycharmProjects/GPT_2MODELS/results/en_qa")

translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
translator_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def translate(text, src_lang, tgt_lang):
    translator_tokenizer.src_lang = src_lang
    inputs = translator_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    generated_tokens = translator_model.generate(
        **inputs,
        forced_bos_token_id=translator_tokenizer.lang_code_to_id[tgt_lang],
        max_length=256,
    )
    return translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


#входные фразы для лучшего отображения
intro_phrases = [
    "I think...",
    "Hmm...",
    "Well...",
    "Let me see...",
    "Actually...",
    "So...",
    "Okay...",
]

#определение языка
def detect_lang(prompt):
    try:
        lang = detect(prompt)
    except LangDetectException:
        lang = "unknown"

    # Примитивная проверка на русские буквы
    if re.search(r'[а-яА-Я]', prompt):
        return "ru"
    return lang if lang in ["ru", "en"] else "en"

def clean_response(text):
    text = re.sub(r'Вопрос:', "", text, flags=re.IGNORECASE)
    text = re.sub(r'Ответ:', "" , text, flags=re.IGNORECASE)
    if not text:
        return "Я не могу ответить на этот вопрос."
    text = text[0].upper() + text[1:]
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'
    return text

#постпроцессинг ответа для английского
def clean_response_en(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\"\']', '', text).strip()
    intro = random.choice(intro_phrases)
    text = re.sub(r'answer:?\.?', intro, text, flags=re.IGNORECASE)
    text = re.sub(r'question:?\.?', intro, text, flags=re.IGNORECASE)
    if len(text) == 0:
        return "I can't answer your question."
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'
    return text

def preprocess_prompt(prompt):
    prompt = prompt.strip()
    if not prompt.endswith('?'):
        prompt = prompt + "?"
    prompt = 'Question: ' + prompt
    return prompt

#генерация через текстовый промпт
def generate_answer(prompt, lang = None):
    if not prompt.strip():
        return "Error: empty request."
    if lang is None:
        lang = detect_lang(prompt)
    if lang == 'en':
        prompt = preprocess_prompt(prompt)
        inputs = en_qa_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = en_qa_model.generate(
                **inputs,
                max_new_tokens=102,
                length_penalty=1.0, #штраф за длину последовательности
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=1.2,
                repetition_penalty=1.3,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=en_qa_tokenizer.pad_token_id,
                eos_token_id=en_qa_tokenizer.eos_token_id
            )
        response = en_qa_tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned_response = clean_response_en(response)
        return cleaned_response

    if lang == 'ru':
        text_en = translate(prompt, src_lang="rus_Cyrl", tgt_lang="eng_Latn")
        text_en = preprocess_prompt(text_en)
        inputs = en_qa_tokenizer(text_en, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = en_qa_model.generate(
                **inputs,
                max_new_tokens=102,
                length_penalty=1.0,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=1.2,
                repetition_penalty=1.3,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=en_qa_tokenizer.pad_token_id,
                eos_token_id=en_qa_tokenizer.eos_token_id,
                bad_words_ids=[[en_qa_tokenizer.unk_token_id]],
            )

        response = en_qa_tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response_ru = translate(response, src_lang="eng_Latn", tgt_lang = "rus_Cyrl")
        cleaned_response_ru = clean_response(response_ru)
        return cleaned_response_ru