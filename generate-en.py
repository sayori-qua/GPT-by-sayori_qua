from datasets import load_dataset
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import datasets
import os
from joblib import Parallel, delayed
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sacrebleu.metrics import BLEU
from rouge import Rouge
from langdetect import detect
from bs4 import BeautifulSoup
from langdetect.lang_detect_exception import LangDetectException
import gc
from tqdm import tqdm
import time

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# кастомизация параметров
num_of_texts_for_generate_text = 5000
max_new_tokens = 45
max_length = 128
epochs_for_generate = 10
batch_size_for_generate = 16
temperature_for_generate_text = 0.9
top_k_for_generate_text = 45
num_beams = 5
top_p = 0.88
do_sample = True
gradient_accumulation_steps_for_generate = 5
repetition_penalty = 1.3

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

dataset_dir = "D:/datasets"
os.makedirs(dataset_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

en_dataset_path = f"{dataset_dir}/en_dataset_cache"
books_dataset_path = f"{dataset_dir}/books_dataset_cache"

#streaming-режим
en_ful_dataset = load_dataset("openwebtext", split='train', streaming=True, trust_remote_code=True)

books_ful_dataset = load_dataset("bookcorpus", split='train', streaming=True)

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

#преобразование стриминга
en_dataset = list(en_ful_dataset.take(num_of_texts_for_generate_text))
books_dataset = list(books_ful_dataset.take(num_of_texts_for_generate_text))

#сохранение данных локально
datasets.Dataset.from_dict({"text": [item["text"] for item in en_dataset]}).save_to_disk(en_dataset_path)
datasets.Dataset.from_dict({"text": [item["text"] for item in books_dataset]}).save_to_disk(books_dataset_path)

# Загрузка данных из локального кэша
en_dataset = datasets.load_from_disk(en_dataset_path)
books_dataset = datasets.load_from_disk(books_dataset_path)

#предобработка
def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'\W+', ' ', text).lower().strip()
    words = [word for word in text.split()]
    return ' '.join(words)

en_stop_words = set(stopwords.words('english'))
preprocess_en_dataset = Parallel(n_jobs=4)(
    delayed(preprocess_text)(part['text'])
    for part in tqdm(en_dataset, desc = "Preprocessing EN")
)

preprocess_books_dataset = Parallel(n_jobs=4)(
    delayed(preprocess_text)(part['text'])
    for part in tqdm(books_dataset, desc = "Preprocessing EN_Books")
)

en_dataset = preprocess_en_dataset + preprocess_books_dataset

def tokenize_function_for_gen(tokenizer, texts, max_length = max_length):
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        return_attention_mask=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

#en_model(gen_text)
en_df = pd.DataFrame({'text': en_dataset})

en_x_train, en_x_test = train_test_split(en_df['text'].tolist(), test_size = 0.2, random_state = 42)

#тренировка
def train_func(x_train, x_test, model, tokenizer, output_dir, logging_dir, path_of_save_model, path_of_save_tokenizer):
    start_time = time.time()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    x_train_tokenized = tokenize_function_for_gen(tokenizer, x_train)
    x_test_tokenized = tokenize_function_for_gen(tokenizer, x_test)

    train_dataset = Dataset.from_dict({
        'input_ids': x_train_tokenized['input_ids'],
        'attention_mask': x_train_tokenized['attention_mask'],
        'labels': x_train_tokenized['labels']
    })

    eval_dataset = Dataset.from_dict({
        'input_ids': x_test_tokenized['input_ids'],
        'attention_mask': x_test_tokenized['attention_mask'],
        'labels': x_test_tokenized['labels']
    })

    training_args = TrainingArguments(
        output_dir = output_dir,
        logging_dir = logging_dir,
        num_train_epochs = epochs_for_generate,
        per_device_train_batch_size = batch_size_for_generate,
        gradient_accumulation_steps = gradient_accumulation_steps_for_generate,
        save_strategy = "steps",
        logging_steps = 200,
        save_total_limit = 1,
        eval_steps = 20,
        fp16 = True,
        evaluation_strategy= "steps",
        report_to = "tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(path_of_save_model)
    tokenizer.save_pretrained(path_of_save_tokenizer)

    end_time = time.time()
    print(f"Обучение {model} заняло {end_time - start_time:.2f} секунд")

    return model, tokenizer

#дообучение англ модели, task = gen_text
en_model, en_tokenizer = train_func(x_train=en_x_train,
                                    x_test = en_x_test,
                                    model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2').to(device),
                                    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2'),
                                    output_dir="./results/gpt2_final",
                                    logging_dir="./logs/gpt2_final",
                                    path_of_save_model="C:/Users/user/PycharmProjects/GPT_2MODELS/results/gpt2_final",
                                    path_of_save_tokenizer="C:/Users/user/PycharmProjects/GPT_2MODELS/results/gpt2_final",
                                    )
#постобработка
def postprocess(text, prompt):
    text = text.replace(" n t", " not").replace("'m", " am").replace("'ve", " have")
    if text.endswith(' moustache'):
        text = ' '.join(text.split()[:-1])
    prompt = prompt.strip()
    text = text.strip()
    if text == prompt:
        return "I'm not sure how to continue that."
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'
    return text

#генерация
en_model = AutoModelForCausalLM.from_pretrained('C:/Users/user/PycharmProjects/GPT_2MODELS/results/gpt2_final').to(device)

def generate_text(prompt, max_new_tokens = max_new_tokens):
    if not prompt.strip():
        return "Ошибка: пустой запрос."

    lang = detect_lang(prompt)
    if lang == 'en':
        inputs = en_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(en_model.device)
        with torch.no_grad():
            output = en_model.generate(
                **inputs,
                max_new_tokens=96,
                num_beams = 4,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.75,
                repetition_penalty=repetition_penalty,
                top_p=0.95,
                top_k=top_k_for_generate_text,
                do_sample=True,
                pad_token_id=en_tokenizer.pad_token_id,
                eos_token_id=en_tokenizer.eos_token_id
            )

        generated_text = en_tokenizer.decode(output[0], skip_special_tokens=True)
        return postprocess(generated_text, prompt)


prompts_continues = ['I can', 'I wonder', 'He always met her', 'GPT the best in the', 'The forest is',
           'I done', 'So boring', 'my patience', 'a lot of people love cats', 'My hamster',
           'Gym is very good because', 'Gogol is ', 'our dog', 'My computer is faster than',
           'Only she knows', 'Dostoevskiy is ', 'I like apples ', 'She stood by the window, watching the people pass by, the cars, the leaves carried away by the wind',
            'I wonder why', "i will", ""]


for prompt_continue in prompts_continues:
    print(f"Prompt: {prompt_continue}")
    generated_text = generate_text(prompt_continue)
    print(f"Generated text: {generated_text}") #print(f"Generated text: {postprocessing_text(generated_text)}")
    print('-' * 90)
