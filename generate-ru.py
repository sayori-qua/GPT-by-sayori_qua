from datasets import load_dataset
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sympy.core.random import shuffle
from torchgen.executorch.api.et_cpp import return_names
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
from nltk import sent_tokenize

from accelerate import Accelerator
from transformers import AutoModel, AutoConfig

from itertools import islice
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# кастомизация параметров
num_of_texts_for_generate_text = 2800
max_new_tokens = 50
max_length = 86
epochs_for_generate = 4
batch_size_for_generate = 16
temperature_for_generate_text = 0.7
top_k_for_generate_text = 50
num_beams = 5
top_p = 0.9
do_sample = True
gradient_accumulation_steps_for_generate = 5
repetition_penalty = 1.3

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

dataset_dir = "D:/datasets"
os.makedirs(dataset_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

ru_dataset_path = f"{dataset_dir}/ru_dataset_cache"

#streaming-режим
ru_ful_dataset = load_dataset("DataSynGen/RUwiki", split='train', streaming=True, trust_remote_code=True)

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
ru_dataset = list(ru_ful_dataset.take(num_of_texts_for_generate_text))

#сохранение данных локально
datasets.Dataset.from_dict({"text": [item["text"] for item in ru_dataset]}).save_to_disk(ru_dataset_path)

# Загрузка данных из локального кэша
ru_dataset = datasets.load_from_disk(ru_dataset_path)

#предобработка
def preprocess_text(text):
    # Чистка HTML, ссылок, спецсимволов
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s.,!?—–\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

preprocess_ru_dataset = Parallel(n_jobs=4)(
    delayed(preprocess_text)(part['text'])
    for part in tqdm(ru_dataset, desc = "Preprocessing RU")
)
def tokenize_function_for_gen(tokenizer, texts, max_length=max_length):
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    labels = tokenizer(
        [text + tokenizer.eos_token for text in texts],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )["input_ids"]
    inputs["labels"] = labels
    return inputs

"""def tokenize_function_for_gen(tokenizer, texts, max_length = max_length):
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        return_attention_mask=True
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized"""

#ru_model(gen_text)
ru_df = pd.DataFrame({'text': preprocess_ru_dataset})
ru_x_train, ru_x_test = train_test_split(ru_df['text'].tolist(), test_size = 0.2, random_state = 42)

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
        eval_steps = 15,
        fp16 = True,
        evaluation_strategy = 'steps',
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


#дообучение рус модели, task = gen_text
ru_model, ru_tokenizer = train_func(x_train=ru_x_train,
                                    x_test = ru_x_test,
                                    model = AutoModelForCausalLM.from_pretrained('IlyaGusev/rugpt3medium_sum_gazeta').to(device),
                                    tokenizer = AutoTokenizer.from_pretrained('IlyaGusev/rugpt3medium_sum_gazeta'),
                                    output_dir="./results/ru_gpt",
                                    logging_dir="./logs/ru_gpt",
                                    path_of_save_model="C:/Users/user/PycharmProjects/GPT_2MODELS/results/ru_gpt2",
                                    path_of_save_tokenizer="C:/Users/user/PycharmProjects/GPT_2MODELS/results/ru_gpt2",
                                    )
#генерация
model_path = "C:/Users/user/PycharmProjects/GPT_2MODELS/results/ru_gpt2"
config = AutoConfig.from_pretrained(model_path)
ru_model = AutoModelForCausalLM.from_pretrained(model_path, config=config).to(device)

def generate_text(prompt):
    if not prompt.strip():
        return "Ошибка: пустой запрос."

    lang = detect_lang(prompt)
    if lang == 'ru':
        inputs = ru_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = ru_model.generate(
                **inputs,
                max_new_tokens=48,
                num_beams=5,
                length_penalty=1.5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.7,
                repetition_penalty=1.35,
                top_p=0.85,
                top_k=40,
                do_sample=True,
                pad_token_id=ru_tokenizer.pad_token_id,
                eos_token_id=ru_tokenizer.eos_token_id
            )
        return ru_tokenizer.decode(output[0], skip_special_tokens=True)

prompts_continues = ['Я люблю гулять по вечерам',
           'Искусственный интеллект - это','Боря завтра идет в школу, а я',
           'Спортзал помогает мне в', 'Новая футболка просто',
           'Она встала у окна, глядя на проходящих людей, машины, листья, которые ветер',
            'Я хочу', "Я завтра пойду", "Я люблю летать, а еще"]

for prompt_continue in prompts_continues:
    print(f"Prompt: {prompt_continue}")
    generated_text = generate_text(prompt_continue)
    print(f"Generated text: {generated_text}") #print(f"Generated text: {postprocessing_text(generated_text)}")
    print('-' * 90)


