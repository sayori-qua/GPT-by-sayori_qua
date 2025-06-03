import random
from datasets import load_dataset
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sympy.core.random import shuffle
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import pymorphy2
from transformers import Trainer, TrainingArguments
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
from accelerate import Accelerator
from torch.utils.data import IterableDataset

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# кастомизация параметров
max_samples_en = 14000
max_new_tokens = 60
max_length = 64
epochs = 7
batch_size = 4
temperature_for_generate_text = 0.9
temperature_for_qa = 0.8
top_k_for_generate_text = 50
top_k_for_qa = 30
num_beams = 5
top_p=0.9
do_sample=True
gradient_accumulation_steps = 8
repetition_penalty = 1.3

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

dataset_dir = "D:/datasets"
os.makedirs(dataset_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'\W+', ' ', text).lower().strip()
    words = [word for word in text.split()]
    return ' '.join(words)

dataset_en = load_dataset("OpenAssistant/oasst1", split='train', streaming=True)

qa_pairs_en = []
id_to_msg_en = {}

def get_label_value(labels, label_name):
    if not isinstance(labels, dict):
        return 0
    name_list = labels.get("name")
    value_list = labels.get("value")
    if not isinstance(name_list, list) or not isinstance(value_list, list):
        return 0
    try:
        idx = name_list.index(label_name)
        return value_list[idx]
    except (ValueError, IndexError, KeyError, TypeError):
        return 0

for ex in dataset_en:
    message_id = ex.get("message_id")
    if message_id:
        id_to_msg_en[message_id] = ex
    if len(id_to_msg_en) > max_samples_en * 10:
        break

for ex in id_to_msg_en.values():
    if ex.get("role") != "assistant":
        continue

    parent_id = ex.get("parent_id")
    parent = id_to_msg_en.get(parent_id)

    if not parent or parent.get("role") != "prompter":
        continue

    input_text = parent.get("text", "").strip()
    output_text = ex.get("text", "").strip()

    if not input_text or not output_text:
        continue
    question_is_en = (parent.get("lang") == "en")
    answer_is_en = (ex.get("lang") == "en")

    if not (question_is_en and answer_is_en):
        continue

    labels = ex.get("labels", {})
    creativity_val = get_label_value(labels, "creativity")
    help_val = get_label_value(labels, "helpfulness")
    tox_val = get_label_value(labels, "toxicity")
    qual_val = get_label_value(labels, "quality")

    if help_val > 0.7 and tox_val < 0.2 and qual_val > 0.7 and creativity_val > 0.45:
        qa_pairs_en.append({
            "input": f"Question: {input_text}",
            "output": f"Answer: {output_text}"
        })

    if len(qa_pairs_en) >= max_samples_en:
        break

my_qa_examples = [
    {
        "input": "Question: How are you?",
        "output": "Answer: I'm fine."
    },
    {
        "input": "Question: What are you doing'?",
        "output": "Answer: I am chatting with you."
    },
    {
        "input": "Question: Do you have feelings?",
        "output": "Answer: I don’t have emotions, but I can simulate friendly conversation."
    },
    {
        "input": "Question: Can you write a poem??",
        "output": "Answer: Sure! Roses are red, Violets are blue, I’m here to help, And so I do."
    },
    {
        "input": "Question: Why are you here?",
        "output": "Answer: I'm here to assist with questions and provide information."
    },
    {
        "input": "Question: Are you human?",
        "output": "Answer: No, I'm an AI language model designed to chat and assist with text generation."
    },
    {
        "input": "Question: Is this real life?",
        "output": "Answer: This is a simulated conversation, but it's based on real language patterns."
    }
]

qa_pairs_en.extend(my_qa_examples)
df_en = pd.DataFrame(qa_pairs_en)

def is_valid_example(row):
    min_input_len = 10
    min_output_len = 10
    return len(row['input']) >= min_input_len and len(row['output']) >= min_output_len

df_en = df_en[df_en.apply(is_valid_example, axis=1)]

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def tokenize(example):
    model_inputs = tokenizer(example["input"], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(example["output"], max_length=128, truncation=True, padding="max_length")
    labels["input_ids"] = [label if label != tokenizer.pad_token_id else -100 for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#en_qa_model(qa)
train_df_en, test_df_en = train_test_split(df_en, test_size=0.2, random_state=42)

#тренировка
def train_func(x_train, x_test, model, tokenizer, output_dir, logging_dir, path_of_save_model, path_of_save_tokenizer):
    start_time = time.time()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = Dataset.from_pandas(x_train).map(tokenize, batched=True)
    eval_dataset = Dataset.from_pandas(x_test).map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        save_strategy="steps",
        logging_steps=100,
        save_total_limit = 1,
        eval_steps=40,
        fp16=False,
        evaluation_strategy='steps',
        report_to="tensorboard",
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

#дообучение англ-qa модели, task = qa
en_qa_model, en_qa_tokenizer = train_func(x_train=train_df_en,
                                          x_test = test_df_en,
                                          model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to(device),
                                          tokenizer =  AutoTokenizer.from_pretrained('google/flan-t5-base'),
                                          output_dir="./results/en_qa",
                                          logging_dir="./logs/en_qa",
                                          path_of_save_model="C:/Users/user/PycharmProjects/GPT_2MODELS/results/en_qa",
                                          path_of_save_tokenizer="C:/Users/user/PycharmProjects/GPT_2MODELS/results/en_qa",
                                          )

intro_phrases = [
    "I think...",
    "Hmm...",
    "Well...",
    "Interesting question...",
    "Let me see...",
    "Actually...",
    "So...",
    "Okay...",
]

en_qa_model = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/user/PycharmProjects/GPT_2MODELS/results/en_qa").to(device)

def clean_response(text):
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
    prompt = 'Question: ' + prompt
    return prompt

def generate_text(prompt):
    if not prompt.strip():
        return "Error: empty request."

    lang = detect_lang(prompt)
    prompt = preprocess_prompt(prompt)
    if lang == 'en':
        inputs = en_qa_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(en_qa_model.device)
        with torch.no_grad():
            output = en_qa_model.generate(
                **inputs,
                max_new_tokens=96,
                length_penalty=1.0, #штраф за длину последвоательности
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=1.2,
                do_sample = True,
                repetition_penalty=1.3,
                top_p=0.95,
                top_k=50,
                pad_token_id=en_qa_tokenizer.pad_token_id,
                eos_token_id=en_qa_tokenizer.eos_token_id
            )

        response = en_qa_tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned_response = clean_response(response)
        return cleaned_response

prompts_qa_en = [
    "Can you really trust your memory?",
    "Is programming good for the brain?",
    "What happens after we leave this place?",
    "Is it true that a human created you?",
    "Why does he always meet her at night?",
    "Where were you born?",
    "What is the fastest car in the world?",
    "Is GPT really the best at understanding people?",
    "Who lives in the forest?",
    "Who is Dostoevsky?",
    "What lives deep in the forest?",
    "What time is it now?",
    "What is the internet?",
    "Did I do enough?",
    "Can you speak Russian?",
    "Why has everything become so boring lately?",
    "Who was with me yesterday?",
    "How long can my patience last?",
    "What is artificial intelligence?",
    "Why do so many people love cats more than people?",
    "Whe do you study?",
    "How are you doing?",
    "Math or Biology?",
    "What are you doing right now?",
    "Do you have feelings?",
    "Are you coming with me to university tomorrow?",
    "Can you create an avatar for my bot?"
]
for prompt_qa in prompts_qa_en:
    print(f"Вопрос: {prompt_qa}")
    answer = generate_text(f"{prompt_qa}")
    print(f"Ответ: {answer}")
    print("-" * 100)

