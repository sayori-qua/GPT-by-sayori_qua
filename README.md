<div align="center">
  <h3>Written by @sayori_qua</h3>
</div>

# English Text Generation with Fine-Tuned GPT-2
This project demonstrates how to fine-tune a GPT-2 model for English text generation using datasets such as OpenWebText and BookCorpus. The model is trained to continue prompts in a natural and coherent way.

📝 Overview
The main goal of this project is to:

Load and preprocess large-scale datasets (OpenWebText and BookCorpus).
Fine-tune a GPT-2 model for causal language modeling.
Generate high-quality English text continuations based on input prompts.\

🧰 Requirements
Make sure you have the following installed:
```bash
pip install torch transformers datasets nltk bs4 langdetect rouge sacrebleu pandas tqdm joblib tensorboard
```
Also, download required NLTK resources:
```bash
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
🧪 Prompts for Testing
```bash
prompts_continues = [
    'I can', 
    'I wonder', 
    'He always met her', 
    'Gym is very good because',
    'She stood by the window watching...',
    ...
]
```
Training Process
Tokenizes and pre-processes data (HTML cleaning, URL removal, stopwords filtering).
Splits data into train/test sets.
Trains the model using Hugging Face's Trainer.
Saves the best checkpoint locally.

🧠 Text Generation
After training, the model generates text continuations using advanced decoding strategies like beam search, top-k sampling, and repetition penalty.

Example output:
"""Вставить картинку"""


# Russian Text Generation with Fine-Tuned GPT Model
This project demonstrates how to fine-tune a Russian GPT model (IlyaGusev/rugpt3medium_sum_gazeta) on the RUwiki dataset for Russian text generation. After training, the model is capable of continuing prompts in natural and coherent Russian.

🧰 Requirements
```bash
pip install torch transformers datasets nltk bs4 langdetect rouge sacrebleu pandas tqdm joblib tensorboard accelerate
```
Also, download required NLTK resources:
```bash
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
🧪 Prompts for Testing
```bash
Some example prompts used for testing the model:
prompts_continues = [
    'Я люблю гулять по вечерам',
    'Искусственный интеллект - это',
    'Боря завтра идет в школу, а я',
    'Она встала у окна, глядя на проходящих людей...',
    ...
]
```
Training Process
The pipeline includes:

Loading and preprocessing Russian texts (HTML cleaning, URL removal, punctuation handling).
Splitting data into train/test sets.
Tokenizing using the Russian GPT tokenizer.
Training the model using Hugging Face's Trainer.
Saving the best checkpoint locally.

🧠 Text Generation
After training, the model generates continuations using advanced decoding strategies such as beam search, top-k sampling, and repetition penalty.

Example output:
"""Вставить картинку"""

English QA Model Training with FLAN-T5
This project demonstrates how to fine-tune a question answering model using the google/flan-t5-base model on the OpenAssistant/oasst1 dataset. The model is trained to answer user questions in a natural and helpful way.

🧰 Requirements
Make sure you have all required libraries installed:
```bash
pip install torch transformers datasets nltk bs4 langdetect rouge sacrebleu pandas tqdm joblib tensorboard accelerate
```
Also, download NLTK resources:
```bash
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
🗂️ Dataset Used
OpenAssistant/oasst1 — A multilingual dataset of human-AI conversations.
Only high-quality English question-answer pairs are selected based on metadata labels such as:
Helpfulness
Toxicity
Quality
Creativity

🧪 Example Prompts
Some example prompts used for testing:
```bash
prompts_qa_en = [
    "Can you really trust your memory?",
    "Is programming good for the brain?",
    "What happens after we leave this place?",
    "Is it true that a human created you?",
    ...
]
```
You can easily extend or modify these prompts.

Training Pipeline
The pipeline includes:

Loading and filtering the dataset.
Preprocessing text (HTML cleaning, URL removal).
Selecting only high-quality English QA pairs.
Tokenizing input/output texts.
Training the T5 model using Hugging Face's Trainer.
Saving the best checkpoint locally.

🧠 Text Generation
After training, the model generates answers using decoding strategies like:

Beam search
Top-p and Top-k sampling
Repetition penalty

Example output:
"""Вставить картинку"""

Multilingual QA System with Translation
This project demonstrates how to build a multilingual question answering system , where a Russian question is translated into English, answered by an English QA model (google/flan-t5-base), and the answer is translated back into Russian.

The full pipeline includes:

Language detection
Text translation using facebook/nllb-200-distilled-600M
QA generation using a fine-tuned FLAN-T5 model

🧰 Requirements
Make sure you have the following libraries installed:
```bash
pip install torch transformers datasets nltk langdetect accelerate
```
Also, download NLTK resources:
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

🌍 Supported Languages
ranslation is powered by NLLB-200 tokenizer. You can see all supported languages via:
```bash
translator_tokenizer.lang_code_to_id.keys()
```
Example: 'rus_Cyrl', 'eng_Latn', 'deu_Latn', 'fra_Latn', etc.

🧪 Example Prompts
Some example prompts used for testing:
```bash
prompts_qa_ru = [
    "Можно ли действительно доверять своей памяти?",
    "Программирование полезно для мозга?",
    "Что происходит после того, как мы уходим отсюда?",
    "Правда ли, что тебя создал человек?",
    ...
]
```

🔁 Translation Pipeline
The flow looks like this:

Input : Russian question.
Language Detection : Detects if input is in Russian or English.
Translate to English : Using NLLB model.
Generate Answer : Using fine-tuned English QA model.
Translate Back to Russian : Final response in native language.

🧠 Sample Output
"Вставить картинку вопрос-ответ"

# Multilingual Text Generation with Fine-Tuned GPT Models
This project demonstrates how to use two fine-tuned GPT models for language-specific text generation :

English model trained on OpenWebText and BookCorpus.
Russian model trained on RUwiki.
The system automatically detects the input language and uses the appropriate model to generate natural-sounding continuations.

🧰 Requirements
Make sure you have the following libraries installed:
```bash
pip install torch transformers langdetect
```

No additional NLTK or spaCy dependencies are required for this module.

🔍 Language Detection
The system uses langdetect + custom keyword rules to determine the input language.

Supported languages:

Russian (ru)
English (en)
Custom keywords ensure better accuracy in cases where language detection fails.

🧠 Text Generation
Each model is used to continue the user's prompt in its native language:

English Model
Model: fine-tuned "openai-community/gpt2"
Postprocessing: Fixes common contractions and capitalization.
Example output:
"""Вставить картинку"""

Russian Model
Model: IlyaGusev/rugpt3medium_sum_gazeta
Postprocessing: Ensures proper capitalization and punctuation.
Example output:
"""Вставить картинку"""
