# %% [markdown]
# ## Import Modules

# %%
from glob import glob
import pandas as pd
import numpy as np
import random
import os
import torch
import wandb

import IPython.display as ipd
from scipy.io import wavfile
import noisereduce as nr

from datasets import Dataset, DatasetDict, Audio

from transformers import WhisperTokenizer,  WhisperFeatureExtractor, WhisperProcessor

import re
import librosa
from tqdm.auto import tqdm


# %% [markdown]
# ## Fix Seed

# %%
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
seed_everything()

# %% [markdown]
# ## Config

# %%
CFG = {
    'model': 'openai/whisper-tiny',
    'sr': 16000,
    'noise_file_path': '../files/noise.wav'
}

# %% [markdown]
# ## Read Files

# %%
TRAIN_PATH = '/mnt/elice/dataset/train/'
TEST_PATH = '/mnt/elice/dataset/test/'

# %%
df = pd.read_csv(f'{TRAIN_PATH}/texts.csv', index_col=False)
submission = pd.read_csv(f'sample_submission.csv', index_col=False)


# %% [markdown]
# ## Label Cleaning

# %%
def clean_text(text, remove_space=True):
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]','', text)
    if remove_space:
        text = ''.join(text.split())
    return text


# %%
# label cleaning (remove punctuations)
df['text'] = df['text'].apply(lambda x: clean_text(x, False))

# remove outlier data
df = df[df['filenames'] != 'audio_5497.wav']

if not os.path.exists('preprocess'):
    os.mkdir('preprocess')

df.to_csv('preprocess/clean_df.csv', index=False)


# %% [markdown]
# ## Split long/short dataframe

# %%
def split_dataframe(df, df_name, is_train=True):
    df_long = []
    df_short = []

    for idx, row in tqdm(df.iterrows()):
        if is_train:
            path = TRAIN_PATH + row['filenames']
        else:
            path = row['path']
        wav, fs = librosa.load(path)
        length = len(wav)/fs

        if length >= 30:
            df_long.append(row)
        else:
            df_short.append(row)

    df_long = pd.DataFrame(df_long, columns=df.columns)
    df_short = pd.DataFrame(df_short, columns=df.columns)

    df_long.to_csv(f'preprocess/long_{df_name}.csv', index=False)
    df_short.to_csv(f'preprocess/short_{df_name}.csv', index=False)


# %%
split_dataframe(df, 'df')
split_dataframe(submission, 'test', False)

# %% [markdown]
# ## Data Preprocess & Train Dataset

# %%
# load feature extractor and tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained(CFG['model'])
tokenizer = WhisperTokenizer.from_pretrained(CFG['model'], language="Korean", task="transcribe")

_, noise_array = wavfile.read(CFG["noise_file_path"])


# %%
def prepare_dataset(batch):
    audio = batch['audio']
    reduced_noise_audio = nr.reduce_noise(y=audio['array'], sr=CFG['sr'], y_noise = noise_array)

    # raw form(reduced_noise_audio) -> log-Mel spectrogram
    batch['input_features'] = feature_extractor(reduced_noise_audio, sampling_rate=audio['sampling_rate']).input_features[0]
    
    # target text -> label ids(by tokenizer)
    batch['labels'] = tokenizer(batch['transcripts']).input_ids

    return batch


# %%
def create_train_datasets(df, dir_name='dataset'):
    # create dataset from csv
    ds = Dataset.from_dict({"audio": [f'{TRAIN_PATH}/{file_path}' for file_path in df["filenames"]],
                        "transcripts": [text for text in df["text"]]}).cast_column("audio", Audio(sampling_rate=CFG['sr']))

    # train/valid split
    train_valid = ds.train_test_split(test_size=0.2)
    train_valid_dataset = DatasetDict({
        "train": train_valid["train"],
        "valid": train_valid["test"]})
    
    train_valid_dataset = train_valid_dataset.map(prepare_dataset, remove_columns = train_valid_dataset.column_names['train'], num_proc=4)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
    train_valid_dataset.save_to_disk(dir_name)


# %%
# create_train_datasets(df)

# create long/short train dataset from csv files
# long_df = pd.read_csv('preprocess/long_df.csv', index_col=False)
short_df = pd.read_csv('preprocess/short_df.csv', index_col=False)

# create_train_datasets(long_df, dir_name='dataset_long')
create_train_datasets(short_df, dir_name='dataset_short')


# %% [markdown]
# ## Test Dataset

# %%
def prepare_test_dataset(batch):
    audio = batch['audio']
    reduced_noise_audio = nr.reduce_noise(y=audio['array'], sr=CFG['sr'], y_noise = noise_array)

    # raw form(reduced_noise_audio) -> log-Mel spectrogram
    batch['input_features'] = feature_extractor(reduced_noise_audio, sampling_rate=audio['sampling_rate']).input_features[0]

    return batch


# %%
def create_test_dataset(df, dir_name='dataset_test'):
    # create dataset from csv
    test_dataset = Dataset.from_dict({"audio": [file_path for file_path in df["path"]]})
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=CFG['sr']))
    sampling_rate = test_dataset.features['audio'].sampling_rate

    # test data preprocess
    test_dataset = test_dataset.map(prepare_test_dataset, remove_columns = test_dataset.column_names, num_proc=4)

    # save test dataset
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        
    test_dataset.save_to_disk(dir_name)


# %%
create_test_dataset(submission)

# create long/short test dataset from csv files
# long_test = pd.read_csv('preprocess/long_test.csv', index_col=False)
# short_test = pd.read_csv('preprocess/short_test.csv', index_col=False)

# create_test_dataset(long_test, dir_name='dataset_long_test')
# create_test_dataset(short_test, dir_name='dataset_short_test')

# %% [markdown]
# ## Import Modules

# %%
from glob import glob
import pandas as pd
import numpy as np
import random
import os
import torch
import wandb

import IPython.display as ipd
from scipy.io import wavfile

from datasets import load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperTokenizer,  WhisperFeatureExtractor, WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import Seq2SeqTrainer

import evaluate
import re


# %% [markdown]
# ## Fix Seed

# %%
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
seed_everything()

# %% [markdown]
# ## Config

# %%
TRAIN_PATH = '/mnt/elice/dataset/train/'
TEST_PATH = '/mnt/elice/dataset/test/'

# %%
CFG = {
    'model': 'openai/whisper-tiny',
    'sr': 16000,
}

# %% [markdown]
# ## Training

# %%
# Load train dataset created by preprocess notebook
train_valid_dataset = load_from_disk('dataset_short')


# %% [markdown]
# ### Data Collator

# %%
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
        # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Tokenize된 레이블 시퀀스를 가져온다.
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
        # 해당 토큰은 이후 언제든 추가할 수 있다.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# %%
# processor, tokenizer, feature extractor 로드
processor = WhisperProcessor.from_pretrained(CFG['model'], language="Korean", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(CFG['model'], language="Korean", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(CFG['model'])

# %%
# 데이터 콜레이터 초기화
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %% [markdown]
# ### Evaluation Metrics

# %%
metric = evaluate.load('cer')


# %%
def clean_text(text, remove_space=True):
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]','', text)
    if remove_space:
        text = ''.join(text.split())
    return text


# %%
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pad_token을 -100으로 치환
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # metrics 계산 시 special token들을 빼고 계산하도록 설정
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 문장부호 + 띄어쓰기 제거하고 계산
    pred_str = [clean_text(text) for text in pred_str]
    label_str = [clean_text(text) for text in label_str]

    cer = metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}


# %% [markdown]
# ### Load Pretrained Checkpoint

# %%
model = WhisperForConditionalGeneration.from_pretrained(CFG['model'])

# %%
# restrict prediction language to korean
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='korean', task='transcribe')
model.config.suppress_tokens = []

# %% [markdown]
# ### Training

# %%
training_args = Seq2SeqTrainingArguments(
    torch_compile=True, # for optimize code

    output_dir="checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=10,
    optimizers=(AdamW, get_cosine_schedule_with_warmup),

    fp16=True,

    evaluation_strategy="steps",
    eval_steps=560,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    dataloader_num_workers=8,
    
    save_strategy='steps',
    save_steps=560,
    logging_steps=25,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

# %%
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_valid_dataset["train"],
    eval_dataset=train_valid_dataset["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# %%
trainer.train()

# %%
## save checkpoint
MODEL_PATH = 'model'
trainer.save_model(MODEL_PATH)

# %%
# set wandb to save all codes, weights, and results
wandb.save("*.py")
wandb.save("*.ipynb")

wandb.save("*.pt")
wandb.save("*.pth")
wandb.save("*.hdf5")

wandb.save("*.csv")

# %% [markdown]
# ## Predict

# %%
# restrict prediction language to korean
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='korean', task='transcribe')
model.config.suppress_tokens = []

# %%
# load test dataset created by preprocess notebook
test_dataset = load_from_disk('dataset_test')
len(test_dataset)

# %%
test_args = Seq2SeqTrainingArguments(
    torch_compile=True, # for optimize code

    output_dir="repo_name",

    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=225,
    dataloader_num_workers=8,
    
    logging_steps=25,
    # report_to=["wandb"],
)

# %%
# inference
test_trainer = Seq2SeqTrainer(args=test_args,
                              model=model)
pred = test_trainer.predict(test_dataset)

# %%
text = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)

# %% [markdown]
# ## Submit

# %%
submission = pd.read_csv('sample_submission.csv')
submission['text'] = text
submission.to_csv('raw_submission.csv', index=False)

# %% [markdown]
# ## Install Modules

# %%
# # %pip install evaluate
# # %pip install swifter

# %% [markdown]
# ## Import Modules

# %%
import pandas as pd
import numpy as np
import os
import random
import evaluate
from tqdm import tqdm
import re
import swifter

# %% [markdown]
# ## Load Raw Submission

# %%
TRAIN_PATH = '/mnt/elice/dataset/train/'
TEST_PATH = '/mnt/elice/dataset/test/'

# %%
df = pd.read_csv(TRAIN_PATH + 'texts.csv')
raw_submission = pd.read_csv('raw_submission.csv')


# %% [markdown]
# ## Remove Duplicates

# %%
def remove_duplicates(s):
    l = s.split(" ")
    while len(l) >= 2 and l[-1] == l[-2]:
        l = l[:-1]
    return " ".join(l)


# %%
raw_submission["text"] = raw_submission["text"].apply(remove_duplicates)


# %% [markdown]
# ## Proofreading with train labels

# %%
def clean_text(text, remove_space=True):
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]','', text)
    if remove_space:
        text = ''.join(text.split())
    return text


# %%
targets = list(map(lambda x : clean_text(x), set(df['text'].tolist()))) # train label값
threshold = 41 # text길이
cer_threshold = 0.5
metric = evaluate.load('cer')

def get_close(pred):
    if len(pred) >= threshold:
        return pred
    min_cer = 1e8
    index = -1
    for j in range(len(targets)):
        target = targets[j]
        if len(target) >= threshold:
            continue
        if len(target) > len(pred) * 2:
            continue
        if len(pred) > len(target) * 1.5:
            continue 
        cnt = 0
        for ch in target:
            if ch in pred:
                cnt += 1
        if cnt * 2 < len(target):
            continue
        cer = metric.compute(predictions=[pred], references=[target])
        if min_cer > cer:
            min_cer = cer
            index = j
    if min_cer < cer_threshold:
        pred = targets[index]
    return pred


# %%
targets = list(map(lambda x : clean_text(x), set(df['text'].tolist()))) #train label값
preds = raw_submission['text'].apply(lambda x : clean_text(x))# predict 값

tqdm.pandas()
preds = preds.swifter.progress_bar(True).allow_dask_on_strings(enable=True).apply(get_close)

# %% [markdown]
# ## Export processed Submission

# %%
submission = raw_submission
submission['text'] = preds
submission.to_csv('submission.csv', index=False)
