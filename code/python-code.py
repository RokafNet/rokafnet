
# ## Import Modules

# %%
import pandas as pd
from glob import glob
import IPython.display as ipd

# %%
import numpy as np
import random
import os
import torch

from scipy.io import wavfile

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
    'model': 'seastar105/whisper-small-ko-zeroth',
    'sr': 16000,
}

# %% [markdown]
# preprocess - read files 삽입 자리

# %% [markdown]
# preprocess - data cleaning 삽입 자리

# %% [markdown]
# preprocess - Data Preprocess & Train Dataset 삽입 자리

# %% [markdown]
# preprocess - Test Dataset 삽입 자리

# %% [markdown]
# ## Training

# %%
from datasets import load_from_disk
train_valid_dataset = load_from_disk('dataset_short')

# %% [markdown]
# ### Data Collator

# %%
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

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
from transformers import WhisperTokenizer,  WhisperFeatureExtractor
from transformers import WhisperProcessor

# 훈련시킬 모델의 processor, tokenizer, feature extractor 로드
processor = WhisperProcessor.from_pretrained(CFG['model'], language="Korean", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(CFG['model'], language="Korean", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(CFG['model'])

# %%
# 데이터 콜레이터 초기화
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %% [markdown]
# ### Evaluation Metrics

# %%
import evaluate

metric = evaluate.load('cer')

# %%
import re

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
# ### Pretrained Checkpoint

# %%
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(CFG['model'])

# %%
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='korean', task='transcribe')
model.config.suppress_tokens = []

# %% [markdown]
# ### Training

# %%
from transformers import Seq2SeqTrainingArguments
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup

training_args = Seq2SeqTrainingArguments(
    torch_compile=True, # for optimize code

    output_dir="checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=15,

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
from transformers import Seq2SeqTrainer

optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer = optimizer,
                                            num_warmup_steps=500,
                                            num_cycles = 2,
                                            num_training_steps=6800)
optimizers = (optimizer, scheduler)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_valid_dataset["train"],
    eval_dataset=train_valid_dataset["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    optimizers=optimizers,
)

# %%
import torch._dynamo
torch._dynamo.config.suppress_errors = True

trainer.train()

# %%
## save checkpoint
MODEL_PATH = 'model'
trainer.save_model(MODEL_PATH)

# %% [markdown]
# !pip install wandb

# %%
import wandb

# set wandb to save all codes, weights, and results


# %% [markdown]
# ## Predict

# %% [markdown]
# preprocess - Test Dataset 삽입 자리

# %%
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# %%
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='korean', task='transcribe')
model.config.suppress_tokens = []

# %%
# load test dataset from local storage
test_dataset = load_from_disk('dataset_test')

# %%
test_args = Seq2SeqTrainingArguments(
    torch_compile=True, # for optimize code

    output_dir="repo_name",

    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    dataloader_num_workers=8,
    
    logging_steps=25,
    # report_to=["wandb"],
)

# %%
from transformers import pipeline

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

# %%



wandb.save("*.py")
wandb.save("*.ipynb")

wandb.save("*.pt")
wandb.save("*.pth")
wandb.save("*.hdf5")

wandb.save("*.csv")
wandb.save("checkpoint/*")

wandb.save("model/*")