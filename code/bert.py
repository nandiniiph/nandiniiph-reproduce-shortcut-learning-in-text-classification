import os
import pandas as pd
import numpy as np
import wandb
import evaluate

from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from utils import (
    load_dataset,
    compute_metrics,
    set_seed,
    DATASETS_INFO,
    DATASET_NAME,
    SET_SHORTCUT_TYPE,
    SET_SHORTCUT_SUBTYPE1,
    seeds,
    wandb_project_name,
    wandb_exp_name,
    BERT_LR,         
    WEIGHT_DECAY      
)


# ===================== CONFIG =====================

MODEL_TYPE = 'bert'
setting_No = 1
n_exp = 0
NUM_EPOCHS = 3
dataset_types = ['train', 'dev', 'test']

label_shift = DATASETS_INFO[DATASET_NAME]['label_shift_to_index']
dataset_dir = DATASETS_INFO[DATASET_NAME]['path']
num_labels = DATASETS_INFO[DATASET_NAME]['num_labels']

shortcut_type = SET_SHORTCUT_TYPE
shortcut_subtype = SET_SHORTCUT_SUBTYPE1

save_path = f"../model/{DATASET_NAME}/{MODEL_TYPE}/{shortcut_type}/{shortcut_subtype}_{setting_No}_{n_exp}"
test_type = 'normal'

model_name = 'bert-base-uncased' if test_type == 'normal' else save_path
repo_name = f'finetuning-{DATASET_NAME}-{MODEL_TYPE}-{shortcut_subtype}-{shortcut_subtype}'

# ====== HYPERPARAM (PASTIKAN ADA DI utils.py) ======
lr = BERT_LR[DATASET_NAME][shortcut_subtype]
weight_decay = WEIGHT_DECAY[DATASET_NAME][shortcut_subtype]

seed = seeds[n_exp]
set_seed(seed)

ori_test_flg = True
anti_test_flg = True
normal_test_flg = True

# ===================== W&B =====================

project_name = wandb_project_name(MODEL_TYPE, shortcut_type, shortcut_subtype)
exp_name = wandb_exp_name(DATASET_NAME, setting_No, n_exp, test_type)

wandb.init(
    project=project_name,
    name=exp_name,
    config={
        "learning_rate": lr,
        "architecture": model_name,
        "dataset": DATASET_NAME,
        "epochs": NUM_EPOCHS,
        "weight_decay": weight_decay,
    },
)

# ===================== DATASET LOADING =====================

dataset_dict = DatasetDict()
dataset_path = os.path.join(
    dataset_dir,
    f'{shortcut_type}/split/{shortcut_subtype}',
    f'{shortcut_subtype}{setting_No}'
)

for split in dataset_types:
    if split == 'test':
        file_path = os.path.join(dataset_path[:-1] + '1', f'{split}.csv')
    else:
        file_path = os.path.join(dataset_path, f'{split}.csv')

    ds = load_dataset(file_path)
    dataset_dict[split] = ds.map(
        lambda x: {'text': x['text'], 'label': int(x['label'] - label_shift)}
    )

# Anti-shortcut test
anti_test_path = os.path.join(
    dataset_dir,
    f'{shortcut_type}/split/{shortcut_subtype}',
    f'{shortcut_subtype}1/test_anti-shortcut.csv'
)
ds = load_dataset(anti_test_path)
dataset_dict['test_anti-shortcut'] = ds.map(
    lambda x: {'text': x['text'], 'label': int(x['label'] - label_shift)}
)

# Original test
original_test_path = os.path.join(dataset_dir, shortcut_type, 'test.csv')
ds = load_dataset(original_test_path)
dataset_dict['ori-test'] = ds.map(
    lambda x: {'text': x['text'], 'label': int(x['label'] - label_shift)}
)

# ===================== TOKENIZER & MODEL =====================

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

tokenized_train = tokenized_datasets['train'].shuffle(seed=seed)
tokenized_val = tokenized_datasets['dev']
tokenized_test = tokenized_datasets['test']
tokenized_test_anti = tokenized_datasets['test_anti-shortcut']
tokenized_test_ori = tokenized_datasets['ori-test']

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===================== TRAINING =====================

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=lr,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=weight_decay,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    report_to='wandb',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if test_type == 'normal':
    trainer.train()

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# ===================== EVALUATION =====================

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def run_test(name, dataset, csv_path, col_name):
    result = trainer.predict(dataset)
    preds = np.argmax(result.predictions, axis=-1)
    labels = result.label_ids

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]

    print(f"{name} Accuracy:", acc)
    print(f"{name} Macro F1:", f1)
    wandb.log({f"{name} Accuracy": acc, f"{name} Macro F1": f1})

    data = pd.read_csv(csv_path)
    data[col_name] = preds + label_shift
    data.to_csv(csv_path, index=False)

if normal_test_flg:
    run_test("Test", tokenized_test, anti_test_path, f'pred_{n_exp}')

if anti_test_flg:
    run_test("Anti-Test", tokenized_test_anti, anti_test_path, f'pred_{n_exp}')

if ori_test_flg:
    run_test(
        "Original Test",
        tokenized_test_ori,
        original_test_path,
        f'{shortcut_subtype}_{setting_No}_{n_exp}_pred'
    )

wandb.finish()
