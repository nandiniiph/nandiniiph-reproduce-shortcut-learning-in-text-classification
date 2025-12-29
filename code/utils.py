import numpy as np
import pandas as pd
import evaluate
import torch
import matplotlib.pyplot as plt
from collections import Counter
from transformers import Trainer
import random
import os
from datasets import Dataset, DatasetDict

# ===================== DATASET INFO =====================

DATASETS_INFO = {
    'go_emotions': {
        'label_shift_to_index': 0,
        'num_labels': 4,
        'path': '../Dataset/go_emotions/',
        'INSTRUCTION_KEY': (
            'Pick the most suitable one from the 4 emotions for the text: '
            'neutral, amusement, joy and excitement. Use [0, 1, 2, 3].\n\n'
            '0: neutral\n1: amusement\n2: joy\n3: excitement\n'
            'Follow the format:\nText:\n[Text]\n### Category: [NUMBER]'
        ),
        'INPUT_KEY': 'Text:',
        'RESPONSE_KEY': '### Category: [',
        'training_dataset_size': 1120,
    },
    'yelp': {
        'label_shift_to_index': 1,
        'num_labels': 5,
        'path': '../Dataset/yelp_review_full_csv/',
        'INSTRUCTION_KEY': (
            'Select one rating from [1, 2, 3, 4, 5] according to this review.\n'
            'Follow the format:\nReview:\n[REVIEW]\n### Rating: [NUMBER]'
        ),
        'INPUT_KEY': 'Review:',
        'RESPONSE_KEY': '### Rating: [',
        'training_dataset_size': 8900,
    },
    'beer': {
        'label_shift_to_index': 2,
        'num_labels': 4,
        'path': '../Dataset/beer_new/target/',
        'INSTRUCTION_KEY': (
            'Select one rating from [2, 3, 4, 5] according to this review.\n'
            'Follow the format:\nReview:\n[REVIEW]\n### Rating: [NUMBER]'
        ),
        'INPUT_KEY': 'Review:',
        'RESPONSE_KEY': '### Rating: [',
        'training_dataset_size': 1999,
    }
}

# ===================== DATASET LOADING =====================

def load_dataset(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, usecols=['label', 'text'])
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, usecols=['label', 'text'])
    else:
        raise ValueError("Unsupported file format")

    if 'beer' in file_path:
        df['label'] = (df['label'] / 2 * 10).astype(int)

    return Dataset.from_pandas(df)


def dataset_read(dataset_dir, shortcut_type, shortcut_subtype, setting_No, label_shift):
    dataset_dict = DatasetDict()
    dataset_path = os.path.join(
        dataset_dir,
        f'{shortcut_type}/split/{shortcut_subtype}',
        f'{shortcut_subtype}{setting_No}'
    )

    for split in dataset_types:
        if split == 'test':
            path = os.path.join(dataset_path[:-1] + '1', f'{split}.csv')
        else:
            path = os.path.join(dataset_path, f'{split}.csv')

        ds = load_dataset(path)
        dataset_dict[split] = ds.map(
            lambda x: {
                'text': x['text'],
                'label': int(x['label'] - label_shift)
            }
        )

    anti_test_path = os.path.join(
        dataset_dir,
        f'{shortcut_type}/split/{shortcut_subtype}',
        f'{shortcut_subtype}1/test_anti-shortcut.csv'
    )
    ds = load_dataset(anti_test_path)
    dataset_dict['test_anti-shortcut'] = ds.map(
        lambda x: {
            'text': x['text'],
            'label': int(x['label'] - label_shift)
        }
    )

    original_test = (
        os.path.join(dataset_dir, shortcut_type, 'test.xlsx')
        if 'beer' in dataset_dir
        else os.path.join(dataset_dir, shortcut_type, 'test.csv')
    )
    ds = load_dataset(original_test)
    dataset_dict['ori-test'] = ds.map(
        lambda x: {
            'text': x['text'],
            'label': int(x['label'] - label_shift)
        }
    )

    return dataset_dict

# ===================== METRICS =====================

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(
        predictions=preds,
        references=labels
    )["accuracy"]

    f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="macro"
    )["f1"]

    return {"accuracy": acc, "f1": f1}

# ===================== UTILITIES =====================

def frequency_show(dataset):
    lengths = [len(t.split()) for t in dataset]
    freq = Counter(lengths)

    plt.bar(freq.keys(), freq.values())
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.show()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wandb_project_name(model, shortcut_type, subtype):
    return f"Shortcuts-{model}-{shortcut_type}-{subtype}"


def wandb_exp_name(dataset, setting, exp, test_type):
    return f"{dataset}_{setting}_{exp}_{test_type}"


# ===================== GLOBAL CONFIG =====================

DATASET_NAME = 'yelp'
dataset_types = ['train', 'dev', 'test']

seeds = [0, 1, 2, 10, 42]

SET_SHORTCUT_TYPE = 'occurrence'
SET_SHORTCUT_SUBTYPE1 = 'single-word'
SET_SHORTCUT_SUBTYPE2 = 'single-word'

BATCH_SIZE = 16
NUM_EPOCHS = 5
GRADIENT_ACCUMULATION = 2
MAX_NEW_TOKENS = 5
TEMPERATURE = 0.0

# ===================== BERT HYPERPARAMETER =====================

BERT_LR = {
    'yelp': {
        'single-word': 2e-5,
        'multiple-word': 2e-5,
        'random': 2e-5,
    },
    'go_emotions': {
        'single-word': 2e-5,
        'multiple-word': 2e-5,
        'random': 2e-5,
    },
    'beer': {
        'single-word': 2e-5,
        'multiple-word': 2e-5,
        'random': 2e-5,
    }
}

WEIGHT_DECAY = {
    'yelp': {
        'single-word': 0.01,
        'multiple-word': 0.01,
        'random': 0.01,
    },
    'go_emotions': {
        'single-word': 0.01,
        'multiple-word': 0.01,
        'random': 0.01,
    },
    'beer': {
        'single-word': 0.01,
        'multiple-word': 0.01,
        'random': 0.01,
    }
}
