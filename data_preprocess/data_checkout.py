import os
import re
import pandas as pd
import numpy as np
from os.path import join as PJ
import json
import scipy.io as sio
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

DATASET = "apy"

CONCEPTS = False
EDIT_ORIGIN_DATA = True

ROOT = PJ("..", "dataset")
concept_filename = PJ(ROOT, DATASET, "list", "concepts.txt")

ATT_SPLITS = sio.loadmat(PJ(ROOT, "xlsa17", "data", DATASET, "att_splits.mat"))
RES101 = sio.loadmat(PJ(ROOT, "xlsa17", "data", DATASET, "res101.mat"))

setting = {
    'apy': {
        'replace_word': {'pottedplant': 'potted_plant',
                         'aeroplane': 'airplane',
                         'tvmonitor': 'tv',
                         'diningtable': 'table'}},
    'awa2': {
        'replace_word': {'persian+cat': 'longhaired_cat',
                         'siamese+cat': 'Siamese_cat',
                         'blue+whale': 'fin_whale',
                         'grizzly+bear': 'grizzly_bear',
                         'killer+whale': 'killer_whale',
                         'german+shepherd': 'german_shepherd',
                         'spider+monkey': 'spider_monkey',
                         'humpback+whale': 'humpback_whale',
                         'giant+panda': 'giant_panda',
                         'polar+bear': 'polar_bear'}},
    'sun': {
        'replace_word': {}
    },
    'cub': {
        'replace_word': {}
    }
}

# Output Replaced Word
if CONCEPTS:

    weight_path = PJ(ROOT, "google_news", "GoogleNews-vectors-gensim-normed.bin")
    word2vec = KeyedVectors.load(weight_path, mmap='r')
    word2vec.vectors_norm = word2vec.vectors

    if DATASET == 'sun':
        wordlist = [c[0][0] for c in ATT_SPLITS['allclasses_names']]
        with open(concept_filename, "w") as f:
            f.writelines("\n".join(wordlist))
    else:
        with open(concept_filename) as f:
            wordlist = [re.sub("[\d\s]", "", word) for word in f.readlines()]

    replace_word = setting[DATASET]['replace_word']

    for i, word in enumerate(wordlist):
        if word in replace_word:
            wordlist[i] = replace_word[word]
        if word not in word2vec.vocab:
            print(word)
            w = {i[0].lower(): i[1] for i in word2vec.most_similar(word.split("_"), topn=50)}
            print(w)

    with open(concept_filename, "w") as f:
        f.writelines("\n".join(wordlist))

if EDIT_ORIGIN_DATA:

    img_files = [filter(None, i[0][0].split('/')) for i in RES101['image_files']]
    img_files = [PJ(*list(i)[5:]) for i in img_files]

    labels = RES101['labels'].reshape(-1)
    labels = labels - 1

    data = pd.DataFrame({'img_path': img_files, 'label': labels})

    train_val = data.iloc[ATT_SPLITS['trainval_loc'].reshape(-1) - 1]
    test_seen = data.iloc[ATT_SPLITS['test_seen_loc'].reshape(-1) - 1]
    test_unseen = data.iloc[ATT_SPLITS['test_unseen_loc'].reshape(-1) - 1]

    train_id = list(set(train_val['label']))
    test_id = list(set(test_unseen['label']))

    ids = {"train_id": train_id, "test_id": test_id}
    with open(PJ(ROOT, DATASET, "list", "id_split.txt"), "w") as f:
        json.dump(ids, f)

    train_val.to_csv(PJ(ROOT, DATASET, "list", "train_val.txt"), index=False, header=False)
    test_seen.to_csv(PJ(ROOT, DATASET, "list", "test_seen.txt"), index=False, header=False)
    test_unseen.to_csv(PJ(ROOT, DATASET, "list", "test_unseen.txt"), index=False, header=False)
