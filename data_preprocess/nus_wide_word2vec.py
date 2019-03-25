import re
from os.path import join as PJ
import pandas as pd
from gensim.models import KeyedVectors

# config
DATASET = "nus_wide"

ROOT = PJ("..", "dataset")
REPLACE = {
    "colour": "color",
    "colours": "colors",
    "colourful": "colorful",
    "grey": "gray",
    "olympus": "Olympus",
    "harbour": "Harbour",
    "nederland": "Nederland",
    "maldives": "Maldives",
    "oahu": "Oahu",
    "kauai": "Kauai"
}

# read train/test label
lists = ["train", "test"]
wordlists = {l: [] for l in lists}
for l in lists:
    with open(PJ(ROOT, DATASET, l, l + "_concept_zsl.txt")) as f:
        wordlists[l] = [re.sub("[\d\s]", "", word).lower() for word in f.readlines()]

# remove duplicate word
for w in wordlists['test']:
    if w in wordlists['train']:
        wordlists['train'].remove(w)

print("number of training concept: ", len(wordlists['train']))
print("number of test concept: ", len(wordlists['test']))

# load word2vec model
weight_path = PJ(ROOT, "google_news", "GoogleNews-vectors-gensim-normed.bin")
word2vec = KeyedVectors.load(weight_path, mmap='r')
word2vec.wv.vectors_norm = word2vec.wv.vectors

for i, w in enumerate(wordlists['train']):
    if w in REPLACE:
        wordlists['train'][i] = REPLACE[w]
    if w not in word2vec.wv.vocab and w not in REPLACE:
        print('"' + w + '":"' + w.capitalize() + '"')

# output vector file
for l in lists:
    word_vec_dict = {w: word2vec[w] for w in wordlists[l]}
    pd.DataFrame.from_dict(word_vec_dict).to_csv(PJ(ROOT, DATASET, l, l + "_concept_vec_zsl.txt"), index=False)
