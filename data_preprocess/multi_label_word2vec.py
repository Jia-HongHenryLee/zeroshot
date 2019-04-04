import os
from os.path import join as PJ
import re
import yaml
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

# config
DATASET = "nus_wide"

ROOT = PJ("..", "dataset")
ORIGIN = PJ(ROOT, DATASET, "0origin")

# replace word

REPLACE = {
    "colour": "Color",
    "colours": "Colors",
    "colourful": "Colorful",
    "grey": "Grey",
    "olympus": "Olympus",
    "harbour": "Harbour",
    "nederland": "Nederland",
    "maldives": "Maldives",
    "oahu": "Oahu",
    "kauai": "Kauai"
}

################################
# check path exist
################################

CONCEPT_LIST = PJ(ROOT, DATASET, "list", "concepts")
SPLIT_LIST = PJ(ROOT, DATASET, "list", "train_test")

if not os.path.isdir(PJ(ROOT, DATASET, "list")):
    os.makedirs(PJ(ROOT, DATASET, "list"))

if not os.path.isdir(CONCEPT_LIST):
    os.makedirs(CONCEPT_LIST)

if not os.path.isdir(SPLIT_LIST):
    os.makedirs(SPLIT_LIST)

################################
# duplicated concept remove
################################

# read train/test concept
print("read concept")
lists = ["train", "test"]
wordlists = {l: [] for l in lists}
for l in lists:
    with open(PJ(CONCEPT_LIST, "concept_" + l + ".txt")) as f:
        wordlists[l] = [re.sub("[\d\s]", "", word).lower() for word in f.readlines()]

# remove duplicate word
remove_id = [i for i, w in enumerate(wordlists['train']) if w in wordlists['test']]
remove_id.reverse()
for r in remove_id:
    wordlists['train'].pop(r)

print("number of training concept: ", len(wordlists['train']))
print("number of test concept: ", len(wordlists['test']))
print()

# output concept file
print("merge train/test concpt into concept.txt")
with open(PJ(CONCEPT_LIST, "concept.txt"), "w") as f:
    f.write('\n'.join(['\n'.join(wordlists[l]) for l in lists]))

# output id_split file
print("output id_split file\n")
train_id = list(range(len(wordlists['train'])))
test_id = [len(wordlists['train']) + i for i in range(len(wordlists['test']))]

with open(PJ(SPLIT_LIST, "id_split.txt"), "w") as f:
    yaml.dump({"train_id": train_id, "test_id": test_id}, f)

################################
# check word2vec
################################

# load word2vec model
weight_path = PJ(ROOT, "google_news", "GoogleNews-vectors-gensim-normed.bin")
word2vec = KeyedVectors.load(weight_path, mmap='r')
word2vec.wv.vectors_norm = word2vec.wv.vectors

# check word2vec
print("check word embedding")
for i, w in enumerate(wordlists['train']):
    if w in REPLACE:
        wordlists['train'][i] = REPLACE[w]
    if w not in word2vec.wv.vocab and w not in REPLACE:
        print('"' + w + '":"' + w.capitalize() + '"')

# output vector file
print("output word embedding file")
word_vec_dict = {}
for l in lists:
    word_vec_dict.update({w: word2vec[w] for w in wordlists[l]})
filename = PJ(CONCEPT_LIST, "concepts_vec.txt")
pd.DataFrame.from_dict(word_vec_dict).to_csv(filename, index=False)

################################
# split data into train / test
################################

# read imagelist
image_list = pd.read_csv(PJ(ORIGIN, "ImageList", "Imagelist.txt"), header=None)

# read label list
train_label = pd.read_csv(PJ(ORIGIN, "NUS_WID_Tags", "AllTags1k.txt"), sep="\t", header=None).iloc[:, :1000]
test_label = pd.read_csv(PJ(ORIGIN, "NUS_WID_Tags", "AllTags81.txt"), sep=" ", header=None).iloc[:, :81]

# remove duplicate label
train_label = train_label.drop(remove_id, axis=1)

# merge into data
data = pd.concat([image_list, train_label, test_label], axis=1, ignore_index=True)

# split dataset and output
print("split dataset")
train_list = pd.read_csv(PJ(ORIGIN, "ImageList", "TrainImagelist.txt"), header=None)
test_list = pd.read_csv(PJ(ORIGIN, "ImageList", "TestImagelist.txt"), header=None)

train_data = data[data[0].isin(train_list[0])]
test_data = data[data[0].isin(test_list[0])]

# insert train/test path
train_data.loc[:, 0] = 'train/img/' + train_data.loc[:, 0].str.replace(r"[^\s]+\\", "", regex=True)
test_data.loc[:, 0] = 'test/img/' + test_data.loc[:, 0].str.replace(r"[^\s]+\\", "", regex=True)

# remove empty label
train_data = train_data[(train_data.iloc[:, 1:926] == 1).sum(axis=1) > 0]
test_data = test_data[(test_data.iloc[:, 1:] == 1).sum(axis=1) > 0]

print("number of train data", train_data.shape[0])
print("number of test data", test_data.shape[0])

# fmt = ['%s'] + ['%i'] * (train_data.shape[1] - 1)
# np.savetxt(PJ(SPLIT_LIST, "train.txt"), train_data.values, delimiter=",", fmt=' '.join(fmt))
# np.savetxt(PJ(SPLIT_LIST, "test.txt"), test_data.values, delimiter=",", fmt=' '.join(fmt))
train_data.to_csv(PJ(SPLIT_LIST, "train.txt"), index=False, header=False)
test_data.to_csv(PJ(SPLIT_LIST, "test.txt"), index=False, header=False)
