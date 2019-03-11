import os
from os.path import join as PJ
import scipy.io as sio
from pandas import DataFrame as df

# Setting
DATASET = "awa2"

ROOT = PJ("..", "dataset")
XLSA17 = PJ(ROOT, "xlsa17", "data", DATASET)
CONCEPT = PJ(ROOT, DATASET, "list", "concepts", "concepts.txt")
CONCEPT_LIST = PJ(ROOT, DATASET, "list", "concepts")
SPLIT_LIST = PJ(ROOT, DATASET, "list", "train_val")

ATT_SPLITS = sio.loadmat(PJ(XLSA17, "att_splits.mat"))
RES101 = sio.loadmat(PJ(XLSA17, "res101.mat"))

# check path exist
if not os.path.isdir(PJ(ROOT, DATASET, "list")):
    os.makedirs(PJ(ROOT, DATASET, "list"))

if not os.path.isdir(CONCEPT_LIST):
    os.makedirs(CONCEPT_LIST)

if not os.path.isdir(SPLIT_LIST):
    os.makedirs(SPLIT_LIST)

# reorganize data
img_files = [filter(None, i[0][0].split('/')) for i in RES101['image_files']]
img_files = [PJ(*list(i)[5:]) for i in img_files]

labels = RES101['labels'].reshape(-1)
labels = labels - 1

if DATASET == 'apy':
    img_files = [PJ("img", i) for i in sorted(os.listdir(PJ(ROOT, DATASET, "img", "img")))]
    data = df({'img_path': img_files, 'label': labels})
else:
    data = df({'img_path': img_files, 'label': labels})

# generate concept file
with open(CONCEPT, "w") as f:
    f.write('\n'.join([label[0][0] for label in ATT_SPLITS['allclasses_names']]))

# split into train and val
split_data = data.iloc[ATT_SPLITS['trainval_loc'].reshape(-1) - 1]
split_data.to_csv(PJ(SPLIT_LIST, "trainval.txt"), index=False, header=False)
