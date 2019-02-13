import pandas as pd 
import cv2
from os.path import join as PJ
import scipy.io as sio


DATASET = "apy"

DATALIST = ["apascal_train", "apascal_test", "ayahoo_test"]

REPLACE_IMG = False
CROP = True
DATA = "ayahoo_test"

ROOT = PJ("..", "dataset")
concept_filename = PJ(ROOT, DATASET, "list", "concepts.txt")

ATT_SPLITS = sio.loadmat(PJ(ROOT, "xlsa17", "data", DATASET, "att_splits.mat"))
RES101 = sio.loadmat(PJ(ROOT, "xlsa17", "data", DATASET, "res101.mat"))

ORIGIN_ATTR = PJ(ROOT, DATASET, "attribute_data")

if REPLACE_IMG:

    img_files = []
    for d in DATALIST:
        data = pd.read_csv(PJ(ORIGIN_ATTR, d + ".txt"), header=None, delimiter=" ")
        img_files += [PJ("img", d + "_" + str(i + 1) + ".jpg") for i in range(len(data))]

    labels = RES101['labels'].reshape(-1)
    labels = labels - 1

    data = pd.DataFrame({'img_path': img_files, 'label': labels})

    train_val = data.iloc[ATT_SPLITS['trainval_loc'].reshape(-1) - 1]
    test_seen = data.iloc[ATT_SPLITS['test_seen_loc'].reshape(-1) - 1]
    test_unseen = data.iloc[ATT_SPLITS['test_unseen_loc'].reshape(-1) - 1]

    train_val.to_csv(PJ(ROOT, DATASET, "list", "train_val.txt"), index=False, header=False)
    test_seen.to_csv(PJ(ROOT, DATASET, "list", "test_seen.txt"), index=False, header=False)
    test_unseen.to_csv(PJ(ROOT, DATASET, "list", "test_unseen.txt"), index=False, header=False)


if CROP:

    data = pd.read_csv(PJ(ORIGIN_ATTR, DATA + ".txt"), header=None, delimiter=" ")

    # length = range(len(data))
    length = [2210]
    for i in length:

        img_path = data.iloc[i, 0]

        if DATA.find("apascal") > -1:
            img_path = PJ(ROOT, DATASET, "img", "APY/images_att/VOCdevkit/VOC2008/JPEGImages", img_path)
        elif DATA.find("ayahoo") > -1:
            img_path = PJ(ROOT, DATASET, "img", "APY/ayahoo_test_images", img_path)

        img = cv2.imread(img_path)
        print(img_path)
        print(img.shape)

        x_min, y_min, x_max, y_max = data.iloc[i, 2: 6]
        print(data.iloc[i, 2: 6])
        croped_img = img[y_min: y_max, x_min: x_max, :]

        cv2.imwrite(PJ(".", "img", DATA + "_" + str(i + 1) + ".jpg"), croped_img)
