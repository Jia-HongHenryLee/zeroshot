import re
import matplotlib.pyplot as plt

from os.path import join as PJ
from pandas import read_csv, DataFrame
from sklearn.manifold import TSNE

DATASET = "aPY"
CONCEPT = 'new'

if __name__ == '__main__':

    datapath = PJ('..', 'dataset', DATASET.lower(), 'list', 'concepts', 'concepts_' + CONCEPT + '.txt')
    data = read_csv(datapath)

    class_name = [re.sub("[\d\s\.]", "", c.lower()) for c in data.columns.values]
    data = data.values.T

    # model = PCA(n_components=2)
    model = TSNE(n_components=2, init='pca', learning_rate=10)
    result = model.fit_transform(data)
    trans_data = DataFrame({'x': result[:, 0], 'y': result[:, 1], 'label': class_name})

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    plt.title(DATASET + " t-sne visualization", y=1.08, weight='semibold')

    for line in range(0, trans_data.shape[0]):
        ax.scatter(trans_data.x[line] - 0.2, trans_data.y[line] - 0.2, color="white")
        ax.text(trans_data.x[line] - 0.2, trans_data.y[line] - 0.2, trans_data.label[line],
                horizontalalignment='left', color='black', weight='normal')

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # adds major gridlines
    ax.grid(which='both', color='grey', linestyle='-', linewidth=0.25, alpha=0.25)

    # ax.set_frame_on(False)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('')

    # plt.show()
    img_filename = ['attr', 'old word2vec', 'new  word2vec']
    c = ['attr', 'old', 'new'].index(CONCEPT)
    plt.savefig(DATASET + " " + img_filename[c] + " t-sne visualization.jpg")
