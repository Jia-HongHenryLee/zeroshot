# import multiprocessing
# from wikicorpus import WikiCorpus
# from gensim.models.word2vec import Word2Vec
# from gensim.models.word2vec import LineSentence
# from gensim.models import KeyedVectors

# import requests
# from bs4 import BeautifulSoup
# import re

# if __name__ == '__main__':

#     PRDUCE_FILE = False
#     REPLACE = False
#     MODEL_TRAIN = False
#     MODEL_TEST = True

#     if PRDUCE_FILE:
#         wiki_path = '../dataset/enwiki/enwiki-20190120-pages-articles.xml.bz2'

#         wiki = WikiCorpus(wiki_path, lemmatize=False, dictionary={})

#         with open("../dataset/enwiki/wiki_corpus.txt", "w") as f:
#             for i, text in enumerate(wiki.get_texts(), 1):
#                 f.write(' '.join([t.decode("utf-8") for t in text]) + '\n')
#                 if (i % 10000 == 0):
#                     print("Saved " + str(i) + " articles")

#     if REPLACE:

#         with open("../dataset/enwiki/wiki_corpus.txt", "r") as f:
#             for line in f:
#                 print(line)
#                 break

#     if MODEL_TRAIN:

#         params = {'size': 400, 'window': 10, 'min_count': 10,
#                   'workers': max(1, multiprocessing.cpu_count() - 2), 'sample': 1E-5, }

#         word2vec = Word2Vec(LineSentence("../dataset/enwiki/wiki_corpus.txt"), **params)
#         word2vec.init_sims(replace=True)
#         word2vec.save("enwiki-gensim-normed.bin")

#     if MODEL_TEST:
#         weight_path = "enwiki-gensim-normed.bin"
#         word2vec = KeyedVectors.load(weight_path, mmap='r')
#         with open("../dataset/sun/Concepts.txt") as f:
#             wordlist = [l.strip() for l in f.readlines()]
#         print(len(wordlist))

#         for w in wordlist:
#             if w not in word2vec.wv.vocab:
#                 print(w)
#                 q = "%20".join(w.split("_"))
#                 page = requests.get("https://www.google.com/search?q=" + q + "%20wiki")
#                 print(BeautifulSoup(page.text, "lxml").h3.a.b)
#                 print(word2vec.most_similar(w.split("_")))
#                 # print([i for i in word2vec.most_similar(["pencil", "skirt"], topn=1000000) if re.search("skirt", i[0])])

from wikipedia2vec import Wikipedia2Vec
import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from pyvirtualdisplay import Display
import re
import time

wiki2vec = Wikipedia2Vec.load("../dataset/enwiki/enwiki_20180420_300d.pkl")

with open("../dataset/sun/Concepts.txt") as f:
    wordlist = [l.strip() for l in f.readlines()]

display = Display(visible=0)
display.start()
opts = Options()
opts.add_argument("--incognito")
opts.add_argument("user-agent={}".format("Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0"))
driver = webdriver.Chrome(chrome_options=opts)

print("Total:", len(wordlist))

for i, w in enumerate(wordlist):

    if wiki2vec.get_word(w):
        continue

    elif wiki2vec.get_entity(" ".join(w.split("_")).capitalize()):
        wordlist[i] = " ".join(w.split("_")).capitalize()

    else:

        q = "+".join(['"' + i + '"' for i in w.split("_")]) + "+wiki"
        driver.get('https://www.google.co.jp/search?gl=us&hl=en&q=' + q + '&ie=utf-8&oe=utf-8')
        text = BeautifulSoup(driver.page_source, "html5lib").find(class_="LC20lb")
        if text.string is None:
            driver.get('https://www.plurk.com/is1024sa')
            time.sleep(600)
            driver.get('https://www.google.co.jp/search?gl=us&hl=en&q=' + q + '&ie=utf-8&oe=utf-8')
            text = BeautifulSoup(driver.page_source, "html5lib").find(class_="LC20lb")
        wordlist[i] = re.sub(" - Wikipedia$", "", text.string)
        time.sleep(30)

    if i % 25 == 0:
        driver.get('https://www.plurk.com/is1024sa')
        time.sleep(600)

    print(i)

driver.quit()

with open("../dataset/sun/new_Concepts.txt", "w") as f:
    for w in wordlist:
        f.write("%s\n" % w)
