import re, pprint, os, numpy, io
import nltk
from bs4 import BeautifulSoup
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from tikapp import TikaApp
from Tika_reader import TikaReader
from textblob import TextBlob

if __name__ == "__main__":
    folder = "CorpusHTMLNoticiasPractica1819"
    # Empty list to hold text documents.
    texts = []
    listing = sorted(os.listdir(folder))
    print(listing)
    for file in listing:
        if file.endswith(".html"):
            url = folder+"/"+file
            processor = TikaReader(url.decode("utf-8"))
            if processor.detect_language() == "en":
                f = io.open(url,encoding="utf-8");
            else:
                f = io.open(url,encoding="latin-1");
            raw = f.read()
            f.close()
            soup = BeautifulSoup(raw,'lxml')
            text = ""
            for node in soup.findAll('p'):
                text = text + node.text
            txt_file = open(file[:-5] + ".txt", "w")
            print(txt_file)
            if processor.detect_language() == "en":
                print("Originally english: ", file)
                txt_file.write(text.encode("utf-8"))
                txt_file.close()
            elif processor.detect_language() == "es":
                text_blob = TextBlob(text)
                translated = text_blob.translate(to="en")
                print("Translated: ", str(translated))
                txt_file.write(str(translated))
                txt_file.close()
            else:
                text_blob = TextBlob(text)
                translated = text_blob.translate(from_lang="es", to="en")
                print("Translated: ", str(translated))
                txt_file.write(str(translated))
                txt_file.close()



