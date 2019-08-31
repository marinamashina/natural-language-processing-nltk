import re, pprint, os, numpy, io
import nltk
from bs4 import BeautifulSoup
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score

def cluster_texts(texts, clustersNumber, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a collection of", len(collection), "terms.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Unique terms found: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF(f,unique_terms, collection)) for f in texts]
    print("Vectors created.")

    # initialize the clusterer
    clusterer = AgglomerativeClustering(n_clusters=clustersNumber,
                                      linkage="average", affinity=distanceFunction) # esto se deja as
    clusters = clusterer.fit_predict(vectors) # que este predict sea parecido a reference

    return clusters

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = [] # array
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf

if __name__ == "__main__":
    folder = "CorpusHTMLNoticiasPractica1819"
    # Empty list to hold text documents.
    texts = []

    listing = sorted(os.listdir(folder))
    print(listing)
    for file in listing:
        if file.endswith(".html"):
            url = folder+"/"+file
            f = io.open(url,encoding="latin-1"); # por defecto latin1 se puede modificar en func del idioma
            raw = f.read()
            f.close()
            soup = BeautifulSoup(raw,'lxml')
            text = ""
            for node in soup.findAll('p'):
                text = text + node.text
            tokens = nltk.word_tokenize(text)
            text = nltk.Text(tokens)
            texts.append(text)

    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")

    distanceFunction ="cosine"
    #distanceFunction = "euclidean"
    test = cluster_texts(texts,6,distanceFunction) # 6
    print("test: ", test)
    # Gold Standard
    reference = [0, 1, 2, 2, 3, 2, 2, 2, 4, 0, 0, 3, 3, 4, 2, 3, 0, 4, 4, 4, 5, 2]
    print("reference: ", reference)

    # Evaluation
    print("rand_score: ", adjusted_rand_score(reference,test))
