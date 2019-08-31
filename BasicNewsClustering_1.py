# encoding=utf8

import re, pprint, os, io, numpy
import nltk
from bs4 import BeautifulSoup
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from nltk.corpus import stopwords
from tikapp import TikaApp
from Tika_reader import TikaReader

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
                                      linkage="average", affinity=distanceFunction)
    clusters = clusterer.fit_predict(vectors)

    return clusters

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf_idf(word, document))
    return word_tf

def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names

def to_unicode(s):
    if type(s) is unicode:
        return s
    elif type(s) is str:
        d = chardet.detect(s)
        (cs, conf) = (d['encoding'], d['confidence'])
        if conf > 0.80:
            try:
                return s.decode( cs, errors = 'replace' )
            except Exception as ex:
                pass
    # force and return only ascii subset
    return unicode(''.join( [ i if ord(i) < 128 else ' ' for i in s ]))

if __name__ == "__main__":
    folder = "CorpusHTMLNoticiasPractica1819"
    # Empty list to hold text documents.
    texts = []

    listing = sorted(os.listdir(folder))
    print(listing)
    for file in listing:
        if file.endswith(".html"):
            url = folder+"/"+file
            print(url)
            f = io.open(url,encoding="latin-1")
            raw = f.read()
            f.close()
            #soup = BeautifulSoup(f, 'lxml')
            soup = BeautifulSoup(raw, 'lxml')
            text = ""
            for node in soup.findAll('p'):
                text = text + node.text
            print("Texto: ", text)
            sentences = nltk.sent_tokenize(text)
            tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

            ## Prueba 1. Lematizacion. Descomentar:
            ## INICIO PRUEBA 1
            # Seleccionamos el lematizador.
            wordnet_lemmatizer = WordNetLemmatizer()
            # Obtenemos los tokens de las sentencias.
            tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
            # print(tagged_sentence)
            tokens = []
            def wordnet_value(value):
                result = ''
                # Filtramos las palabras y nos quedamos solo las que nos pueden interesar.
                # Estas son Adjetivos, Verbos, Sustantivos y Adverbios.
                if value.startswith('J'):
                    return wordnet.ADJ
                elif value.startswith('V'):
                    return wordnet.VERB
                elif value.startswith('N'):
                    return wordnet.NOUN
                elif value.startswith('R'):
                    return wordnet.ADV
                return result

            for sentence in tagged_sentences:
                for token in sentence:
                    if token[0] is not None and len(token[0]) > 0:
                        pos = wordnet_value(token[1])
                        # Filtramos las palabras que no nos interesan.
                        if pos != '':
                            tokens.append(wordnet_lemmatizer.lemmatize(str(token[0]).lower(), pos=pos))
            print("Lemmas: ", tokens)
            print("Type of lemmas: ", type(tokens))
            ## FIN PRUEBA 1


            ## Prueba 2. Quitar puntuacion
            ## INICIO PRUEBA 2
            punctuation = string.punctuation
            tokens_wo_punctuation = []
            for t in tokens:
                if t not in punctuation:
                    tokens_wo_punctuation.append(t)
            print("Wo punctuation: ", tokens_wo_punctuation)
            tokens = tokens_wo_punctuation
            ## FIN PRUEBA 2

            # ## Prueba 3. Quitar stopwords
            # ## INICIO PRUEBA 3
            # tokens_wo_stopwords = []
            # for t in tokens:
            #     if t not in stopwords.words('english'):
            #         tokens_wo_stopwords.append(t)
            # print("Wo stopwords: ", tokens_wo_stopwords)
            # tokens = tokens_wo_stopwords
            # ## FIN PRUEBA 3

            ## Prueba 3.a. Detectar idioma y quitar stoprwords para cada idioma
            ## INICIO PRUEBA 3.a
            processor = TikaReader(url)
            tokens_wo_stopwords = []
            if processor.detect_language() == "en":
                for t in tokens:
                    if t not in stopwords.words('english'):
                        tokens_wo_stopwords.append(t)
            elif processor.detect_language() == "es":
                for t in tokens:
                    if t not in stopwords.words('spanish'):
                        tokens_wo_stopwords.append(t)
            else:
                for t in tokens:
                    if t not in stopwords.words('spanish'):
                        tokens_wo_stopwords.append(t)
            print("Idioma: ", processor.detect_language())
            print("Wo stopwords: ", tokens_wo_stopwords)
            tokens = tokens_wo_stopwords
            # FIN PRUEBA 3.a

            ## Prueba 4. Chunker por defecto de NLTK - roconocedor de NE
            ## INICIO PRUEBA 4
            tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
            print("Tagged: ", tagged_sentences)
            #universal_tagged_sentences = [nltk.pos_tag(sentence, tagset='universal') for sentence in
                                          #tokenized_sentences]
            chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
            entity_names = []
            for tree in chunked_sentences:
                entity_names.extend(extract_entity_names(tree))
            print("NEs: ", entity_names)
            tokens = entity_names
              ## FIN PRUEBA 4

            text = nltk.Text(tokens)
            texts.append(text)

    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")

    distanceFunction ="cosine"
    #distanceFunction = "euclidean"
    test = cluster_texts(texts,6,distanceFunction)
    print("test: ", test)
    # Gold Standard
    reference =[0, 1, 2, 2, 2, 3, 2, 2, 2, 4, 0, 0, 3, 3, 4, 2, 3, 0, 4, 4, 5, 1]
    print("reference: ", reference)

    # Evaluation
    print("rand_score: ", adjusted_rand_score(reference,test))