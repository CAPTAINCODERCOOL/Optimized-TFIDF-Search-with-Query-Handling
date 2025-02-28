import os

import math
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set up environment
corpus_root = './US_Inaugural_Addresses'  # Path to the directory with text files
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

def load_documents(root_path):
    documents = []
    filenames = []  # Store filenames
    
    for filename in os.listdir(root_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(root_path, filename)
            with open(file_path, 'r', encoding='windows-1252') as file:
                doc = file.read().lower()
            tokens = tokenizer.tokenize(doc)
            tokens = [token for token in tokens if token not in stop_words]
            tokens = [stemmer.stem(token) for token in tokens]
            
            documents.append(tokens)
            filenames.append(filename)  # Store filename
            
    return documents, filenames

documents, filenames = load_documents(corpus_root)  # Load docs & filenames


# Build TF-IDF model
def build_tfidf(documents):
    N = len(documents)
    document_frequency = Counter()
    for tokens in documents:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            document_frequency[token] += 1

    tfidf_vectors = []
    for tokens in documents:
        tf = Counter(tokens)
        tfidf = {}
        doc_length = 0
        for token, count in tf.items():
            if token in document_frequency:
                tf_val = 1 + math.log10(count)
                idf_val = math.log10(N / document_frequency[token])
                tfidf[token] = tf_val * idf_val
                doc_length += tfidf[token] ** 2

        doc_length = math.sqrt(doc_length)
        for token in tfidf:
            tfidf[token] /= doc_length
        tfidf_vectors.append(tfidf)
    return tfidf_vectors, document_frequency

tfidf_vectors, document_frequency = build_tfidf(documents)
print("TF-IDF model built.")

# Additional required functions
def calculate_query_vector(query):
    tokens = tokenizer.tokenize(query.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    tf_query = Counter(stemmed_tokens)
    query_vector = {}
    for token, count in tf_query.items():
        if token in document_frequency:
            tf_val = 1 + math.log10(count)
            idf_val = math.log10(len(documents) / document_frequency[token])
            query_vector[token] = tf_val * idf_val
    return query_vector

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([val ** 2 for val in vec1.values()])
    sum2 = sum([val ** 2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

# Define other necessary functions based on TF-IDF vectors
def getidf(token):
    token = stemmer.stem(token)
    return math.log10(len(documents) / document_frequency.get(token, len(documents)))

def getweight(filename, token):
    token = stemmer.stem(token)
    doc_index = int(filename.split('_')[0]) - 1
    return tfidf_vectors[doc_index].get(token, 0.0)

def query(qstring):
    query_vector = calculate_query_vector(qstring)
    similarities = [(filenames[i], cosine_similarity(query_vector, doc_vector)) for i, doc_vector in enumerate(tfidf_vectors)]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0] if similarities else ("None", 0)


# Example calls to the functions
print("%.12f" % getidf('president'))
print("%.12f" % getweight('01_washington_1789.txt', 'citizen'))
print("%.12f" % getidf('british'))
print("%.12f" % getidf('union'))
print("%.12f" % getidf('dollar'))
print("%.12f" % getidf('constitution'))
print("%.12f" % getidf('power'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','states'))
print("%.12f" % getweight('07_madison_1813.txt','war'))
print("%.12f" % getweight('05_jefferson_1805.txt','false'))
print("%.12f" % getweight('22_grant_1873.txt','proposition'))
print("%.12f" % getweight('16_taylor_1849.txt','duties'))
print("--------------")
print("(%s, %.12f)" % query("executive power"))
print("(%s, %.12f)" % query("foreign government"))
print("(%s, %.12f)" % query("public rights"))
print("(%s, %.12f)" % query("people government"))
print("(%s, %.12f)" % query("states laws"))
# Query the system
result = query("british are invading usa")
print(f"The most relevant document: {result}")