import os
import math
from collections import Counter, defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set up environment
corpus_root = 'C:/Users/chira/Downloads/P1/US_Inaugural_Addresses'  # Path to the directory with text files
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

# Load and preprocess documents
def load_documents(root_path):
    documents = []
    for filename in os.listdir(root_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(root_path, filename)
            with open(file_path, 'r', encoding='windows-1252') as file:
                doc = file.read().lower()
            tokens = tokenizer.tokenize(doc)
            tokens = [token for token in tokens if token not in stop_words]
            tokens = [stemmer.stem(token) for token in tokens]
            documents.append(tokens)
    return documents

documents = load_documents(corpus_root)
print("Documents loaded and preprocessed.")

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
            tf_val = 1 + math.log10(count)
            idf_val = math.log10(N / document_frequency[token])
            tfidf[token] = tf_val * idf_val
            doc_length += tfidf[token] ** 2

        doc_length = math.sqrt(doc_length)
        for token in tfidf:
            tfidf[token] /= doc_length
        tfidf_vectors.append(tfidf)
    return tfidf_vectors

tfidf_vectors = build_tfidf(documents)
print("TF-IDF model built.")

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
    similarities = [(i, cosine_similarity(query_vector, doc_vector)) for i, doc_vector in enumerate(tfidf_vectors)]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0] if similarities else ("None", 0)

# Example calls to the functions
print("%.12f" % getidf('president'))
print("%.12f" % getweight('01_washington_1789.txt', 'citizen'))

# Query the system
result = query("democracy in America")
print(f"The most relevant document: {result}")