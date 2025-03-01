import os
import math
from collections import Counter, defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Set up environment
corpus_root = './US_Inaugural_Addresses'  # Path to the directory with text files
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

# Load and preprocess documents
def load_documents(root_path):
    documents = []
    filenames = []
    for filename in os.listdir(root_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(root_path, filename)
            with open(file_path, 'r', encoding='windows-1252') as file:
                doc = file.read().lower()
            tokens = tokenizer.tokenize(doc)
            tokens = [token for token in tokens if token not in stop_words]
            tokens = [stemmer.stem(token) for token in tokens]
            documents.append(tokens)
            filenames.append(filename)
    return documents, filenames

documents, filenames = load_documents(corpus_root)
print("Documents loaded and preprocessed.")

# Build TF-IDF model and Posting Lists
def build_tfidf(documents):
    N = len(documents)
    document_frequency = Counter()
    postings_lists = defaultdict(list)
    tfidf_vectors = []
    
    for doc_id, tokens in enumerate(documents):
        unique_tokens = set(tokens)
        for token in unique_tokens:
            document_frequency[token] += 1
    
    for doc_id, tokens in enumerate(documents):
        tf = Counter(tokens)
        tfidf = {}
        doc_length = 0
        
        for token, count in tf.items():
            if token in document_frequency:
                tf_val = 1 + math.log10(count)
                idf_val = math.log10(N / document_frequency[token])
                tfidf_weight = tf_val * idf_val
                tfidf[token] = tfidf_weight
                doc_length += tfidf_weight ** 2
                postings_lists[token].append((doc_id, tfidf_weight))

        doc_length = math.sqrt(doc_length)
        if doc_length > 0:
            for token in tfidf:
                tfidf[token] /= doc_length
                postings_lists[token] = [(d, w) for d, w in postings_lists[token] if d != doc_id]
                postings_lists[token].append((doc_id, tfidf[token]))
                postings_lists[token].sort(key=lambda x: x[1], reverse=True)

        tfidf_vectors.append(tfidf)
    
    return tfidf_vectors, document_frequency, postings_lists

tfidf_vectors, document_frequency, postings_lists = build_tfidf(documents)
print("TF-IDF model built.")

# Define necessary functions
def getidf(token):
    token = stemmer.stem(token)
    if token not in document_frequency:
        return -1
    return math.log10(len(documents) / document_frequency[token])

def getweight(filename, token):
    token = stemmer.stem(token)
    if filename not in filenames:
        return 0.0
    doc_index = filenames.index(filename)
    return tfidf_vectors[doc_index].get(token, 0.0)

def calculate_query_vector(query):
    tokens = tokenizer.tokenize(query.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    tf_query = Counter(stemmed_tokens)
    query_vector = {}
    query_length = 0
    
    for token, count in tf_query.items():
        if token in document_frequency:
            tf_val = 1 + math.log10(count)
            idf_val = math.log10(len(documents) / document_frequency[token])
            tfidf_weight = tf_val * idf_val
            query_vector[token] = tfidf_weight
            query_length += tfidf_weight ** 2

    query_length = math.sqrt(query_length)
    if query_length > 0:
        for token in query_vector:
            query_vector[token] /= query_length
    
    return query_vector

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    denominator = math.sqrt(sum([val ** 2 for val in vec1.values()])) * math.sqrt(sum([val ** 2 for val in vec2.values()]))
    return numerator / denominator if denominator else 0.0

def query(qstring):
    query_vector = calculate_query_vector(qstring)
    token_top10 = {token: postings_lists[token][:10] for token in query_vector if token in postings_lists}
    
    if not token_top10:
        return ("None", 0.0)
    
    candidate_docs = set(d for top10 in token_top10.values() for d, _ in top10)
    actual_scores = {doc_id: cosine_similarity(query_vector, tfidf_vectors[doc_id]) for doc_id in candidate_docs}
    
    best_doc_id = max(actual_scores, key=actual_scores.get, default=None)
    if best_doc_id is not None and actual_scores[best_doc_id] > 0:
        return (filenames[best_doc_id], actual_scores[best_doc_id])
    
    return ("fetch more", 0.0)

# Example calls to the functions
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

result = query("the british are invading usa")
print(f"The most relevant document: {result}")