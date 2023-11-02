import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from Tests import scraper  
from sentence_transformers import SentenceTransformer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import pickle
import numpy as np
from numpy.linalg import norm
import networkx as nx

def embed_documents(documents):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeds = model.encode
    return embeds(documents)

def cosine_similarity(A, B):
    return np.dot(A,B)/(norm(A)*norm(B))

def generate_summary(text, topic_match):
    sentences = sent_tokenize(text)
    clean_sentences = [' '.join(simple_preprocess(strip_tags(sent), deacc=True)) for sent in sentences]
    sentence_vectors = embed_documents(clean_sentences)
    embeds = pickle.load(open('Data/embeds.pickle', 'rb'))
    topics = pickle.load(open('Data/topics.pickle', 'rb'))

    topic_embed = embeds[list(topics).index(topic_match)]
    
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentence_vectors)):
        topic_similarity = cosine_similarity(sentence_vectors[i], topic_embed)
        position_score = 10*(1.0 - (i / len(sentence_vectors)))
        for j in range(len(sentence_vectors)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i], sentence_vectors[j]) + topic_similarity + position_score
        

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    num_sentences = 10
    summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]

    summary = ' '.join(summary_sentences)

    return summary


text = scraper.getArticle('wiki/Markiplier')
print(generate_summary(text, 'Science '))
