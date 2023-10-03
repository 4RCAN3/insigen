# author: Aaryan Tyagi
#
# License: 	apache-2.0

#importing libraries

import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import re 
import nltk
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.phrases import Phrases
from sentence_transformers import SentenceTransformer
import pickle
import scraper
import numpy as np
import logging 

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

sbert_models = ['all-distilroberta-v1',
                'all-mpnet-base-v2',
                'all-MiniLM-L12-v2',
                'all-MiniLM-L6-v2']

chunking_methods = ['token', 'sentence', 'paragraph']

class insigen:
    def __init__(self,
                 use_pretrained_embeds: bool = True,
                 chunking_method: str = 'token',
                 max_num_chunks: int = None,
                 embedding_model: str = 'all-mpnet-base-v2',
                 chunk_length: int = 100,
                 overlapping_ratio: float = 0.2,
                 tokenizer: callable = None,
                 verbose = True) -> None:
        
        if chunking_method not in chunking_methods:
            raise ValueError("Unkown chunking method found. Allowed chunking methods: ['token', 'sentence', 'paragraph']")
        
        self.use_pretrained_embeds = use_pretrained_embeds
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer

        if chunking_method == 'token':
            self.chunker = self._token_chunking
            self.max_num_chunks = max_num_chunks
            self.chunk_length = chunk_length
            self.overlapping_ratio = overlapping_ratio

        elif chunking_method == 'sentence':
            self.chunker = self._sentence_chunking
            pass
        else:
            self.chunker = self._paragraph_chunking
            pass

        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
            logger.setLevel(logging.WARNING)
            self.verbose = False

        #set tokenizer
        if tokenizer is None: 
            self.tokenizer = self._default_tokenizer
        

        #Checking pretrained embeddings
        if use_pretrained_embeds:
            self.embeds = self._load_embeds('embeds.pickle')
            self.dataset = self._load_wiki_dataset()
            self.unique_topics = np.unique(self.dataset['category_label'])
        else:
            self.embeds = None
    
    def analyse(self, document):
        self.document = document
        self.document_tokens = self.tokenizer(document)
        self.document_chunks = self.chunker(self.document_tokens)

        
    def _load_wiki_dataset(self):
        return pd.read_csv('wiki.csv')
    
    def _load_embeds(self, filename):
        embFile = open(filename, 'rb')
        embeds = pickle.load(embFile)

        return embeds

    def _clean_text(self, chunks):
        return [' '.join(self.tokenizer(chunk)) for chunk in chunks]
    
    def _default_tokenizer(self, text):
        return simple_preprocess(strip_tags(text), deacc=True)

    def _embed_document(self, documents):
        embedding_model = self.embedding_model
        model = SentenceTransformer(embedding_model)
        embeds = model.encode
        return embeds(documents)
    
    def _sentence_chunking(self, document):
        return nltk.sent_tokenize(document)

    def _paragraph_chunking(self, document):
        return document.split('\n\n')

    def _token_chunking(self, tokens):
        num_chunks = int(np.ceil(len(tokens)/self.chunk_length))
        if self.max_num_chunks != None:
            num_chunks = num_chunks if num_chunks < self.max_num_chunks else self.max_num_chunks

        return [" ".join(tokens[i:i + self.chunk_length])
            for i in list(range(0, len(tokens), int(self.chunk_length * (1 - self.overlapping_ratio))))[0:num_chunks]]



article = scraper.getArticle('wiki/League_of_Legends')
print(nltk.sent_tokenize(article))