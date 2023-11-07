# author: Aaryan Tyagi
#
# License: 	Apache License Version 2.0


#importing libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import re 
from wordcloud import WordCloud
from nltk.util import everygrams
from sentence_transformers import SentenceTransformer
import pickle
from gensim.parsing.preprocessing import remove_stopwords
import Tests.scraper as scraper
import numpy as np
from numpy.linalg import norm
import networkx as nx
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import logging
from collections import Counter
import PyPDF2
import warnings


#Allowed sentence bert models
sbert_models = ['all-distilroberta-v1',
                'all-mpnet-base-v2',
                'all-MiniLM-L12-v2',
                'all-MiniLM-L6-v2']

chunking_methods = ['token', 'sentence', 'paragraph']

default_embeds = 'Data/embeds.pickle'
default_dataset = 'Data/wiki.csv'


# -------------------------------------------------------------------------
#Exceptions
class EmbeddingsNotFound(Exception):
    pass

class IncorrectFormat(Exception):
    pass

class UnknownEmbeddingModel(Exception):
    pass
# -------------------------------------------------------------------------


class insigen:
    """
    insigen 

    Creates a topic distribution, wordcloud and summary of a document

    Parameters
    -----------

    :param tokenizer: callable (Optional)
            Use this parameter to specify your own tokenization method. 
            If this is set to None, the default tokenizer will be used.

    :param chunking_method: str (Default = token)
            This parameter specifies how the document will be segmented.
            Valid chunking methods are: 
                * token
                * sentence
                * paragraph
    
    :param max_num_chunks: int (Optional)
            This parameter is used when using the token chunking method.
            It specifies the maximum number of chunks that should be 
            created while segmenting the document
    
    :param chunk_length: int (Default = 100)
            This parameter should be used when using token chunking method.
            It specifies how many tokens a chunk can contain. 
    
    :param overlapping_ratio: float (Default = 0.2)
            This parameter should be used when using token chunking method.
            It specifies the ratio of text that overlaps in each chunk to 
            maintain the context in each chunk. 
            The value should range between 0 and 1.    
    
    :param use_pretrained_embeds: bool (Optional, default = True)
            Setting this parameter to False will allow you to train
            your own embeddings. Further parameters need to be 
            specified for training
    
    :embed_file: str (Optional)
            This parameter should be used when you've trained your own 
            embeddings. Specify the path to your sentence embeddings.
            Note that the embeddings shape should match your dataset, 
            i.e. if there are 5000 text rows in the dataset, there should 
            be 5000 sentence embeddings in the file.
    
    :dataset_file: str (Optional)
            This parameter should be used when you've trained your own 
            embeddings. Specify the path to your own dataset.
            Note that the dataset should contain the following columns:
                * text
                * category_label
    
    :param embedding_model: str (Default = all-mpnet-base-v2)
            Insigen uses sentence bert models to train it's embeddings.
            Valid models are: 
                * all-distilroberta-v1
                    Performance Sentence Embeddings (14 Datsetes): 68.73
                    Speed: 4000
                    Model Size: 290 MB 
                    Max Sequence Length: 512
                    Dimensions:	768
                * all-mpnet-base-v2
                    Performance Sentence Embeddings (14 Datsetes): 69.57
                    Speed: 2800
                    Model Size: 420 MB 
                    Max Sequence Length: 384
                    Dimensions:	768
                * all-MiniLM-L12-v2
                    Performance Sentence Embeddings (14 Datsetes): 68.70
                    Speed: 7500
                    Model Size: 120 MB 
                    Max Sequence Length: 256
                    Dimensions:	384
                * all-MiniLM-L6-v2
                    Performance Sentence Embeddings (14 Datsetes): 68.06
                    Speed: 14200
                    Model Size: 80 MB 
                    Max Sequence Length: 256
                    Dimensions:	384

    ----------

    """    

    def __init__(self,
                 tokenizer: callable = None,
                 chunking_method: str = 'token',
                 max_num_chunks: int = None,
                 chunk_length: int = 100,
                 overlapping_ratio: float = 0.2,
                 use_pretrained_embeds: bool = True,
                 embed_file: str = 'Data/embeds.pickle',
                 dataset_file: str = 'Data/wiki.csv',
                 embedding_model: str = 'all-mpnet-base-v2',
                 verbose: bool = False,
                 ) -> None:
        
        if chunking_method not in chunking_methods:
            raise ValueError("Unkown chunking method found. Allowed chunking methods: ['token', 'sentence', 'paragraph']")
        if embedding_model not in sbert_models:
            raise UnknownEmbeddingModel("""Unknown model found. Allowed models: 
                                        ['all-distilroberta-v1',
                                        'all-mpnet-base-v2',
                                        'all-MiniLM-L12-v2',
                                        'all-MiniLM-L6-v2']""")
        
        self.use_pretrained_embeds = use_pretrained_embeds
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.chunking_method = chunking_method
        self.trained = True

        if verbose:
            logging.basicConfig(level=logging.INFO)

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

        #set tokenizer
        if tokenizer is None: 
            self.tokenizer = self._default_tokenizer
        

        #Checking pretrained embeddings
        if use_pretrained_embeds:
            self.embeds = self._load_embeds('Data/embeds.pickle')
            self.dataset = self._load_wiki_dataset()
            self.unique_topics = np.unique(self.dataset['category_label'])
            self.topic_vectors = self._calculate_topic_vectors()

        else:
            if embed_file == default_embeds or dataset_file == default_dataset:
                warnings.warn("Use the train() method to train the model on your own dataset")
                self.trained = False
            else:
                self.embeds = self._load_embeds(embed_file)
                self.dataset = pd.read_csv(dataset_file)
        
                self.unique_topics = np.unique(self.dataset['category_label'])
                self.topic_vectors = self._calculate_topic_vectors()
    
    def _form_sentence_vectors(self, document):
        """This method chunks a document and creates sentence vectors
        for those chunks

        Args:
            document (str): The document that is converted to sentence vectors

        Raises:
            EmbeddingsNotFound: If Pre-trained embeddings are not found this error is raised
        """        

        logging.info("Converting document to sentence vectors")
        if not self.trained:
            raise EmbeddingsNotFound("Embeddings Not Found. Use Pre-Trained Embeddings instead, or train your own model using the train method")

        self.document = document

        if self.chunking_method == 'token':
            self.document_tokens = self.tokenizer(document)
            self.document_chunks = self.chunker(self.document_tokens)
            self.clean_chunks = [''.join(chunk) for chunk in self.document_chunks]

        else:
            self.document_chunks = self.chunker(document)
            self.tokenized_chunks = [self.tokenizer(chunk) for chunk in self.document_chunks]
            self.clean_chunks = [' '.join(chunk) for chunk in self.tokenized_chunks]
        
        self.sentence_vectors = self._embed_document(self.clean_chunks)
    

    def _calculate_chunk_weights(self, chunks):
        """This method calculates the weight for each chunk.
        It is used in calculating the topic distribution

        Args:
            chunks (list): A list of chunks for the documenet

        Returns:
            list: weight matrix
        """ 

        logging.info("Calculating chunk weight matrix")
        max_len = max([len(chunk.split(' ')) for chunk in chunks])
        weight_matrix = [1]*len(chunks)

        for i in range(len(chunks)):
            chunk = chunks[i].split(' ')
            weight_matrix[i] = len(chunk)/max_len

        return weight_matrix

    def _calculate_topic_vectors(self):
        """This method is used to compute topic vectors from the
        trained embeddings. The embeddings must match the dataset.

        Returns:
            Numpy Array: Topic vectors
        """        
        logging.info("Calculating topic vectors")
        topic_inds = [np.where(self.dataset['category_label'] == topic) for topic in self.unique_topics]
        topic_vecs = [self.embeds[ind] for top in topic_inds for ind in top]

        return np.array([np.mean(vecs, axis = 0) for vecs in topic_vecs])
    

    def _compute_cosine(self, A: list, B: list) -> float:
        """
        Function to calculate the cosine angle between two vectors
        In other words, it is used to check the similarity of two vectors/embeddings

        Args:
            A (nd array): Vector A
            B (nd array): Vector B

        Returns:
            float: Returns a value between 0 and 1, where 0 represents least similar and 1 represents most similar
        """        

        return np.dot(A,B)/(norm(A)*norm(B))


    def _calculate_topics(self, query_embed, metric = 'max', max_count = 1, threshold = 0.5):
        """This method calculates the similar topics based on a metric

        Args:
            query_embed (Numpy Array): Text sentence vector 
            metric (str, optional): This metric defines how the 
            topics will be found Can be set to 'threshold', to get
            all the topics above a similarity threshold. Defaults to 'max'.
            'Max' metric gets the top "n" topics
            max_count (int, optional): This argument should be used with 
            max metric. It specifies the top x amount of topics that
            get fetched. Defaults to 1.
            threshold (float, optional): This argument should be used with
            threshold metric. It specifies the threshold similarity over 
            which all topics will be fetched. Defaults to 0.5.

        Returns:
            tuple: (list of topics, corresponding similarity scores)
        """        
        logging.info("Finding topic distribution")
        similarity = np.array([self._compute_cosine(query_embed, vec) for vec in self.topic_vectors])
        if metric == 'threshold':
            indexes = np.where(similarity >= threshold)
        else:
            indexes = np.argsort(similarity)[-max_count:][::-1]

        return self.unique_topics[indexes], similarity[indexes]

    def _load_wiki_dataset(self):
        """Loads the default wikipedia article dataset

        Returns:
            DataFrame: Wikipedia dataset
        """        
        return pd.read_csv('Data/wiki.csv')
    
    def _load_embeds(self, filename):
        """Load the sentence embeddings.

        Args:
            filename (str): Path to the file where the embeddings are stored

        Returns:
            Numpy Array: Sentence vectors
        """        
        embFile = open(filename, 'rb')
        embeds = pickle.load(embFile)

        return embeds

    def _clean_text(self, text):
        return re.sub(r'\[ \d+ \]', '', text)
    
    def _default_tokenizer(self, text):
        """This is the default tokenization method. It is 
        used when no other tokenizer is specified

        Args:
            text (str): The text that needs to be tokenized

        Returns:
            List: List of tokens
        """        
        return simple_preprocess(strip_tags(text), deacc=True)

    def _embed_document(self, documents):
        """Convert text to a sentence vector.
        Sentence bert is used for forming the embeddings. 

        Args:
            documents (str, List): List of documents or a single document that needs to be vectorized

        Returns:
            Numpy Array: Sentence vector for the document
        """        

        embedding_model = self.embedding_model
        model = SentenceTransformer(embedding_model)
        return model.encode(documents)
    
    def _sentence_chunking(self, document):
        """This method is used to chunk a document sentence wise

        Args:
            document (str): Text that needs to be chunked

        Returns:
            list: List of document chunks
        """        
        return sent_tokenize(document)

    def _paragraph_chunking(self, document):
        """This method is used to chunk a document paragraph wise

        Args:
            document (str): Text that needs to be chunked

        Returns:
            list: List of document chunks
        """        
        return document.split('\n\n')

    def _token_chunking(self, tokens):
        """This method is used to chunk a document using tokens.
        Number of tokens in a chunk and number of chunks can be specified

        Args:
            document (str): Text that needs to be chunked

        Returns:
            list: List of document chunks
        """        

        num_chunks = int(np.ceil(len(tokens)/self.chunk_length))
        if self.max_num_chunks != None:
            num_chunks = num_chunks if num_chunks < self.max_num_chunks else self.max_num_chunks

        return [" ".join(tokens[i:i + self.chunk_length])
            for i in list(range(0, len(tokens), int(self.chunk_length * (1 - self.overlapping_ratio))))[0:num_chunks]]

    def get_text_from_pdf(self, filename: str) -> str:
        """A method to fetch text from a pdf file

        Args:
            filename (str): path to the pdf that needs to be fetched

        Returns:
            str: Text from the pdf
        """        
        with open(filename, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = '\n'.join([reader.pages[i].extract_text() for i in range(len(reader.pages))])

            f.close()

        return text
    
    def generate_wordcloud(self, word_frequency_count: dict,
                           width = 800,
                           height = 400,
                           background_color = 'white',
                           colormap = 'viridis'):
        
        """
        Generates a wordcloud from the frequency distribution of words in a text

        Args:
        word_frequency_count (dict): A dictionary containing the frequency of each word 
        width (int): Wordcloud parameter; Sets the width of the wordcloud image. Defaults to 800.
        height (int): Wordcloud parameter; Sets the height of the wordcloud image. Defaults to 400.
        background_color (str):  Wordcloud parameter; sets the background color of the image. Defaults to white.
        colormap (str): Wordcloud parameter; sets the colormap of the image. Defaults to viridis 

        Returns:
            obj: Wordcloud object
        """                
        
        logging.info("Generating Word Cloud")
        
        wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap
            )

        cloud = wordcloud.generate_from_frequencies(frequencies=word_frequency_count)

        return cloud
    
    def generate_summary(self, text, topic_match, topic_weight = 1, similarity_weight = 1, position_weight = 10, num_sentences = 10):
        """This method generates an extractive summary of the text

        Args:
            text (str): The text for which the summary needs to be generated
            topic_match (str): a topic that can match with the text. 
            This adds additional weight to sentences that are more related to the topic. 
            use the unique_topic attribute to get a list of topics that can match.
            topic_weight (int, optional): Adds weightage to the topic similarity score
            Increasing this parameter results to more topic oriented summary.
            similarity_weight (int, optional): Adds weightage to sentence similarity score.
            Increasing this parameter results in extracting more co-related sentences.
            position_weight (int, optional): Adds weightage to the position of the sentences.
            Increasing this parameter results to more position oriented summary; i.e 
            Texts present early in the document are given more weightatge. Defaults to 10.
            num_sentences (int, optional): This specifies the number of sentences that 
            are to be included in the summary. Defaults to 10.

        Returns:
            str: Summary of the text
        """        
        sentences = self._sentence_chunking(text)
        clean_sentences = [' '.join(simple_preprocess(strip_tags(sent), deacc=True)) for sent in sentences]
        sentence_vectors = self._embed_document(clean_sentences)
        embeds = pickle.load(open('Data/embeds.pickle', 'rb'))
        topics = pickle.load(open('Data/topics.pickle', 'rb'))

        topic_embed = embeds[list(topics).index(topic_match)]
        
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for i in range(len(sentence_vectors)):
            topic_similarity = topic_weight*(self._compute_cosine(sentence_vectors[i], topic_embed))
            position_score = position_weight*(1.0 - (i / len(sentence_vectors)))
            for j in range(len(sentence_vectors)):
                if i != j:
                    similarity_matrix[i][j] = similarity_weight*(self._compute_cosine(sentence_vectors[i], sentence_vectors[j])) + topic_similarity + position_score
            

        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]

        summary = ' '.join(summary_sentences)

        return summary

    
    def get_keyword_frequency(self, document, min_len = 2, max_len = 4, frequency_threshold = 2):
        """This method can be used to generate a word frequency dictionary

        Args:
            document (str): The text for which the word frequency need to be generated
            min_len (int, optional): This parameter specifies the minimum length 
            of the n-grams that are to be found in the text. Defaults to 2.
            max_len (int, optional): This parameter specifies the maximum length
            of the n-grams that are to be found in the text. Defaults to 4.
            frequency_threshold (int, optional): This parameter specifies
            the minimum occurence of the n-grams to be included in the dictionary. Defaults to 2.

        Returns:
            dict: word count dictionary
        """        
        logging.info("Finding keyword frequencies")
        document_tokens = self.tokenizer(document)

        ngrams = list(everygrams(document_tokens, min_len=min_len, max_len=max_len))

        frequent_grams = dict()
        for gram in ngrams:
            without_stopwords = remove_stopwords(' '.join(gram))
            if without_stopwords == '':
                continue
            else:
                frequent_grams[' '.join(gram)] = ngrams.count(gram)
        
        filtered_words = remove_stopwords(' '.join(document_tokens)).split(' ')
        word_freq = Counter(filtered_words)

        frequent_grams.update(word_freq)

        self.frequent_grams = {key: val for key, val in frequent_grams.items() if val > frequency_threshold}

        return self.frequent_grams

    def get_topic_distribution(self, document, metric = 'max', max_count = 1, threshold = 0.5):
        """Get a topic distribution for a given document, or text

        Args:
            document (str): the text for which the topic distribution is to be generated
            metric (str, optional): This metric defines how the 
            topics will be found Can be set to 'threshold', to get
            all the topics above a similarity threshold. Defaults to 'max'.
            'Max' metric gets the top "n" topics
            max_count (int, optional): This argument should be used with 
            max metric. It specifies the top x amount of topics that
            get fetched. Defaults to 1.
            threshold (float, optional): This argument should be used with
            threshold metric. It specifies the threshold similarity over 
            which all topics will be fetched. Defaults to 0.5.

        Returns:
            dict: topic distribution, representing what percentage of the topic 
            matches with the text
        """        
        self._form_sentence_vectors(document)

        chunk_weights = self._calculate_chunk_weights(self.clean_chunks)
        topic_scores = {}
        topic_counts = {}

        for i, vec in enumerate(self.sentence_vectors):
            topics, scores = self._calculate_topics(vec, metric=metric, max_count=max_count, threshold = threshold)
            for j, top in enumerate(topics):
                if top in topic_scores.keys():
                    topic_scores[top] += scores[j] * chunk_weights[i]
                    topic_counts[top] += 1
                else:
                    topic_scores[top] = scores[j]
                    topic_counts[top] = 1

        topic_distribution = {topic: topic_scores[topic] / topic_counts[topic] for topic in topic_scores.keys()}
        topic_sum = sum(topic_distribution.values())
        
        percentage_distribution = {topic: (score / topic_sum) * 100 for topic, score in topic_distribution.items()}

        return percentage_distribution
    
    def train_embeds(self, dataset, batch_size = 32):
        """This method can be used to train your own embeddings 
        using your own dataset 

        Args:
            dataset (Pandas DataFrame): A pandas dataframe, that contains the 
            data for training. The dataset must include the columns "text" and 
            "category_label"
            batch_size (int, optional): Trains the embeddings in batches. Defaults to 32.

        Raises:
            IncorrectFormat: If the dataframe does not contain the "text" and
            "category_label" column, this error is raised

        Returns:
            Numpy Array: Trained embeddings/sentence vectors
        """        
        if 'text' not in dataset.columns or 'category_label' not in dataset.columns:
            raise IncorrectFormat('The dataframe must contain a "text" and "category_label" column')
        
        dataset = dataset.dropna()
        dataset['text'] = dataset['text'].apply(self._clean_text)
        tokenized_text = dataset.apply(self._default_tokenizer)
        train_corpus = [' '.join(tokens) for tokens in tokenized_text]

        num_batches = len(train_corpus) // batch_size
        if len(train_corpus) % batch_size != 0:
            num_batches += 1

        sentence_vectors = []

        for batch_idx in tqdm(range(num_batches), desc="Training Progress"):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_texts = train_corpus[start_idx:end_idx]

            batch_vectors = self._embed_document(batch_texts)
            sentence_vectors.extend(batch_vectors)

        sentence_vectors = np.vstack(sentence_vectors)

        self.embeds = sentence_vectors
        self.dataset = dataset
        self.unique_topics = np.unique(self.dataset['category_label'])
        self.topic_vectors = self._calculate_topic_vectors()
        self.trained = True

        return sentence_vectors



'''article = scraper.getArticle('wiki/League_of_Legends')
ins = insigen(use_pretrained_embeds=False)
print("Topic Distribution:\n", ins.get_topic_distribution(article))
print("Summary:\n", ins.generate_summary(article, topic_match='Games and Toys'))
freq = ins.get_keyword_frequency(article, min_len=2, max_len=3)
cloud = ins.generate_wordcloud(freq)
plt.imshow(cloud)
plt.show()'''