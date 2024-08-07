<h1 align="center">InsiGEN</h1>
<p align="center"><i>A state of the art NLP model to generate insights from a text piece</i></p>

## Features of topic modelling
- Generating a distribution of generalized topics covered in a document/articler
- Extracting contextualized keywords from the text piece
- Generating a summary of the text
- Trained on a corpus of 6000 wikipedia articles for generalized topics
- Can be trained on custom data for more specific topics

## How to use the model
- Clone this repository
- Install the dependencies from the `requirements.txt`
- Basic Usage:

### Get a topic distribution

```python
from insigen import insigen
model = insigen()
topic_distribution = model.get_distribution(document)
```

#### Important parameters for `insigen`:
*  `use_pretrained_embeds`: Setting this parameter to False will allow you to train your own embeddings. Further parameters need to be specified for training
*  `embed_file`: This parameter should be used when you've trained your own embeddings. Specify the path to your sentence embeddings.
*  `dataset_file`: This parameter should be used when you've trained your own embeddings. Specify the path to your own dataset.
*  `embedding_model`: (Default = all-mpnet-base-v2) Insigen uses sentence bert models to train it's embeddings. Valid models are:

                all-distilroberta-v1
                all-mpnet-base-v2
                all-MiniLM-L12-v2
                all-MiniLM-L6-v2

Important parameters for `get_distribution`
* `document`: The text for which the topic distribution is to be generated
* `metric`: This metric defines how the topics will be found. Can be set to 'threshold', to get all the topics above a similarity threshold. Defaults to 'max'. 'Max' metric gets the top "n" topics
* `max_count`: This argument should be used with max metric. It specifies the top x amount of topics that get fetched. Defaults to 1.
* `threshold`: This argument should be used with threshold metric. It specifies the threshold similarity over which all topics will be fetched. Defaults to 0.5.


### Get keyword frequency

```python
frequency = model.get_keyword_frequency(document, min_len=2, max_len=3)

#Generate a wordcloud using the frequency
cloud = model.generate_wordcloud(frequency)
```

#### Important parameters for `get_keyword_frequency`
* `document`: The text for which the keyword frequency is to be generated
* `frequency_threshold`: minimum frequency of a n-gram to be considered in the keywords (`min_len` and `max_len` are also used to adjust the length of n-grams in the text)

### Generate Summary

```python
summary = model.generate_summary(article, topic_match=relevant_topic))

# To get a list of topics, use this
#print(model.unique_topics)
```

#### Import parameters for `generate_summary`
* `document`:The text for which the summary is to be generated
* `topic_match`: a topic that can match with the text. This adds additional weight to sentences that are more related to the topic. use `model.unique_topics` to get a list of topics that can match. Defaults to None, in which case weightage to related sentence will not be given.
* `topic_weight`: Adds weightage to the topic similarity score. Increasing this parameter results to more topic oriented summary. Defaults to 1.
* `similarity_weight`: Adds weightage to sentence similarity score. Increasing this parameter results in extracting more co-related sentences. Defaults to 1.
* `position_weight`: Adds weightage to the position of the sentences. Increasing this parameter results to more position oriented summary; i.e Texts present early in the document are given more weightatge. Defaults to 10.
* `num_sentences`: This specifies the number of sentences that are to be included in the summary. Defaults to 10.

### Train on your dataset

```python
embeddings = model.train_embeds(dataset)
```

#### Important parameters for `train_embeds`
* `dataset`: A pandas dataframe for the dataset to be trained
* `batch_size`: Batches to divide the dataset into. Defaults to 32.


## How does the model work? 

### Topic Distribution
- Create embedded vectors of labelled training articles
- Find mean embeddings of each topic in the corpus to create topic vectors and create clusters of articles\
  ![image](https://github.com/4RCAN3/insigen/assets/69053040/0180cc83-9369-43cb-85b3-ab2ec4ca947c)

- Use KNN to place new articles in the topic vector cluster\
  ![image](https://github.com/4RCAN3/insigen/assets/69053040/2fe59d9b-4273-4407-925e-97552f9116f8)

- Chunking each article and finding relevant topic from the topic vectors\
  ![image](https://github.com/4RCAN3/insigen/assets/69053040/311cdfba-b8da-46a3-936f-3439162da5b5)

### Keyword extraction
- N-grams and keywords are filtered from the text
- Contextually similar keywords to the article are given higher scoring
- A threshold is applied to the filtered list of keywords to get the final list of keywords


### Summary Extraction
- The PageRank algorithm is used to create a similarity matrix for sentences in the text
- Additionally, sentences are scored based on their position in the text and their similarity to a relevant topic
- Top N sentences from the similarity matrix are extracted to create a summary.