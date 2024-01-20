<h1 align="center">Insigen</h1>
<p align="center"><i>A novel approach towards analysis of documents and gaining insights from textual data</i></p>

- Trained on around 6000 wikipedia articles
- Semantic understanding of textual data
- Finding organizations within textual structures
- Mapping textual data to relevant generalized topics
- Finding a topic distribution within a textual structure
- Extracting a relevant summary of the textual data
- Finding a keyword distrbution with n-grams and ner
- Can be trained on customized dataset and relevant topics

<h2 aligh="center">Working of Insigen</h2>
The idea that insigen is based upon is that no textual structure is 100% related to a singular topic. Insigen builds upon the concept of distributed representations of topics, which involves capturing the semantic associations between words and documents in a continuous vector space. It draws inspiration from models like (Top2Vec)[https://github.com/ddangelov/Top2Vec/tree/master/] and extends their functionality. The fundamental idea is to create a spatial representation where distance reflects semantic association, allowing for a more nuanced understanding of the relationships between words and topics.