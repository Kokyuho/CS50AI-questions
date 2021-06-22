# CS50’s Introduction to Artificial Intelligence with Python
# Project 6: Language: Questions

**Aim**: Write an AI to answer questions.

**Description**: Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. In this problem, we design a very simple question answering system based on inverse document frequency (tf-idf). *nltk* functions are used for tokenization and stopwords removal.

Our question answering system will perform two tasks: document retrieval and passage retrieval. Our system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.

More problem set info here: https://cs50.harvard.edu/ai/2020/projects/6/questions/

More info about tf-idf modeling: https://cs50.harvard.edu/ai/2020/notes/6/#tf-idf


Usage:
```
python questions.py corpus
```

Example:
```
$ python questions.py corpus
Query: What are the types of supervised learning?
Types of supervised learning algorithms include Active learning , classification and regression.

$ python questions.py corpus
Query: When was Python 3.0 released?
Python 3.0 was released on 3 December 2008.

$ python questions.py corpus
Query: How do neurons connect in a neural network?
Neurons of one layer connect only to neurons of the immediately preceding and immediately following layers.
```
