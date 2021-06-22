import nltk
import sys
import os
import string
import math

FILE_MATCHES = 2
SENTENCE_MATCHES = 2


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    print("Loading, please wait...")
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for filename in os.listdir(directory):
        f = open(os.path.join(directory, filename), "r", encoding="utf8")
        files[filename] = f.read()
        f.close()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # String to lower case
    s = document.lower()

    # Split into token words
    wordList = nltk.word_tokenize(s)

    # Clean list removing puntuation symbols and stopwords
    wordListClean = []
    punctuationSymbols = [i for i in string.punctuation]
    for word in wordList:
        if word not in punctuationSymbols and word not in nltk.corpus.stopwords.words("english"):
            wordListClean.append(word)

    return wordListClean


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Create a set of unique words across all documents
    uniqueWords = set()
    for document in documents:
        for word in documents[document]:
            uniqueWords.add(word)
    
    # Create a dictionary mapping each word to their IDF value
    idf_map = dict()
    for word in uniqueWords:
        repetitions = 0
        for document in documents:
            if word in documents[document]:
                repetitions += 1
        idf_map[word] = math.log(len(documents)/repetitions)

    return idf_map


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Create a dictionary mapping files to calculated ranking (sum tf-idf values of query words that appear in the file)
    rankings = dict()
    for filename in files:
        tf_idf_sum = 0
        for word in query:
            if word in files[filename]:
                tf = files[filename].count(word)
                idf = idfs[word]
                tf_idf = tf * idf
                tf_idf_sum += tf_idf
        rankings[filename] = tf_idf_sum

    # Create top_files list ordered according to file rankings just calculated
    top_files_tuples = sorted(rankings.items(), key = lambda item: item[1], reverse = True)
    top_files = [top_file_tuple[0] for top_file_tuple in top_files_tuples]
    n_top_files = []
    for i in range(n):
        n_top_files.append(top_files[i])
    
    return n_top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Create a dictionary mapping sentences to calculated ranking (sum idf values of query words that appear in the sentence)
    sentences_tuples = []
    for sentence in sentences:
        idf_sum = 0
        for word in query:
            if word in sentences[sentence]:
                idf = idfs[word]
                idf_sum += idf

        # Calculate sentence qtd (query term density)
        queryWordCount = 0
        for word in sentences[sentence]:
            if word in query:
                queryWordCount += 1
        qtd = queryWordCount / len(sentences[sentence])

        # Append list to be sorted
        sentences_tuples.append((sentence, sentences[sentence], idf_sum, qtd))

    # Sort with first key idf and second key query term density
    sentences_tuples_sorted = sorted(sentences_tuples, key = lambda item: (item[2], item[3]), reverse = True)

    # Take only the sentences from the sorted tuples
    sentences_sorted = [sentence_tuple_sorted[0] for sentence_tuple_sorted in sentences_tuples_sorted]

    # For debugging
    # print(sentences_sorted[:5])

    # Return n top sentences sorted by higher query term density in case of tie
    top_sentences = []
    for i in range(n):
        top_sentences.append(sentences_sorted[i])
    
    return top_sentences


if __name__ == "__main__":
    main()
