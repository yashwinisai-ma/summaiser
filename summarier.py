import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import nltk
import os
ps = PorterStemmer()

#preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return [ps.stem(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]

#frequency tble of all the word in the corpus
def _create_frequency_table(words):
    freq_table = defaultdict(int)
    for word in words:
        freq_table[word] += 1
    return freq_table

#frequency tble of all the word in each sentence the corpus
def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    for sent in sentences:
        words = preprocess_text(sent)
        freq_table = _create_frequency_table(words)
        frequency_matrix[sent] = freq_table
    return frequency_matrix

#term frequency table of each word in each sentence
def _create_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, f_table in freq_matrix.items():
        total_words = sum(f_table.values())
        tf_table = {word: count / total_words for word, count in f_table.items()}
        tf_matrix[sent] = tf_table
    return tf_matrix

#table for no of document that contains each word
def _create_documents_per_words(freq_matrix):
    word_per_doc_table = defaultdict(int)
    for f_table in freq_matrix.values():
        for word in f_table:
            word_per_doc_table[word] += 1
    return word_per_doc_table

#create IDF matrix
def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {word: math.log10(total_documents / count_doc_per_words[word]) for word in f_table}
        idf_matrix[sent] = idf_table
    return idf_matrix

#TF-IDF scores for each word in the corpus
def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for sent in tf_matrix:
        tf_idf_table = {word: tf_matrix[sent][word] * idf_matrix[sent][word] for word in tf_matrix[sent]}
        tf_idf_matrix[sent] = tf_idf_table
    return tf_idf_matrix

#score for each sentence based on the sum of its TF-IDF scores
def score_sentences(tf_idf_matrix):
    sentence_scores = defaultdict(float)
    for sent, f_table in tf_idf_matrix.items():
        sentence_scores[sent] = sum(f_table.values())
    return sentence_scores

#generate the summary
def get_summary(sentences, sentence_scores, num_sentences):
    # Sort sentences based on their scores
    sorted_sentences = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
    # Select the top 'num_sentences' sentences
    top_sentences = sorted(sorted_sentences[:num_sentences], key=lambda item: sentences.index(item[0]))
    # Combine the top sentences into a summary
    summary = [item[0] for item in top_sentences]
    return ' '.join(summary)

#input and its respective summary
def summarize_text(text, num_sentences):
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    freq_matrix = _create_frequency_matrix(sentences)
    tf_matrix = _create_tf_matrix(freq_matrix)
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    sentence_scores = score_sentences(tf_idf_matrix)
    summary = get_summary(sentences, sentence_scores, num_sentences)
    return summary

def save_summary_to_file(summary, filename="summary.txt"):
    with open(filename, "w") as file:
        file.write(summary)

# Example 
text = """
Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language. 
NLP enables computers and digital devices to recognize, understand and generate text and speech by combining computational linguistics—the rule-based modeling of human language—together with statistical modeling, machine learning (ML) and deep learning. 
NLP research has enabled the era of generative AI, from the communication skills of large language models (LLMs) to the ability of image generation models to understand requests. 
NLP is already part of everyday life for many, powering search engines, prompting chatbots for customer service with spoken commands, voice-operated GPS systems and digital assistants on smartphones. NLP also plays a growing role in enterprise solutions that help streamline and automate business operations, increase employee productivity and simplify mission-critical business processes.
"""
num_sentences = 3  
summary = summarize_text(text, num_sentences)
print(summary)

a=input("Do you want to save the summary? Yes/No: ")
if a.lower().startswith('y'):
    b=os.getcwd()
    save_summary_to_file(summary)
    print(f"Summary saved as \"summary.txt\" to \"{b}\"")
