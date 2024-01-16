import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText

# Load datasets
dataset_df = pd.read_csv('dataset.csv')
authors_df = pd.read_csv('authors.csv')
authors_df = authors_df.dropna(subset=['author_name'])
authors_df['author_id'] = range(1, len(authors_df) + 1)
dataset_df['authors'] = dataset_df['authors'].apply(lambda x: x[1:-1] if isinstance(x, str) else x)
authors_df['author_id'] = authors_df['author_id'].astype(str)
merged_df = pd.merge(dataset_df, authors_df, left_on='authors', right_on='author_id', how='left')
merged_df['authors'] = merged_df['author_name']
merged_df = merged_df[['title', 'description', 'bestsellers-rank', 'authors']]
merged_df.description.fillna('', inplace=True)
merged_df.dropna(inplace=True)
merged_df = merged_df.sort_values(by=['bestsellers-rank'], ascending=False)

# Text preprocessing
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = contractions.fix(doc)
    tokens = nltk.word_tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(list(merged_df['description']))

# TF-IDF vectorization
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)

# Cosine similarity
doc_sim = cosine_similarity(tfidf_matrix)
doc_sim_df = pd.DataFrame(doc_sim)

# FastText model
tokenized_docs = [doc.split() for doc in norm_corpus]
ft_model = FastText(tokenized_docs, window=30, min_count=2, workers=4, sg=1)

# Averaged Word2Vec vectorizer
def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)

doc_vecs_ft = averaged_word2vec_vectorizer(tokenized_docs, ft_model, 100)
doc_sim_ft = cosine_similarity(doc_vecs_ft)
doc_sim_df_ft = pd.DataFrame(doc_sim_ft)

# Streamlit App
st.title('Book Recommendation System')

# Input Text
input_text = st.text_input('Input Book Description:', '')

if input_text:
    st.markdown('**Input Book Description:**')
    st.write(input_text)

    # TF-IDF Recommendations
    st.subheader('TF-IDF Recommendations:')
    input_tfidf = tf.transform([normalize_document(input_text)])
    input_sim_tfidf = cosine_similarity(input_tfidf, tfidf_matrix)
    input_sim_tfidf = input_sim_tfidf.flatten()
    recommended_books_tfidf = merged_df.iloc[np.argsort(-input_sim_tfidf)[:5]]['title'].tolist()
    st.write(recommended_books_tfidf)

    # FastText Recommendations
    st.subheader('FastText Recommendations:')
    input_vec_ft = averaged_word2vec_vectorizer([normalize_document(input_text).split()], ft_model, 100)
    input_sim_ft = cosine_similarity(input_vec_ft, doc_vecs_ft)
    input_sim_ft = input_sim_ft.flatten()
    recommended_books_ft = merged_df.iloc[np.argsort(-input_sim_ft)[:5]]['title'].tolist()
    st.write(recommended_books_ft)
