import pandas as pd
from preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

def sort_matrix(coo_matrix):
    matrix_tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(matrix_tuples, key=lambda a: (a[1], a[0]), reverse=True)

def exctract_vector_topn(feature_names, sorted_vectors, topn=5):
    sorted_vectors = sorted_vectors[:topn]
    score_values = []
    feature_values = []
    for i, score in sorted_vectors:
        score_values.append(round(score, 3))
        feature_values.append(feature_names[i])
    results = {}
    for i in range(len(feature_values)):
        results[feature_values[i]] = score_values[i]

    return results


def get_keywords(vectorizer, feature_names, desc):

    vectorized_desc = vectorizer.transform([desc])

    sorted_vectors = sort_matrix(vectorized_desc.tocoo())

    keywords = exctract_vector_topn(feature_names, sorted_vectors)

    return list(keywords.keys())

def exctract_keywords_from_df(books_df,ngram_rng = (1,1)):
    # books_df = pd.read_csv("processed_data/books_preprocessed.csv")
    descriptions = books_df["desc_clean"]
    vectorizer = TfidfVectorizer(analyzer='word', smooth_idf=True, use_idf=True, ngram_range=ngram_rng)
    vectorizer.fit_transform(descriptions)
    features = vectorizer.get_feature_names_out()

    books_df["keywords"] = descriptions.apply(lambda desc: get_keywords(vectorizer, features, desc))
    books_df['keywords'] = books_df['keywords'].apply(lambda row: TreebankWordDetokenizer().detokenize(row))
    books_df["keywords"] = books_df[["book_info", "keywords"]].fillna('').agg(' '.join, axis=1)

    books_df.to_csv("processed_data/books_added_keywords.csv", sep=",", index=False)

    return books_df


def exctract_description_keywords(desc_text, books_df, ngram_rng = (1,1)):
    desc_cleaned = clean_text(desc_text)
    descriptions = books_df["desc_clean"]
    vectorizer = TfidfVectorizer(analyzer='word', smooth_idf=True, use_idf=True, ngram_range=ngram_rng)
    vectorizer.fit_transform(descriptions)
    features = vectorizer.get_feature_names_out()
    result = []
    keywords_df = {}
    keywords_df['desc_text'] = desc_text
    keywords_df['top_keywords'] = get_keywords(vectorizer, features, desc_cleaned)
    result.append(keywords_df)
    return pd.DataFrame(result)