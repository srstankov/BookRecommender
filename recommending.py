from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt


def calc_tfidf(books_df):
    vectorizer = TfidfVectorizer(analyzer = 'word',
                                min_df=3,
                                max_df = 0.6,
                                stop_words="english",
                                encoding = 'utf-8',
                                token_pattern=r"(?u)\S\S+")
    tfidf_vals = vectorizer.fit_transform(books_df["keywords"])
    # tfidf_df = pd.DataFrame(tfidf_vals.toarray(), index=books_df["Name"], columns=vectorizer.get_feature_names_out())
    return tfidf_vals

def print_tfidf(tfidf_df):
    tfidf_df_preview = tfidf_df.iloc[100:150, 25:].copy()
    tfidf_df_preview = tfidf_df_preview.stack().reset_index()
    tfidf_df_preview = tfidf_df_preview.rename(columns={0: 'tfidf', 'Name': 'book', 'level_1': 'term'})
    tfidf_df_preview = tfidf_df_preview.sort_values(by=['book', 'tfidf'], ascending=[True, False]).groupby(['book']).head(10)
    tfidf_df_preview = tfidf_df_preview.term.str.replace('_', ' ')
    tfidf_df_preview = tfidf_df_preview[tfidf_df_preview["tfidf"] > 0]
    # print(tfidf_df_preview['tfidf'])
    print(tfidf_df_preview.iloc[0:5])

def recommend_book(example_book_name, books_df, rec_size = 10):
    tfidf_vals = calc_tfidf(books_df)
    book_cosine_similarity = cosine_similarity(tfidf_vals, tfidf_vals)
    book_names_series = pd.Series(books_df["Name"])
    example_book_ind = book_names_series[book_names_series == example_book_name].index[0]
    most_similar_books = list(pd.Series(book_cosine_similarity[example_book_ind]).sort_values(ascending = False).iloc[1:rec_size+1].index)
    if bool(most_similar_books):
        recommendations = [book_names_series[b] for b in most_similar_books]
    else:
        recommendations = []
    return recommendations

def plot_cosine_sim(tfidf_vals):
    cos_sim = cosine_similarity(tfidf_vals, tfidf_vals)
    print(cos_sim)
    plt.figure(figsize=(6, 6), dpi=80)
    plt.spy(cos_sim, precision=0.1, markersize=0.04)
    plt.tight_layout()
    plt.show()


