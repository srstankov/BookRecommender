import pandas as pd
import numpy
import re
import string
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


def load_dataset(dataset_path = "dataset/book1000k-1100k.csv"):
    books_df = pd.read_csv(dataset_path)
    return books_df

def remove_non_ascii_chars(text):
    return ''.join(c for c in text if  ord(c)<128)

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_html(text):
    return re.sub('<[^<]+?>', ' ', text)

def remove_punctuation(text):
    formatted_text = "".join([p for p in text if p not in string.punctuation])
    return formatted_text

def remove_empty_strings(text):
    text = [s for s in text if len(s) > 0]
    return text

def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    result_text = [w for w in text if w not in stopwords]
    return result_text

def transform_short_forms(text):
    text = text.replace(r"can't", 'can not')
    text = text.replace(r"cannot", 'can not')
    text = text.replace(r"'m", ' am')
    text = text.replace(r"'re", ' are')
    text = text.replace(r"n't", ' not')
    text = text.replace(r"'ll", ' will')
    text = text.replace(r"'s", ' is')
    text = text.replace(r"'ve", ' have')
    return text

def tokenize(text):
    tokenized_text = [word for word in text.split(" ")]
    return tokenized_text

def untokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(text)
    lemmatized_text = [wordnet_lemmatizer.lemmatize(pt[0], get_wordnet_pos(pt[1])) for pt in pos_tags]
    return lemmatized_text

def clean_text(text):
    text = text.lower()
    text = remove_non_ascii_chars(text)
    text = remove_url(text)
    text = remove_html(text)
    text = transform_short_forms(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = remove_empty_strings(text)
    text = lemmatize_text(text)
    return text

def clean_user_keywords(text):
    text = remove_stopwords(text)
    text = remove_empty_strings(text)
    text = lemmatize_text(text)
    return text

def clean_description(books_df):
    books_df.dropna(subset=["Description"], inplace=True)
    books_df["desc_clean"] = books_df["Description"].apply(lambda d: clean_text(d))
    return books_df

def remove_duplicate_book_versions(books_df):
    sorted_by_publisher_books_df = books_df.sort_values(by="Publisher", na_position='last')
    removed_dup_versions_books_df = sorted_by_publisher_books_df.drop_duplicates(subset=["Name", "Authors",
                                                                                         "Description"], keep='first')
    return removed_dup_versions_books_df

def extract_series_from_name(name):
    search_pattern = "(?:[;]\s*|\(\s*)([^\(;]*\s*#\s*\d+(?:\.?\d+|\\&\d+|-?\d*))"
    found_pattern = re.findall(search_pattern, name)
    if not found_pattern:
        return numpy.nan
    else:
        series = " ".join([s.replace(" ", "_") for s in found_pattern])
        return series

def clean_book_name(books_df):
    series_pattern = re.compile("(?:[\(]\s*[^\(;]*\s*#\s*\d+(?:\.?\d+|\\&\d+|-?\d*)(?:;|\))|\s*[^\(;]*\s*#\s*\d+(?:\.?\d+|\\&\d+|-?\d*)\))")
    books_df["name_clean"] = books_df["Name"].str.replace(series_pattern, r'').str.strip()
    return books_df

def format_cleaned_description(books_df):
    pd.options.mode.chained_assignment = None
    books_df["desc_clean_length"] = [len(desc) for desc in books_df["desc_clean"]]
    books_df = books_df[books_df["desc_clean_length"] > 3]

    books_df["Publisher"] = books_df["Publisher"].replace("unknown", numpy.nan)
    books_df["Publisher"] = books_df["Publisher"].str.replace('"', '')
    books_df["Publisher"] = books_df["Publisher"].str.strip().str.replace(' ', '_')
    books_df["Authors"] = books_df["Authors"].str.strip().str.replace(' ', '_')

    books_df = remove_duplicate_book_versions(books_df)
    books_df["book_series"] = books_df["Name"].apply(lambda name: extract_series_from_name(name))
    books_df = clean_book_name(books_df)

    books_df["book_info"] = books_df[["book_series", "Authors", "Publisher"]].fillna('').agg(' '.join, axis=1)


    books_df["book_info"] = books_df["book_info"].apply(lambda i: tokenize(i))
    books_df["book_info"] = books_df["book_info"].apply(lambda k: remove_empty_strings(k))

    books_df['desc_clean'] = books_df['desc_clean'].apply(lambda row: TreebankWordDetokenizer().detokenize(row))
    books_df['book_info'] = books_df['book_info'].apply(lambda row: TreebankWordDetokenizer().detokenize(row))

    books_df = books_df.sort_values("Id", ascending=True)
    books_df.to_csv("processed_data/books_preprocessed.csv", sep=",", index=False)

    return books_df
