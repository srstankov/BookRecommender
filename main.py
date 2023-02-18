from preprocessing import clean_description, load_dataset, format_cleaned_description, clean_user_keywords, untokenize
from keywords import exctract_keywords_from_df
from recommending import calc_tfidf, recommend_book, print_tfidf, plot_cosine_sim
from user_interface import command_interface, evaluate_system, print_recommended_books
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# books_df = load_dataset()

# books_df = clean_description(books_df)

# print(books_df[["Description", "desc_clean"]].head(5))

# books_df = format_cleaned_description(books_df)

# print(books_df[["Name", "name_clean"]])
# print(books_df[["Id", "Publisher"]])
# print(books_df["book_info"])


# books_df = pd.read_csv("processed_data/books_preprocessed.csv")

# print((books_df[["Id", "Language", "book_info"]]))
# print((books_df[["Id", "Language", "desc_clean"]]))

# books_df = exctract_keywords_from_df(books_df)
# print(books_df[["desc_clean", "keywords"]])

command_interface()

# evaluate_system()

# books_df = pd.read_csv("processed_data/books_added_keywords.csv")

# print(books_df[["desc_clean", "keywords"]])
# print(books_df["keywords"][29990])

# print(books_df["book_info"][29990])

# print("")
# print("A Year of the Stars: A Month-By-Month Journey of Skywatching")
# print_recommended_books(recommend_book("A Year of the Stars: A Month-By-Month Journey of Skywatching", books_df, rec_size=5))
#
# print("")
# print("Data Structures and Algorithms in Java")
# print_recommended_books(recommend_book("Data Structures and Algorithms in Java", books_df, rec_size=7))
#
# print("")
# print("Death on the Nile (Hercule Poirot, #17)")
# print_recommended_books(recommend_book("Death on the Nile (Hercule Poirot, #17)", books_df, rec_size=5))
#
# print("")
# print("The Eastland Disaster (Images of America: Illinois)")
# print_recommended_books(recommend_book("The Eastland Disaster (Images of America: Illinois)", books_df, rec_size=5))
#
# print("")
# print("Poirot: The Complete Battles of Hastings, Vol. 1 (Hercule Poirot & Arthur Hastings Omnibus, #1)")
# print_recommended_books(recommend_book("Poirot: The Complete Battles of Hastings, Vol. 1 (Hercule Poirot & Arthur "
#                                        "Hastings Omnibus, #1)", books_df, rec_size=5))


# tfidf_vals = calc_tfidf(books_df)
# print(tfidf_vals)
# print(cosine_similarity(tfidf_vals, tfidf_vals))
# plot_cosine_sim(tfidf_vals)
