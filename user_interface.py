from recommending import recommend_book
from preprocessing import clean_user_keywords, untokenize
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def recommend_by_keywords():
    books_df = pd.read_csv("processed_data/books_added_keywords.csv")
    kw = keywords_interface()
    if len(kw) == 0:
        return []
    kw = clean_user_keywords(kw)
    kw = untokenize(kw)
    books_df = books_df.append({'Name': "user_keywords", "keywords": kw}, ignore_index=True)
    recommended_books = recommend_book("user_keywords", books_df)
    return recommended_books


def recommend_by_book_names():
    books_df = pd.read_csv("processed_data/books_added_keywords.csv")
    user_book_names = book_names_interface()
    names_length = len(user_book_names)
    rec_size_for_name = 10

    if names_length == 0:
        return []
    elif names_length == 1:
        rec_size_for_name = 10
    elif names_length == 2:
        rec_size_for_name = 5
    elif names_length == 3:
        rec_size_for_name = 3
    elif names_length == 4 or names_length == 5:
        rec_size_for_name = 2
    elif names_length > 5:
        rec_size_for_name = 1

    if names_length > 0:
        recommended_books = []
        for book_name in user_book_names:
            try:
                recommended_books += recommend_book(book_name, books_df, rec_size_for_name)
            except IndexError:
                recommended_books += ['The book name from the input was not found in the dataset!']
        return recommended_books


def command_interface():
    print("Welcome to BookRecommender!")
    print("Write command 'keywords' to use keywords for recommending books.")
    print("Write command 'books' to use favourite book names for recommending books.")
    print("Enter command:")
    command = input()
    if command == "books":
        recommended_books = recommend_by_book_names()
        if len(recommended_books) > 0:
            print_recommended_books(recommended_books)
        else:
            print("No book names have been written!")

    elif command == "keywords":
        recommended_books = recommend_by_keywords()
        if len(recommended_books) > 0:
            print_recommended_books(recommended_books)
        else:
            print("No keywords detected!")
    else:
        print("Command not recognised!")


def keywords_interface():
    print("Write a few keywords (each on a new line!) about the book that you search for.")
    print("Note that if you want specific author as a keyword you should write it with capital first letters and _ for space")
    print("for example: Agatha_Christie")
    print("Enter keywords (each on a new line):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    if len(lines) == 0:
        return []
    else:
        user_input = '\n'.join(lines)
        keywords = user_input.split('\n')
        return keywords


def book_names_interface():
    print("Each book name should be on a new line.")
    print("It should also be exactly the same (including capital letters and special symbols) as in the dataset.")
    print("Check the files in dataset to view book names.")
    print("Enter the names of the books that you like:")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    if len(lines) == 0:
        return []
    else:
        user_input = '\n'.join(lines)
        book_names = user_input.split('\n')
        return book_names

def print_recommended_books(recommended_books):
    print("Recommended books:")
    for rec_book in recommended_books:
        print(rec_book)

def evaluate_system():
    books_df = pd.read_csv("processed_data/books_added_keywords.csv")
    # exeriment 1
    expected_recs = ["Poirot: The Complete Battles of Hastings, Vol. 2 (Hercule Poirot & Arthur Hastings Omnibus, #2)",
                     "Poirot: Four Classic Cases",
                     "Poirot: The War Years",
                     "Poirot: The Post War Years",
                     "The Big Four",
                     "Murder on the Links: A BBC Radio 4 Full-Cast Dramatisation",
                     "Agatha Christie’s Poirot: The Life and Times of Hercule Poirot",
                     "Shades of Death: A Mystery",
                     "Death on the Nile (Hercule Poirot, #17)",
                     "What Mrs. McGillicuddy Saw!"]

    recommended = recommend_book("Poirot: The Complete Battles of Hastings, Vol. 1 (Hercule Poirot & Arthur Hastings Omnibus, #1)"
                                 , books_df)

    f1_1 = f1_score(expected_recs, recommended, average="macro").round(3)
    accuracy_1 = accuracy_score(expected_recs, recommended).round(3)
    print("")
    print("Experiment 1")
    print("F1_1 score: ", f1_1)
    print("Accuracy_1: ", accuracy_1)

    #exeriment 2
    expected_recs = ['Forgotten Chicago (Images of America: Illinois)',
                     "Chicago's Southeast Side (Images of America: Illinois)",
                     'Japanese Americans in Chicago (Images of America: Illinois)',
                     'Greeneville (Images of America: Tennessee)',
                     "San Francisco's Portola (Images of America: California)",
                     "Oakland (Images of America: Pennsylvania)",
                     'Jewish Chicago: A Pictorial History (Images of America: Illinois)',
                     "Altgeld's America",
                     '100 Best Cruise Vacations',
                     'Mount Carmel and Queen of Heaven Cemeteries (Images of America: Illinois)']

    recommended = recommend_book(
        "The Eastland Disaster (Images of America: Illinois)", books_df)

    f1_2 = f1_score(expected_recs, recommended, average="macro").round(3)
    accuracy_2 = accuracy_score(expected_recs, recommended).round(3)
    print("")
    print("Experiment 2")
    print("F1_2 score: ", f1_2)
    print("Accuracy_2: ", accuracy_2)

    # exeriment 3
    expected_recs = ['Java Programming Fundamentals: Problem Solving Through Object Oriented Analysis and Design',
                     'Parallel Programming Using C++',
                     'Data Structures & Other Objects Using Java',
                     'Java & XML Data Binding',
                     "Java in a Nutshell, Deluxe Edition",
                     'Programming in C++: Lessons and Applications',
                     'Object-Oriented Programming with Visual Basic .NET',
                     'Object Oriented Systems Analysis and Design',
                     'Eclipse Web Tools Platform: Developing Java Web Applications',
                     'TCP/IP Sockets in Java: Practical Guide for Programmers']

    recommended = recommend_book(
        "Data Structures and Algorithms in Java", books_df)

    f1_3 = f1_score(expected_recs, recommended, average="macro").round(3)
    accuracy_3 = accuracy_score(expected_recs, recommended).round(3)
    print("")
    print("Experiment 3")
    print("F1_3 score: ", f1_3)
    print("Accuracy_3: ", accuracy_3)

    #experiment 4
    expected_recs = ['Data Structures and Algorithms in Java',
                     'Unix for the Mainframer: The Essential Reference for Commands, Conversions, TCP/IP',
                     'The Practice of Programming (Addison-Wesley Professional Computing Series)',
                     'Handbook of Computer Vision Algorithms in Image Algebra',
                     'Oracle9i Unix Administration Handbook',
                     'Extended Prelude to Programming: Concepts & Design',
                     'UNIX & Linux Answers!: Certified Tech Support',
                     'TCP/IP Sockets in Java: Practical Guide for Programmers',
                     'Java Programming Fundamentals: Problem Solving Through Object Oriented Analysis and Design',
                     'Introduction to Microcontrollers: Architecture, Programming, and Interfacing for the Freescale 68hc12']

    kw = ['programming', 'unix', 'java', 'computers', 'algorithms']
    kw = clean_user_keywords(kw)
    kw = untokenize(kw)
    books_df = books_df.append({'Name': "user_keywords_1", "keywords": kw}, ignore_index=True)
    recommended = recommend_book("user_keywords_1", books_df)
    f1_4 = f1_score(expected_recs, recommended, average="macro").round(3)
    accuracy_4 = accuracy_score(expected_recs, recommended).round(3)
    print("")
    print("Experiment 4")
    print("F1_4 score: ", f1_4)
    print("Accuracy_4: ", accuracy_4)

    # experiment 5
    expected_recs = ['The Essentials of Astronomy (Essentials)',
                     'The Living Cosmos: Our Search for Life in the Universe',
                     'A Different Approach to Cosmology',
                     'Lifting Titan\'s Veil: Exploring the Giant Moon of Saturn',
                     "Epitome of Copernican Astronomy and Harmonies of the World",
                     'Infinite Worlds: An Illustrated Voyage to Planets beyond Our Sun',
                     'Opening the Doors of Heaven: Unlocking the Mysteries of the Cosmos',
                     'A Year of the Stars: A Month-By-Month Journey of Skywatching',
                     'A Glorious Accident: Understanding Our Place in the Cosmic Puzzle',
                     'In Search Of Planet Vulcan: The Ghost In Newton\'s Clockwork Universe']

    kw = ['universe', 'cosmos', 'astronomy', 'planets', 'stars']
    kw = clean_user_keywords(kw)
    kw = untokenize(kw)
    books_df = books_df.append({'Name': "user_keywords_2", "keywords": kw}, ignore_index=True)
    recommended = recommend_book("user_keywords_2", books_df)
    f1_5 = f1_score(expected_recs, recommended, average="macro").round(3)
    accuracy_5 = accuracy_score(expected_recs, recommended).round(3)
    print("")
    print("Experiment 5")
    print("F1_5 score: ", f1_5)
    print("Accuracy_5: ", accuracy_5)

    f1 = (f1_1 + f1_2 + f1_3 + f1_4 + f1_5) / 5
    accuracy = (accuracy_1 + accuracy_2 + accuracy_3 + accuracy_4 + accuracy_5) / 5
    print("")
    print("Overall summary")
    print("F1 score: ", f1.round(3))
    print("Accuracy: ", accuracy.round(3))