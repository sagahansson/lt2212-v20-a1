import os
# import sys
import pandas as pd
import numpy as np
import glob
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def get_tokenized_files(directory, directory2):
    """


    Loads both directories and returns them as tokenized lists in tuples, where
    each tuple represents a file. The tuples also contains directory name and
    filename. Also returns a list of all the words in all files.


    Parameters
    ----------
    directory : Name of first directory where files to be processed are
    located. Type: str.

    directory2 : Name of second directory where files to be processed are
    located. Type: str.


    Returns
    -------
    A tuple of all_files and all_words.

    all_files: a list of tuples, where each tuple represents a file. The tuple
    contains the name of the directory and the name of the file as strings, and
    a list of strings representing the file content.  Each string is a
    word/punctuation from the file.

    all_words: a list of strings, representing the content of all files from
    directory and directory2. Each string is a word/punctuation from the files.

    """
    all_files = []
    all_words = []
    dir1_files = glob.glob('{}/*.txt'.format(directory))
    dir2_files = glob.glob('{}/*.txt'.format(directory2))

    both_dirs_files = dir1_files
    both_dirs_files.extend(dir2_files)

    for filename in both_dirs_files:
        words = []
        with open(filename, 'r') as f:
            if os.path.dirname(filename).endswith(os.path.basename(directory)):
                basename_directory = os.path.basename(directory)
                basename_filename = os.path.basename(filename)
            else:
                basename_directory = os.path.basename(directory2)
                basename_filename = os.path.basename(filename)

            for line in f:
                words.extend(line.split())
        all_files.append((basename_directory, basename_filename, words))
        all_words.extend(words)

    return all_files, all_words


def filter_corpus_words(directory, directory2, n):
    """

    Takes all_words from the output of get_tokenized_files, and creates a
    dictionary contains the unique words in all_words and their frequency in
    directory and directory2, if their frequency is higher than n.


    Parameter
    ----------
    directory : Name of first directory where files to be processed are
    located. Type: str.

    directory2 : Name of second directory where files to be processed are
    located. Type: str.

    n : an integer representing the minimum count of a word in the entire
    corpus. Type: int.

    Returns
    -------
    filtered_words : a dictionary of all words that appear in directory and
    directory2. Each key is a word, and each value is an integer representing
    the frequency of that word in directory and directory2. Type: dict.

    """
    corpus = get_tokenized_files(directory, directory2)[1]
    lowered_corpus = []
    for word in corpus:
        lowered_corpus.append(word.lower())
    counted_words = Counter(lowered_corpus)

    filtered_words = {}

    for item in counted_words.items():
        if item[1] <= n:
            del counted_words[item]
        else:
            if item[0].isalpha():
                filtered_words[item[0]] = item[1]
            else:
                del counted_words[item]
    return filtered_words


def filter_article_words(directory, directory2, n):
    """

    Filters the words in all_files from the output of get_tokenized_files with
    respect to the output of filter_corpus_words, i.e. if a word is in the
    output of filter_corpus_words, it will be in the output of
    filter_article_words.


    Parameters
    ----------
    directory : Name of first directory where files to be processed are
    located. Type: str.

    directory2 : Name of second directory where files to be processed are
    located. Type: str.

    n : an integer representing the minimum count of a word in the entire
    corpus. Type: int.

    Returns
    -------
    all_files_filtered: a list of lists, where each inner list represents a
    file from directory or directory2. Each inner list contains a tuple,
    containing the directory name and the file name as strings, and a list.
    The list contains the words in that file that appear more than n times
    through all files. Type: list.

    """
    all_files = get_tokenized_files(directory, directory2)[0]
    filtered_words = list(filter_corpus_words(directory, directory2, n))
    all_files_filtered = []

    for file in all_files:
        article_words = file[2]
        a_file = []
        lowered_words = []
        for word in article_words:
            word = word.lower()
            if word not in filtered_words:
                pass
            else:
                lowered_words.append(word.lower())
        a_file.insert(0, (file[0], file[1]))
        a_file.append(lowered_words)
        all_files_filtered.append(a_file)

    return all_files_filtered


def get_word_counts(directory, directory2, n):
    """

    Gets frequency of each word (that occurs more than n times) in each file in
    each directory.

    Parameters
    ----------
    directory : Name of first directory where files to be processed are
    located. Type: str.

    directory2 : Name of second directory where files to be processed are
    located. Type: str.

    n : an integer representing the minimum count of a word in the entire
    corpus. Type: int.


    Returns
    -------
    all_files_dicts : a list of tuples, where each tuple represents a file from
    directory or directory2. Each tuple contains another tuple and a Counter
    object (dict subclass). The tuple contains directory name and file name as
    strings. The Counter represents the frequency of each word (that occurs 
    more than n times) in the file, where the key is a word and the value is 
    the count.


    """
    all_files = filter_article_words(directory, directory2, n)
    all_files_dicts = []

    for file in all_files:
        count = Counter(sorted(file[1]))
        all_files_dicts.append((file[0], count))

    return all_files_dicts

def add_all_words(directory, directory2, n):
    """

    Makes each Counter object contain the same keys (words). Keys that are
    added are given the value 0, as that is their frequency.

    Parameter
    ----------
    directory : Name of first directory where files to be processed are
    located. Type: str.

    directory2 : Name of second directory where files to be processed are
    located. Type: str.

    n : an integer representing the minimum count of a word in the entire
    corpus. Type: int.

    Returns
    -------
    allwords_in_alldicts : a list of tuples, where each tuple represents a file
    from directory or directory2. Each tuple contains a Counter object (dict
    subclass) and another tuple. The tuple contains directory name and file
    name as strings. The Counter represents the frequency of each word (that
    occurs more than n times) in the file, where the key is a word and the
    value is the count.

    """
    filtered_words = list(filter_corpus_words(directory, directory2, n))
    all_files_dicts = get_word_counts(directory, directory2, n)
    allwords_in_alldicts = []

    for file_tpl in all_files_dicts:

        art_dict = file_tpl[1]
        file_info = file_tpl[0]
        for word in filtered_words:
            if word not in art_dict.keys():
                art_dict[word] = 0
            else:
                pass
        allwords_in_alldicts.append((art_dict, file_info))

    return allwords_in_alldicts

def turn_dictvals_to_lists(directory, directory2, n):
    """

    Turns the values of each Counter object in the output of add_all_words into
    lists.


    Parameter
    ----------
    directory : Name of first directory where files to be processed are
    located. Type: str.

    directory2 : Name of second directory where files to be processed are
    located. Type: str.

    n : an integer representing the minimum count of a word in the entire
    corpus. Type: int.

    Returns
    -------
    all_wc_lists : a list of tuples, where each tuple represents a file
    from directory or directory2. Each tuple contains a list and another tuple.
    The tuple contains directory name and file name as strings. The list
    contains represents the frequency of each word (that occurs more than n
    times) in the file.

    """
    all_words_in_alldicts = add_all_words(directory, directory2, n)
    all_wc_lists = []
    all_elems = []

    for file_tpl in all_words_in_alldicts:
        art_dict = file_tpl[0]
        file_info = file_tpl[1]
        wc_list = []
        elems = []

        for keyval in sorted(art_dict.items()):
            wc_list.append(keyval[1])
            elems.append(keyval)
        all_wc_lists.append((wc_list, file_info))
        all_elems.append(elems)
    return all_wc_lists


def fileinfo_to_countlists(directory, directory2, n):
    """

    Adds file name and directory name to each list of word counts and
    'Filename' and 'Directory' to the output of filter_corpus_words.


    Parameter
    ----------
    directory : Name of first directory where files to be processed are
    located. Type: str.

    directory2 : Name of second directory where files to be processed are
    located. Type: str.

    n : an integer representing the minimum count of a word in the entire
    corpus. Type: int.

    Returns
    -------
    sorted_total_vocab : A list containing 'Filename' and 'Directory' and all
    unique words that occur in a file in either directory or directory2. Words
    are sorted alphabetically. Type: list.

    wc_lists_info : a list of lists, where each inner list represents a file
    from either directory or directory2. Each list contains the files name and
    directory name as strings, and the frequencies of the words in
    sorted_total_vocab. Type: list.

    """
    sorted_wc_lists = turn_dictvals_to_lists(directory, directory2, n)
    sorted_total_vocab = list(sorted(filter_corpus_words(directory, directory2, n)))

    sorted_total_vocab.insert(0, 'Filename')
    sorted_total_vocab.insert(1, 'Directory')

    wc_lists_info = []

    for file_tpl in sorted_wc_lists:
        count_list = file_tpl[0]
        file_info = file_tpl[1]
        count_list.insert(0, file_info[1])
        count_list.insert(1, file_info[0])
        wc_lists_info.append(count_list)

    return sorted_total_vocab, wc_lists_info


def part1_load(folder1, folder2, n=1):
    """
    Takes the output of fileinfo_to_countlists and turns it into a dataframe.

    Parameter
    ----------
    folder1 : Name of first directory where files to be processed are
    located. Type: str.

    folder2 : Name of second directory where files to be processed are
    located. Type: str.

    n : an integer representing the minimum count of a word in the entire 
    corpus. The default is 1. Type: int., optional.
    

    Returns
    -------
    dataframe : a dataframe where file name, directory and words are column
    labels. Each row contains the specific files name, which directory it comes
    from and counts of each word in the column labels.
    Type: pandas.core.frame.DataFrame.

    """
    data = fileinfo_to_countlists(folder1, folder2, n)[1]
    columns = fileinfo_to_countlists(folder1, folder2, n)[0]
    dataframe = pd.DataFrame(data, columns=columns)

    return dataframe

def part2_vis(df, m=1):
    """
    Returns a bar chart containing information from a dataframe. M is an
    integer representing the maximum number of bars to appear in the chart.

    Parameters
    ----------
    df : a dataframe. Type: pandas.core.frame.DataFrame.

    m : a value representing the maximum number of words to appear
        in the diagram. The default is 1. Type: int., optional.

    Returns
    -------
        A bar chart of the top m term frequences in the DataFrame
        with matching bars per class.  That is, if "very" is one of the
        selected top-occurring words, have two bars side-by-side in the chart
        for "very" with different colours, one for the first class and the
        other for the second. Type: matplotlib.axes._subplots.AxesSubplot.

    """
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)
    df_grouped = df.groupby(['Directory']).sum()
    df_transposed = df_grouped.T
    idx = df_transposed.sum(axis=1).sort_values(ascending=False).head(m).index
    df = df_transposed.loc[idx]
    return df.plot(kind="bar")


def part3_get_tf(df):
    """
    Gets term frequency for each word in each file.

    Parameters
    ----------
    df : a dataframe. Type: pandas.core.frame.DataFrame.

    Returns
    -------
    tf_dicts : a list of dictionaries, where one dictionary represents one
    file. In each dictionary, the keys are tuples containing file name,
    directory name and word as strings. The values are term frequencies.
    Type: list.

    """
    data = df.values
    column_labels = df.columns    
    words = column_labels[2:]

    tf_dict = {}
    tf_dicts = []

    for file_list in data:
        tf_dict = {}
        just_wc = file_list[2:]
        for i, count in enumerate(just_wc):
            word = words[i]
            file_info = list(file_list[:2])
            file_info.append(word)
            key = tuple(file_info)
            tf_dict[key] = count
        tf_dicts.append(tf_dict)

    return tf_dicts


def part3_get_idf(df):
    """

    Gets the inverse document frequency for each word.


    Parameters
    ----------
    df : a dataframe. Type: pandas.core.frame.DataFrame.


    Returns
    -------
    idf_dict : a dictionary, where the keys are words and the values are
    inverse document frequencies. Type: dict.

    """
    column_labels = df.columns
    column_dict = {}
    column_dicts = []
    file_count = df.shape[0]
    idf_dict = {}

    for column in column_labels:
        column_list = df[column].tolist()
        column_dict[column] = column_list
    column_dicts.append(column_dict)

    for column_dict in column_dicts:
        for key, val in column_dict.items():
            idf_dict[key] = np.log(file_count/(np.count_nonzero(val)))

    return idf_dict


def part3_tfidf(df):
    """

    Takes the outputs of part3_get_tf and part3_get_idf, calculates the tf-idf
    for each word in each file and puts the values into a Pandas DataFrame.

    Parameters
    ----------
    df : a dataframe. Type: pandas.core.frame.DataFrame.

    Returns
    -------
    dataframe : a pandas dataframe, where the columns are filename, directory
    and words. Each row represents a file, and contains its file name, its
    directory and the tf-idf for each word in that file.

    """
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    tf_dicts = part3_get_tf(df)
    idf_dict = part3_get_idf(df)
    column_labels = df.columns.tolist()

    tf_idf_data = []

    for file_dict in tf_dicts:
        tf_idf_row = []
        for file_word, tf in sorted(file_dict.items()):
            filename = file_word[0]
            directory = file_word[1]
            word = file_word[2]
            if filename in tf_idf_row:
                if directory in tf_idf_row:
                    tf_idf_row.append(tf*idf_dict[word])
            else:
                tf_idf_row.insert(0, filename)
                tf_idf_row.insert(1, directory)
                tf_idf_row.append(tf*idf_dict[word])
        tf_idf_data.append(tf_idf_row)

    dataframe = pd.DataFrame(tf_idf_data, columns=column_labels)
    return dataframe


def part_bonus(df):
    """
    Runs K nearest neibours on a dataframe, and prints the accuracy score.

    Parameters
    ----------
    df : a dataframe. Type: pandas.core.frame.DataFrame.

    Returns
    -------
    None. (Prints accuracy score.)

    """
    x = df.drop(['Directory', 'Filename'], axis=1)
    y = df['Directory']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    
    print(accuracy_score(y_test, y_pred))
