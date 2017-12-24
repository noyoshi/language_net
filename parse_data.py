#!/usr/bin/end python3.5

import numpy as np
import random


def parse_word(s):
    # this will return a hot list of the word in vector form
    # word is a list of lists, each letter is a list of 27 bits, representing
    # all characters, with the last as a space

    MAX_SZ = 40
    # max length of a word

    position_dict = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8,
    'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16,
    'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25
    }

    word_vector = np.zeros(1080)

    i = 0
    # print(s)
    # print(len(s))
    for c in s:
        # freaking word list are in form "> word" and sometimes have "r12309823sdf123"
        # as an entry.........gah. Need to FIX TODO FIX THE LISTS
        # however, since both lists are like this, and we are handling errors with the
        # bad try-except, the NN works!
        # temp = np.zeros(len(position_dict)+1)
        # print((i*26) + position_dict[c])
        try:
            word_vector[(i*26) + position_dict[c]] = 1 # setting the hot bit
        except:
            pass
            # print(c)
            # print(type(c))
            # print(type(s))
        # np.concatenate((word_vector,temp), axis=0)
        i += 1

    spaces = MAX_SZ - i

    for j in range(spaces):
        # temp = np.zeros(len(position_dict)+1)
        # temp[26] = 1 # last bit = 1, ie it is an empty space
        word_vector[((i+1)*26)] = 1
        i += 1
        # np.concatenate((word_vector,temp), axis=0)

    return word_vector
    # returns word vector in form:
    # [|1 0 0 ...|0 1 0 ...|0 0 1...|...1|...] = "abc"
    # one vector, the pipes are used for visual aid


def create_feature_sets_and_labels(lists, test_size = 0.1):
    data = []
    temp = []

    # creates the labels for the data, english will be [1,0], japanese will be
    # [0,1] if we feed in the data as ["japanese","english"] as lists

    english = [0, 1]
    japanese = [1, 0]
    labels = [english,japanese]

    j = 0
    for l in lists:
        print(l)
        with open(l, 'r') as temp_list:
            contents = temp_list.readlines()
            for word in contents:
                data.append([parse_word(word.strip().lower()), np.array(labels[j])])
            print(labels[j])
            j += 1

    random.shuffle(data)
    # for i in range(100):
    #     print(data[i])
    data = np.array(data, dtype=object)

    testing_size = int(test_size*len(data))


    train_x = list(data[:,0][:-testing_size])
    train_y = list(data[:,1][:-testing_size])

    test_x = list(data[:,0][-testing_size:])
    test_y = list(data[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y
    # data is a list of lists of strings, labels are the labels used to classify
    # the data


if __name__=='__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(['japanese_list.txt',
            'english_list.txt'])

    random.shuffle(test_x)
    for i in range(10):
        print(test_y[i])
