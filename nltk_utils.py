import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
    #   pass

# --- Tokenize Example ---
# a = "Berapa lama memesan makanan?"
# print(a)
# a = tokenize(a)
# print(a)

# --- Stemmed Word example ---
# words = ["Program", "programs", "programer", "programming"]
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)


# --- Bag Of Words example ---
#     sentence_words = [stem(word) for word in tokenized_sentence]
#     bag = np.zeros(len(words), dtype=np.float32)
#     for idx, w in enumerate(words):
#         if w in sentence_words: 
#             bag[idx] = 1

#     return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "saya", "kamu", "bye", "terimakasih", "sehat"]
# bag = bag_of_words(sentence, words)
# print(bag)