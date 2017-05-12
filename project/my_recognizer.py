import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_id in test_set.get_all_Xlengths():
        x, l = test_set.get_all_Xlengths()[word_id]
        probs = {}
        max_logL = float("-inf")
        recognized_word = None
        for key in models:
            try:
                probs[key] = models[key].score(x, l)
                if probs[key] > max_logL:
                    max_logL = probs[key]
                    recognized_word = key
            except Exception:
                probs[key] = float("-inf")
        probabilities += [probs]
        guesses += [recognized_word]
    return (probabilities, guesses)
