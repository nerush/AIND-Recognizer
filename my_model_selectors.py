import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            states = range(self.min_n_components, self.max_n_components + 1)
            _, model = max(map(lambda n: self.select_model(n), states))
            return model
        except:
            return self.base_model(self.n_constant)

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select_model(self, num_states):
        """
            Calculate the BIC score
        """
        model = self.base_model(num_states)
        logL = model.score(self.X, self.lengths)

        N, num_features = self.X.shape
        p = num_states**2 + 2*num_features*num_states-1

        logN = np.log(N)
        BIC = -2 * logL + p * logN
        return BIC, model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select_model(self, num_states):
        model = self.base_model(num_states)
        logL = model.score(self.X, self.lengths)

        scores = [model.score(x, l) for i, (x, l) in self.hwords.items() if i != self.this_word]
        DIC = logL - sum(scores)/(len(scores) - 1)
        return DIC, model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select_model(self, num_states):
        nos = math.floor(math.log(len(self.sequences), 2) + 1)
        folds = KFold(n_splits=nos)
        scores = []

        for train_idx, test_idx in folds.split(self.sequences):
            self.X, self.lengths = combine_sequences(train_idx, self.sequences)

            model = self.base_model(num_states)

            x, l = combine_sequences(test_idx, self.sequences)

            score = model.score(x, l)
            scores.append(score)

        return np.mean(scores), model