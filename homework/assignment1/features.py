from collections import OrderedDict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np
import math

class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        # raise NotImplementedError
        word_freq = {}
        for token in y:
          if token not in word_freq.keys():
            word_freq[token] = 1
          else:
            #print('here!')
            word_freq[token] += 1
        # sort by value
        word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
        # convert into list to use index
        self.bow = list(word_freq.keys())

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """
        word_freq = {}
        for token in text.split():
          if token not in word_freq.keys():
            word_freq[token] = 1
          else:
            #print('here!')
            word_freq[token] += 1
        # sort by value
        word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
        
        # string to list of words
        word_list = text.split()
        result = []
        for word in self.bow:
          if word not in word_list:
            result.append(0)
          else:
            result.append(word_freq[word])
              
        #result = list(word_freq.values())
        #raise NotImplementedError
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # raise NotImplementedError
        reviewTFDict = OrderedDict()
        x = ' '.join(X).split()
        for word in x:
            if word in reviewTFDict:
                reviewTFDict[word] += 1
            else:
                reviewTFDict[word] = 1
        # Computes tf for each word
        for word in reviewTFDict:
            reviewTFDict[word] = reviewTFDict[word] / len(x)
        #self.bow = list(reviewTFDict.keys())
        self.idf = reviewTFDict
        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """
        '''reviewTFDict = {}
        x = text.split()
        for word in x:
            if word in reviewTFDict:
                reviewTFDict[word] += 1
            else:
                reviewTFDict[word] = 1
        # Computes tf for each word
        for word in reviewTFDict:
            reviewTFDict[word] = reviewTFDict[word] / len(x)'''
            # print(word, ":", reviewTFDict[word])
            #print('separator')
        #print(len(list(reviewTFDict.keys())))

        # string to list of words
        word_list = text.split()
        result = []
        alpha = 1

        countDict = {}
        # Run through each review's tf dictionary and increment countDict's (word, doc) pair
        for word in word_list:
          if word in countDict:
            countDict[word] += 1
          else:
            countDict[word] = 1
        idfDict = {}
        for word in countDict:
            idfDict[word] = math.log(self.k / (countDict[word] + alpha))
       
        for word in self.idf:
            if word in idfDict:
                result.append(idfDict[word])
            else:
                result.append(0)
        # print(result)
        #raise NotImplementedError
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
