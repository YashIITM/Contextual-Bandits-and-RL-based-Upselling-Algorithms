import numpy as np
from pandas import *
from sklearn.preprocessing import normalize

class InconsistentNumberOfFeatures(Exception):
    """Exception raised for errors in the input features.

    """

    def __init__(self, message="Inconsistent Dimensions of feature vectors:"):
        self.message = message
        super().__init__(self.message)

def CustomContext(filename,list_of_features,d,K):
    """This function return context, as an K*d matrix, each row represents a context of action

    Args:
        filename: Name of custom data set
        list_of_features: list of names of features that we wanna incorporate for the algorithm
        d (int): Dimension of context
        K (int): Number of arms

    Returns:
        context: an np.ndarray whose shape is (K, d), each row represents a context
    """
    data = read_csv(filename)
    if len(list_of_features)!=d:
        raise InconsistentNumberOfFeatures()
    features=[]
    for feature in list_of_features:
        temp_array = np.array(data[feature].to_list())
        features.append(temp_array[0:K])
    context = normalize(np.array(features),axis = 1,norm='l2')

    return context


def SampleContext(d: int, K: int) -> np.ndarray:
    """This function return context, as an K*d matrix, each row represents a context of action

    Args:
        d (int): Dimension of context
        K (int): Number of arms

    Returns:
        context: an np.ndarray whose shape is (K, d), each row represents a context
    """
    context = np.random.normal(loc=0, scale=1, size=(K, d // 2))
    length = np.sqrt(np.sum(context * context, axis=1, keepdims=True))
    context = np.tile(context, (1, 2))
    length = np.tile(length, (1, d))
    context = context / length / np.sqrt(2)  # each column represent a context
    return context


def GetRealReward(context: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Given the context, return the realized reward

    Args:
        context (np.ndarray): An np.ndarray whose shape is (K, d), each column represents a context of an arm
        A (np.ndarray): The parameter of this reward function

    Returns:
        reward: an np.ndarray whose shape is (K,), reward = context^T A^T A context + N(0, 0.05^2)
    """
    if len(context.shape) == 1:
        return context.transpose().dot(A.transpose().dot(A)).dot(context) + np.random.normal(loc=0, scale=0.05)
        # return context.transpose().dot(A.transpose().dot(A)).dot(context)
    else:
        return np.diag(context.dot(A.transpose().dot(A)).dot(context.transpose())) + np.random.normal(loc=0, scale=0.05, size=context.shape[0])
