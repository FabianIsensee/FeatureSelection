from copy import deepcopy
from multiprocessing.pool import Pool
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score


def use_this_for_accuracy_maximization(gt, pred):
    return -accuracy_score(gt, pred)


def evaluation_function(classifier_or_regressor, data, target, selected_features=None, additional_norm_fn=None,
                        scoring_function=mean_absolute_error, n_folds=3, size_penalty=0):
    """

    :param classifier_or_regressor: must implement a .fit(data, target) and a .predict(data) method
    :param data: NxD np.ndarray
    :param target: N np.ndarray
    :param selected_features: list of int, points to indices in D (feature dimension of data)
    :param additional_norm_fn: ignore this
    :param scoring_function: must compare a prediction with a ground truth scoring_function(gt, pred) and return a
    scalar. This scalar will be minimized! (If your scoring is a classification accuracy then you need to negate the
    output to convert the maximization of the accuracy into a minimization problem - see use_this_for_accuracy_maximization)
    :param n_folds:
    :param size_penalty: float > 0, drives the algorithm to select smaller feature sets. Default=0 -> no size penalty.
    What size to set depends on the application and the scoring function. Using all features will add a penalty of
    size_penalty to the score computed by scoring_function. Using half the features will add size_penalty/2 etc.
    :return:
    """
    num_features_total = data.shape[1]

    splitter = KFold(n_splits=n_folds, shuffle=False, random_state=12345)
    splits = splitter.split(data)
    if selected_features is None:
        selected_features = list(range(data.shape[1]))

    selected_features = deepcopy(selected_features)
    selected_features.sort()

    data = np.copy(data[:, selected_features])

    # print("pre norm: ", np.mean(data, 0))
    if additional_norm_fn is not None:
        data = additional_norm_fn(data)
    # print("post norm: ", np.mean(data, 0))

    scores = []
    for tr, te in splits:
        classifier_or_regressor.fit(data[tr], target[tr])
        res = classifier_or_regressor.predict(data[te])
        score = scoring_function(target[te], res)
        scores.append(score)

    size_pen = len(selected_features) / float(num_features_total) * size_penalty

    return np.mean(scores) + size_pen


def evaluate_set(args):
    eval_fct, data, target, feature_set = args
    return eval_fct(data, target, feature_set)


def multithreaded_run_evaluation_function(data, target, examine_these, num_threads=8, eval_fn=evaluation_function):
    """
    Can be used to run an evaluation function on several feature sets in a parallel way. Not used by this package but
    may be useful to you
    :param data:
    :param target:
    :param examine_these:
    :param num_threads:
    :param eval_fn:
    :return:
    """
    p = Pool(num_threads)
    scores = p.map(evaluate_set, zip([eval_fn] * len(examine_these), [data] * len(examine_these),
                                     [target] * len(examine_these), examine_these))
    p.close()
    p.join()
    return scores
