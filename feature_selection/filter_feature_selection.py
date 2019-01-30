# This code is largely copied from https://github.com/ilastik/ilastik-feature-selection of which I (Fabian) am the
# original author. The code here was adapted to support python multiprocessing and can therefore run a lot faster than
# the original implementation (provided you have the required hardware)

from multiprocessing.pool import Pool
from sklearn.metrics import mutual_info_score
import numpy as np
import logging


logger = logging.getLogger(__name__)


class FilterFeatureSelection(object):
    def __init__(self, X, Y, method="ICAP", num_threads=8):
        """
        This class provides easy access to mutual information based filter feature selection.
        The default mutual information estimation algorithm used is the histogram binning method. If a more
        sophisticated approach is required, use the change_MI_estimator function to apply your own method.

        IMPORTANT! For regression problems we recommend you use Kraskov's nearest neighbor MI estimator! You can get
        it here: https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py
        Force FilterFeatureSelection to use it via:
        FilterFeatureSelection._change_mi_method(mi)
        FilterFeatureSelection._change_cmi_method(cmi)

        :param X: (n_samples, n_features) numpy array containing the training data
        :param Y: (n_samples) numpy array containing target labels
        :param method: filter criterion that will be applied to select the features. Available criteria are: (as string)
                       "CIFE" [Lin1996], "ICAP" [Jakulin2005], "CMIM" [Fleuret2004], "JMI"[Yang1999]
        """
        X = np.copy(X)
        Y = np.copy(Y)
        self.num_threads = num_threads
        if X.shape[0] != len(Y):
            raise ValueError("X must have as many samples as there are labels in Y")

        self._n_features = X.shape[1]

        def normalize_data_for_MI(X):
            for i in range(X.shape[1]):
                std = X[:, i].std()
                if std != 0.:
                    X[:, i] /= std
                    X[:, i] -= X[:, i].min()
            return np.floor(X).astype("int")

        self._X = normalize_data_for_MI(np.asarray(X))
        self._Y = np.asarray(Y)

        self._method_str = method
        self._methods = {
            "CIFE": self._J_CIFE,
            "ICAP": self._J_ICAP,
            "CMIM": self._J_CMIM,
            "JMI": self._J_JMI,
            "mRMR": self._J_mRMR,
            "MIFS": self._J_MIFS
        }
        self._filter_criterion_kwargs = {}
        self.change_method(method)
        self._method = self._methods[method]
        self._mutual_information_estimator = lambda X1, X2: mutual_info_score(X1, X2) / np.log(2.0)

        self._redundancy = np.zeros((self._n_features, self._n_features)) - 1.
        self._relevance = np.zeros((self._n_features)) - 1
        self._class_cond_red = np.zeros((self._n_features, self._n_features)) - 1
        self._class_cond_mi_method = self._calculate_class_conditional_MI

    def change_method(self, method, **method_kwargs):
        """
        Changes the filter criterion which is used to select the features

        :param method: string indicating the desired criterion
        """
        if method not in self._methods.keys():
            raise ValueError("method must be one of the following: %s" % str(self._methods.keys()))
        self._method = self._methods[method]
        self._method_str = method
        self._filter_criterion_kwargs = method_kwargs

    def get_current_method(self):
        """
        Prints the currently selected criterion
        """
        print(self._method)

    def get_available_methods(self):
        """
        Returns the implemented criteria as strings
        :return: list of strings containing the implemented criteria
        """
        return self._methods.keys()

    def _calculate_class_conditional_MI(self, X1, X2, Y):
        states = np.unique(Y)
        con_mi = 0.

        for state in states:
            indices = (Y == state)
            p_state = float(np.sum(indices)) / float(len(Y))
            mi = self._mutual_information_estimator(X1[indices], X2[indices])
            con_mi += p_state * mi
        return con_mi

    def _change_cmi_method(self, method):
        """
        Do not use this

        :param method: Seriously. Don't. Its for some testing purposes
        :return:
        """
        self._class_cond_mi_method = method

    def _change_mi_method(self, method):
        """
        Do not use this

        :param method: Seriously. Don't. Its for some testing purposes
        :return:
        """
        self._mutual_information_estimator = method

    def _J_MIFS(self, features_in_set, feature_to_be_tested, beta=1):
        relevancy = self._get_relevance(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                tmp += this_redundancy
            j = relevancy - beta * tmp
        else:
            j = relevancy
        return j

    def _J_mRMR(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevance(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                tmp += this_redundancy
            j = relevancy - 1. / float(len(features_in_set)) * tmp
        else:
            j = relevancy
        return j

    def _J_JMI(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevance(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                this_class_cond_red = self._get_class_cond_red(feature, feature_to_be_tested)
                tmp += (this_redundancy - this_class_cond_red)
            j = relevancy - 1. / float(len(features_in_set)) * tmp
        else:
            j = relevancy
        return j

    def _J_CIFE(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevance(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                this_class_cond_red = self._get_class_cond_red(feature, feature_to_be_tested)
                tmp += (this_redundancy - this_class_cond_red)
            j = relevancy - tmp
        else:
            j = relevancy
        return j

    def _J_ICAP(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevance(feature_to_be_tested)
        tmp = 0.
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                this_class_cond_red = self._get_class_cond_red(feature, feature_to_be_tested)
                tmp += np.max([0, (this_redundancy - this_class_cond_red)])
            j = relevancy - tmp
        else:
            j = relevancy
        return j

    def _J_CMIM(self, features_in_set, feature_to_be_tested):
        relevancy = self._get_relevance(feature_to_be_tested)
        tmps = []
        if len(features_in_set) > 0:
            for feature in features_in_set:
                this_redundancy = self._get_redundancy(feature, feature_to_be_tested)
                this_class_cond_red = self._get_class_cond_red(feature, feature_to_be_tested)
                tmps += [this_redundancy - this_class_cond_red]
            j = relevancy - np.max(tmps)
        else:
            j = relevancy
        return j

    def _get_relevance(self, feat_id):
        if self._relevance[feat_id] == -1:
            self._relevance[feat_id] = self._mutual_information_estimator(self._X[:, feat_id], self._Y)
        return self._relevance[feat_id]

    def _get_redundancy(self, feat1, feat2):
        if self._redundancy[feat1, feat2] == -1:
            this_redundancy = self._mutual_information_estimator(self._X[:, feat1], self._X[:, feat2])
            self._redundancy[feat1, feat2] = this_redundancy
            self._redundancy[feat2, feat1] = this_redundancy
        return self._redundancy[feat1, feat2]

    def _get_class_cond_red(self, feat1, feat2):
        if self._class_cond_red[feat1, feat2] == -1:
            this_class_cond_red = self._class_cond_mi_method(self._X[:, feat1], self._X[:, feat2], self._Y)
            self._class_cond_red[feat1, feat2] = this_class_cond_red
            self._class_cond_red[feat2, feat1] = this_class_cond_red
        return self._class_cond_red[feat1, feat2]

    def _evaluate_feature(self, features_in_set, feature_to_be_tested):
        return self._method(features_in_set, feature_to_be_tested, **self._filter_criterion_kwargs)

    def compute_mi(self, args):
        x1, x2 = args
        return self._mutual_information_estimator(x1, x2)

    def compute_cc_mi(self, args):
        x1, x2, c = args
        return self._class_cond_mi_method(x1, x2, self._Y)

    def compute_MIs(self, current_feature_set):
        p = Pool(self.num_threads)

        current_feature_set = list(current_feature_set)
        features_not_in_set = [i for i in range(self._n_features) if i not in current_feature_set]

        need_to_compute_relevance = [i for i in features_not_in_set if self._relevance[i] == -1]

        relevances = p.map(self.compute_mi, zip([self._X[:, i] for i in need_to_compute_relevance], [self._Y] * len(need_to_compute_relevance)))
        for i, r in zip(need_to_compute_relevance, relevances):
            self._relevance[i] = r

        need_to_compute_redundancy = [[i, j] for i in current_feature_set for j in features_not_in_set if self._redundancy[i, j] == -1]
        redundancies = p.map(self.compute_mi, [(self._X[:, i], self._X[:, j]) for i, j in need_to_compute_redundancy])
        for r, (i, j) in zip(redundancies, need_to_compute_redundancy):
            self._redundancy[i, j] = r
            self._redundancy[j, i] = r

        need_to_compute_cc_red = [[i, j] for i in current_feature_set for j in features_not_in_set if self._class_cond_red[i, j] == -1]
        cc_redundancies = p.map(self.compute_cc_mi, [(self._X[:, i], self._X[:, j], self._Y) for i, j in need_to_compute_redundancy])
        for r, (i, j) in zip(cc_redundancies, need_to_compute_cc_red):
            self._class_cond_red[i, j] = r
            self._class_cond_red[j, i] = r

        p.close()
        p.join()

    def run(self, n_features_to_select):
        """
        Performs the actual feature selection using the specified filter criterion

        :param n_features_to_select: number of features to select
        :return: numpy array of selected features (as IDs)
        """
        logger.info("Initialize filter feature selection:")
        logger.info("using filter method: %s" % self._method_str)

        def find_next_best_feature(current_feature_set):
            features_not_in_set = set(np.arange(self._n_features)).difference(set(current_feature_set))
            best_J = -999999.9
            best_feature = None

            for feature_candidate in features_not_in_set:
                # compute relevance, reduncancy, class cond redundancy
                j_feature = self._evaluate_feature(current_feature_set, feature_candidate)
                if j_feature > best_J:
                    best_J = j_feature
                    best_feature = feature_candidate
            if best_feature is not None:
                logger.info("Best feature found was %d with J_eval= %f. Feature set was %s" % (
                best_feature, best_J, str(current_feature_set)))
            return best_feature

        if n_features_to_select > self._n_features:
            raise ValueError("n_features_to_select must be smaller or equal to the number of features")

        selected_features = 0
        current_feature_set = []
        while selected_features < n_features_to_select:
            print(self._method_str, selected_features)
            self.compute_MIs(current_feature_set)
            best_feature = find_next_best_feature(current_feature_set)
            if best_feature is not None:
                current_feature_set += [best_feature]
                selected_features += 1
            else:
                break

        logger.info("Filter feature selection done. Final set is: %s" % str(current_feature_set))

        return np.array(current_feature_set)


## REFERENCES

# Francois Fleuret. Fast Binary Feature Selection with Conditional Mutual Informa-
# tion. Journal of Machine Learning Research,

# Aleks Jakulin. Machine Learning Based on Attribute Interactions. PhD Thesis,
# University of Ljubljana, Slovenia, 2005.

# D Lin and X Tang. Conditional infomax learning: An integrated framework for
# feature extraction and fusion. In European Conference on Computer Vision, 1996.

# Howard Hua Yang and John Moody. Feature selection based on joint mutual infor-
# mation. Proceedings of International ICSC Symposium on Advances in Intelligent
# Data Analysis, pages 22 25, 1999.

# Roberto Battiti. Using mutual information for selecting features in supervised neural
# net learning. IEEE Transactions on Neural Networks, 5(4):537 550, 1994.

# H C Peng, F H Long, and C Ding. Feature selection based on mutual information:
# Criteria of max-dependency, max-relevance, and min-redundancy. Ieee Transactions
# on Pattern Analysis and Machine Intelligence, 27(8):1226 1238, 2005.
