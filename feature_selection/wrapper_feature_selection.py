# This code is a reimplementation of the wrapper methods that are also present in
# https://github.com/ilastik/ilastik-feature-selection. The implementation here is a lot cleaner and supports multi-
# processing.

from copy import deepcopy
import numpy as np
from multiprocessing import Pool


def evaluate_set(args):
    eval_fct, data, target, feature_set = args
    return eval_fct(data, target, feature_set)


def check_identical(set1, set2):
    """
    set1 and set2 must be lists of int
    :param set1:
    :param set2:
    :return:
    """
    set1 = deepcopy(set1)
    set2 = deepcopy(set2)

    set1.sort()
    set2.sort()

    if len(set1) != len(set2):
        return False
    return all([i == j for i, j in zip(set1, set2)])


def check_set_in_already_explored(set, already_explored):
    if isinstance(already_explored, dict):
        return any([check_identical(set, i) for i in already_explored.keys()])
    else:
        return any([check_identical(set, i) for i in already_explored])


def find_set_expansions(current_set, all_features, allow_add=True, allow_remove=False, already_explored=None):
    new_sets = []
    features_not_in_set = [i for i in all_features if i not in current_set]

    if allow_add:
        for f in features_not_in_set:
            potential_new = current_set + [f]
            potential_new.sort()
            add = True
            if already_explored is not None:
                if check_set_in_already_explored(potential_new, already_explored):
                    add = False
            if add:
                new_sets.append(potential_new)

    if allow_remove:
        for f in current_set:
            potential_new = [i for i in current_set if i != f]
            potential_new.sort()
            add = True
            if already_explored is not None:
                if check_set_in_already_explored(potential_new, already_explored):
                    add = False
            if len(potential_new) == 0:
                add = False
            if add:
                new_sets.append(potential_new)
    return new_sets


def sequential_forward_selection(evaluation_function, data, target, max_num_features, num_threads=8):
    """
    evaluation function has signature evaluation_function(data, selected_features)
    :param evaluation_function:
    :param data:
    :param max_num_features:
    :param num_threads:
    :param verbose:
    :return:
    """
    all_feature_sets = {}
    best_feature_set_by_size = {i: {'set':None, 'criterion':9999999999} for i in range(max_num_features + 1)}
    current_set = []
    order_of_addition = []
    current_best_score = 999999
    p = Pool(num_threads)
    best_set = None
    curr_score = None
    
    while len(current_set) < max_num_features:
        print(len(current_set), str(current_set), curr_score)
        examine_these = find_set_expansions(current_set, range(data.shape[1]), True, False, None)
        scores = p.map(evaluate_set, zip([evaluation_function]*len(examine_these), [data] * len(examine_these), [target] * len(examine_these), examine_these))
        idx_best = np.argmin(scores)
        curr_score = scores[idx_best]

        was_added = [i for i in examine_these[idx_best] if i not in current_set]
        order_of_addition.append(was_added)

        current_set = examine_these[idx_best]
        if scores[idx_best] < current_best_score:
            best_set = deepcopy(current_set)
            current_best_score = scores[idx_best]

        # go through all sets and scores, add them to dicts for later analysis
        for score, set in zip(scores, examine_these):
            if best_feature_set_by_size[len(set)]['criterion'] > score:
                best_feature_set_by_size[len(set)]['criterion'] = score
                best_feature_set_by_size[len(set)]['set'] = set
            all_feature_sets[tuple(set)] = score
    return best_set, current_best_score, best_feature_set_by_size, all_feature_sets, order_of_addition


def find_best_set(all_feature_sets, all_feature_sets_that_were_completely_expanded):
    """
    finds best set in all_feature_sets
    :param all_feature_sets:
    :return:
    """
    if len(all_feature_sets) == 0:
        return []
    best_score = 99999999999
    best_set = None
    for s in all_feature_sets.keys():
        if all_feature_sets[s] < best_score and not check_set_in_already_explored(s, all_feature_sets_that_were_completely_expanded):
            best_score = all_feature_sets[s]
            best_set = s
    best_set = list(best_set)
    best_set.sort()
    return best_set, best_score


def best_first_search(evaluation_function, data, target, max_num_features, num_threads=8, max_iters=50):
    """
    evaluation function has signature evaluation_function(data, selected_features)
    :param evaluation_function:
    :param data:
    :param max_num_features:
    :param num_threads:
    :param verbose:
    :return: best feature set, score of that set (as computed by evaluation_function), best feature set of each size
    (best set of size 1, best set of size 2, etc), all feature sets that were evaluated throughout the seartch (useful
    for counting)
    """
    all_evaluated_feature_sets = {}
    best_feature_set_by_size = {i: {'set':None, 'criterion':9999999999} for i in range(max_num_features + 1)}
    current_set = []
    current_best_score = 999999999999
    p = Pool(num_threads)
    best_set = None
    iteration = 0
    all_feature_sets_that_were_completely_expanded = []

    while len(current_set) < max_num_features and iteration < max_iters:
        print(iteration, len(current_set), str(current_set))
        examine_these = find_set_expansions(current_set, range(data.shape[1]), True, True, all_evaluated_feature_sets)
        scores = p.map(evaluate_set, zip([evaluation_function]*len(examine_these), [data] * len(examine_these), [target] * len(examine_these), examine_these))
        idx_best = np.argmin(scores)

        # go through all sets and scores, add them to dicts for later analysis
        for score, set in zip(scores, examine_these):
            if best_feature_set_by_size[len(set)]['criterion'] > score:
                best_feature_set_by_size[len(set)]['criterion'] = score
                best_feature_set_by_size[len(set)]['set'] = set
            all_evaluated_feature_sets[tuple(set)] = score

        if scores[idx_best] < current_best_score:
            print("new best set: ", examine_these[idx_best], "score: ", scores[idx_best])
            best_set = deepcopy(examine_these[idx_best])
            current_best_score = scores[idx_best]
            # if best_set was updated then we dont need to go through the whole list of feature sets to find the
            # current best one
            current_set = best_set
        else:
            print("no new best set")
            # find global best set in all_feature_sets. This may be scores[idx_best], but the search can also
            # continue from a diferent node in the graph
            examine_these = []
            ctr = 0
            while len(examine_these) == 0 and ctr < 1000:
                possible_current_set, _ = find_best_set(all_evaluated_feature_sets, all_feature_sets_that_were_completely_expanded)
                #print(possible_current_set)
                # the new best set may not have any valid expansions, check for expansions and if there are none then redraw new set
                examine_these = find_set_expansions(possible_current_set, range(data.shape[1]), True, True, all_evaluated_feature_sets)
                all_feature_sets_that_were_completely_expanded.append(possible_current_set)
                ctr += 1
            if ctr == 1000:
                break
            current_set = possible_current_set

        all_feature_sets_that_were_completely_expanded.append(current_set)

        iteration += 1

    p.close()
    p.join()
    return best_set, current_best_score, best_feature_set_by_size, all_evaluated_feature_sets
