import math
from abc import ABC, abstractmethod
import random


class AbstractAugmenter(ABC):
    @abstractmethod
    def add_change(self, dom_i, dom_j, dom_op):
        pass


class NullAugmenter(AbstractAugmenter):

    def add_change(self, dom_i, dom_j, dom_op):
        return lambda *x: x


class SelfAugmenter(AbstractAugmenter):

    def __init__(self, n_residues=1):
        self.n_residues = n_residues

    def add_change(self, dom_i, dom_j, dom_op):
        if dom_i != dom_j or dom_op == "dom_i":
            return lambda *x: x
        return _remove_random_residues(self.n_residues)


class SelfAugmenterRandomFraction(AbstractAugmenter):

    def __init__(self, fraction=0.1, max_remove=10):
        self.fraction = fraction
        self.max_remove = max_remove

    def add_change(self, dom_i, dom_j, dom_op):
        if dom_i != dom_j or dom_op == "dom_i":
            return lambda *x: x
        return _remove_random_fraction_residues(self.fraction, self.max_remove)


class SegmentPermutationAugmenter(AbstractAugmenter):

    def __init__(self, permute_prob=0.1, n_splits=5):
        self.permute_prob = permute_prob
        self.n_splits = n_splits

    def add_change(self, dom_i, dom_j, dom_op):
        if _with_probability(self.permute_prob):
            return _random_permute_regions(self.n_splits)
        return lambda *x: x


class AugmenterComposer(AbstractAugmenter):
    def __init__(self, augmenters):
        self.augmenters = augmenters

    def add_change(self, dom_i, dom_j, dom_op):
        def __composer(*x):
            for augmenter in self.augmenters:
                x = augmenter.add_change(dom_i, dom_j, dom_op)(*x)
            return x
        return __composer


def _remove_random_fraction_residues(fraction, max_remove):
    def __remove_random_fraction_from_multiple_lists(*lists):
        result = [lst.copy() for lst in lists]
        if not result:
            raise ValueError("At least one list must be provided")
        list_lengths = [len(lst) for lst in result]
        _remove = math.floor(fraction * list_lengths[0])
        n_remove = random.randrange(1, min(_remove, max_remove) + 1)
        indices_to_remove = random.sample(range(list_lengths[0]), n_remove)
        for lst in result:
            for index in sorted(indices_to_remove, reverse=True):
                del lst[index]
        return result
    return __remove_random_fraction_from_multiple_lists


def _remove_random_residues(n):
    def __remove_n_random_elements_from_multiple_lists(*lists):
        if not lists:
            raise ValueError("At least one list must be provided")
        list_lengths = [len(lst) for lst in lists]
        if any(length < n for length in list_lengths):
            raise ValueError("n cannot be greater than the length of any of the lists")
        indices_to_remove = random.sample(range(list_lengths[0]), n)
        for lst in lists:
            for index in sorted(indices_to_remove, reverse=True):
                del lst[index]
        return lists
    return __remove_n_random_elements_from_multiple_lists


def _random_permute_regions(n_splits):
    def ___random_permute_list_regions(*lists):
        if not lists:
            raise ValueError("At least one list must be provided")
        n = len(lists[0])
        list_lengths = [len(lst) for lst in lists]
        if any(length != n for length in list_lengths):
            raise ValueError("all lists should have he same length")
        return _permute_regions_lists(lists, random.randrange(2, n_splits))
    return ___random_permute_list_regions


def _with_probability(probability):
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    return random.random() < probability


def _permute_regions_lists(lst_of_lists, n):
    region_indices = list(range(n))
    random.shuffle(region_indices)
    result = []
    for lst in lst_of_lists:
        lst_len = len(lst)
        regions = [lst[i * lst_len // n : (i + 1) * lst_len // n] for i in range(n)]
        permuted_regions = [regions[i] for i in region_indices]
        permuted_list = [elem for region in permuted_regions for elem in region]
        result.append(permuted_list)
    return tuple(result)
