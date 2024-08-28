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


def _remove_random_fraction_residues(fraction, max_remove):

    def __remove_random_fraction_from_multiple_lists(*lists):
        if not lists:
            raise ValueError("At least one list must be provided")
        list_lengths = [len(lst) for lst in lists]
        n_remove = math.floor(fraction * list_lengths[0])
        if n_remove == 0:
            return lists
        n_remove = random.randrange(0, min(n_remove, max_remove))
        indices_to_remove = random.sample(range(list_lengths[0]), n_remove)
        for lst in lists:
            for index in sorted(indices_to_remove, reverse=True):
                del lst[index]
        return lists
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


def _with_probability(probability):
    """
    Returns True with the given probability.

    :param probability: A float between 0 and 1 representing the probability of returning True.
    :return: True with the given probability, otherwise False.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    return random.random() < probability
