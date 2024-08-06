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

    def __init__(self, n_residues):
        self.n_residues = n_residues

    def add_change(self, dom_i, dom_j, dom_op):
        if dom_i != dom_j or dom_op == "dom_i":
            return lambda *x: x
        return remove_random_residues(self.n_residues)


def remove_random_residues(n):
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
