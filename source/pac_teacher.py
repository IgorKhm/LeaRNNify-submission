import time

import numpy as np

from dfa import DFA
from modelPadding import RNNLanguageClasifier
from random_words import random_word
from teacher import Teacher


class PACTeacher(Teacher):

    def __init__(self, model: DFA, epsilon=0.001, delta=0.001):
        assert ((epsilon <= 1) & (delta <= 1))
        Teacher.__init__(self, model)
        self.epsilon = epsilon
        self.delta = delta
        self._log_delta = np.log(delta)
        self._log_one_minus_epsilon = np.log(1 - epsilon)
        self._num_equivalence_asked = 0

        self.prev_examples = {}

        self.is_counter_example_in_batches = isinstance(self.model, RNNLanguageClasifier)

    def equivalence_query(self, dfa: DFA):
        """
        Tests whether the dfa is equivalent to the model by testing random words.
        If not equivalent returns an example
        """

        number_of_rounds = int(
            (1 / self.epsilon) * (np.log(1 / self.delta) + np.log(2) * (self._num_equivalence_asked + 1)))

        self._num_equivalence_asked = self._num_equivalence_asked + 1

        if self.is_counter_example_in_batches:
            batch_size = 200
            for i in range(int(number_of_rounds / batch_size) + 1):
                batch = [random_word(self.model.alphabet) for _ in range(batch_size)]
                for x, y, w in zip(self.model.is_words_in_batch(batch), [dfa.is_word_in(w) for w in batch],
                                   batch):
                    if x != y:
                        return w
            return None

        else:
            for i in range(number_of_rounds):
                word = random_word(self.model.alphabet)
                if self.model.is_word_in(word) != dfa.is_word_in(word):
                    return word
            return None

    def model_subset_of_dfa_query(self, dfa: DFA):
        """
        Tests whether the model language is a subset of the dfa language by testing random words.
        If not subset returns an example
        """

        number_of_rounds = int(
            (1 / self.epsilon) * (np.log(1 / self.delta) + np.log(2) * (self._num_equivalence_asked + 1)))
        self._num_equivalence_asked = self._num_equivalence_asked + 1

        if isinstance(self.model, RNNLanguageClasifier):
            batch_size = 200
            for i in range(int(number_of_rounds / batch_size) + 1):
                batch = [random_word(self.model.alphabet) for _ in range(batch_size)]
                for x, y, w in zip(self.model.is_words_in_batch(batch) > 0.5, [dfa.is_word_in(w) for w in batch],
                                   batch):
                    if x and (not y):
                        return w
            return None

        else:
            for i in range(number_of_rounds):
                word = random_word(self.model.alphabet)
                if self.model.is_word_in(word) != dfa.is_word_in(word):
                    return word
            return None

    def membership_query(self, word):
        return self.model.is_word_in(word)

    def teach(self, learner, timeout=600):
        self._num_equivalence_asked = 0
        learner.teacher = self
        i = 0
        start_time = time.time()
        t100 = start_time
        while True:
            if time.time() - start_time > timeout:
                print(time.time() - start_time)
                return
            i = i + 1
            if i % 50 == 0:
                print("this is the {}th round".format(i))
                print("{} time has passed from the begging and {} from the last 100".format(time.time() - start_time,
                                                                                            time.time() - t100))
                t100 = time.time()

            counter = self.equivalence_query(learner.dfa)
            if counter is None:
                break
            num_of_ref = learner.new_counterexample(counter, self.is_counter_example_in_batches)
            self._num_equivalence_asked += num_of_ref

    def check_and_teach(self, learner, checker, timeout=600):
        learner.teacher = self
        self._num_equivalence_asked = 0
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                return

            # Searching for counter examples in the spec:
            counter_example = checker.check_for_counterexample(learner.dfa)

            if counter_example is not None:
                if not self.model.is_word_in(counter_example):
                    self._num_equivalence_asked += 1
                    num = learner.new_counterexample(counter_example, self.is_counter_example_in_batches)
                    if num > 1:
                        self._num_equivalence_asked += num - 1
                else:
                    print('found counter mistake in the model: ', counter_example)
                    return counter_example

            # Searching for counter examples in the the model:
            else:

                counter_example = self.model_subset_of_dfa_query(learner.dfa)
                if counter_example is None:
                    return None
                else:
                    num_equivalence_used = learner.new_counterexample(counter_example,
                                                                      self.is_counter_example_in_batches)
                    if num_equivalence_used > 1:
                        self._num_equivalence_asked += num_equivalence_used - 1
