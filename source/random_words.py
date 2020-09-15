import sys
import time
import numpy as np
import torch

from dfa import DFA
from modelPadding import LSTM, RNNLanguageClasifier


# def random_word_func_entropy_dfa(dfa:DFA,p=0.01):
#
#     for s in dfa.state:
#

def random_word(alphabet, p=0.01):
    nums_of_letters = len(alphabet)
    word = []
    while np.random.randint(0, int(1 / p)) != 0:
        letter = np.random.randint(0, nums_of_letters)
        word.append(alphabet[letter])
    return tuple(word)


def random_word_by_letter(alphabet, p=0.01):
    nums_of_letters = len(alphabet)
    while np.random.randint(0, int(1 / p)) != 0:
        letter = np.random.randint(0, nums_of_letters)
        yield alphabet[letter]


def confidence_interval(language1, language2, sampler, delta=0.001, epsilon=0.001, samples=None):
    n = np.log(2 / delta) / (2 * epsilon * epsilon)
    print(n)
    if samples is None:
        samples = set()
        while len(samples) < n:
            w = sampler(language1.alphabet)
            if w not in samples:
                samples.add(w)
            # print(len(samples))
    mistakes = 0
    print("got it")
    for w in samples:
        if language1.is_word_in(w) != language2.is_word_in(w):
            mistakes = mistakes + 1
            # print(mistakes)
    return mistakes / n, samples


def confidence_interval_many(languages, sampler, confidence=0.001, width=0.005, samples=None):
    """
    Produce the probabilistic distance of the given languages. Using the Chernoff-Hoeffding bound we get that
    in order to have:
        P(S - E[S]>width)< confidence
        S = 1/n(n empirical examples)

    the number of examples that one needs to use is:
        #examples = log(2 / confidence) / (2 * width * width)

    For more details:
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    """
    num_of_lan = len(languages)
    if num_of_lan < 2:
        raise Exception("Need at least 2 languages to compare")

    num_of_samples = np.log(2 / confidence) / (2 * width * width)
    print("size of sample:" + str(int(num_of_samples)))
    if samples is None:
        samples = [sampler(languages[0].alphabet) for _ in range(int(num_of_samples))]

    in_langs_lists = []
    i = 0
    sys.stdout.write('\r Creating bool lists for each lan:  {}/{} done'.format(i, num_of_lan))
    torch.cuda.empty_cache()
    for lang in languages:
        if not isinstance(lang, RNNLanguageClasifier):
            in_langs_lists.append([lang.is_word_in(w) for w in samples])
        else:
            rnn_bool_list = []
            batch_size = 1000
            num_of_batches = int((len(samples) / batch_size))
            for i in range(num_of_batches):
                if i % 50 == 0:
                    print("done {}/{}".format(i, num_of_batches))
                batch = samples[i * batch_size:(i + 1) * batch_size]
                batch_out = (lang.is_words_in_batch(batch))
                rnn_bool_list.extend(batch_out)
            batch = samples[num_of_batches * batch_size:len(samples)]
            batch_out = (lang.is_words_in_batch(batch))
            rnn_bool_list.extend(batch_out)
            in_langs_lists.append(rnn_bool_list)

    output = []
    for i in range(num_of_lan):
        output.append([1] * num_of_lan)

    for lang1 in range(num_of_lan):
        for lang2 in range(num_of_lan):
            if lang1 == lang2:
                output[lang1][lang2] = 0
            elif output[lang1][lang2] == 1:
                output[lang1][lang2] = ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
                                         range(len(samples))].count(False)) / num_of_samples

    print()
    return output, samples


def confidence_interval_subset(language_inf, language_sup, samples=None, confidence=0.001, width=0.001):
    """
    Getting the confidence interval(width,confidence) using the Chernoff-Hoeffding bound.
    The number of examples that one needs to use is n= log(2 / confidence) / (2 * width * width.
    For more details:
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    :return:
    """
    start_time = time.time()
    n = np.log(2 / confidence) / (2 * width * width)

    if samples is None:
        samples = []
        while len(samples) <= n:
            # if len(samples) % 1000 == 0:
            #     sys.stdout.write('\r Creating words:  {}/100 done'.format(str(int((len(samples) / n) * 100))))
            samples.append(random_word(language_inf.alphabet))

        sys.stdout.write('\r Creating words:  100/100 done \n')

    mistakes = 0

    for w in samples:
        if (language_inf.is_word_in(w)) and (not language_sup.is_word_in(w)):
            if mistakes == 0:
                print("first mistake")
                print(time.time() - start_time)
            mistakes = mistakes + 1
    return mistakes / n, samples


def confidence_interval_many_for_reuse(languages, sampler, previous_answers=None, confidence=0.001, width=0.005,
                                       samples=None):
    """
    Produce the probabilistic distance of the given languages. Using the Chernoff-Hoeffding bound we get that
    in order to have:
        P(S - E[S]>width)< confidence
        S = 1/n(n empirical examples)

    the number of examples that one needs to use is:
        #examples = log(2 / confidence) / (2 * width * width)

    For more details:
    https://en.wikipedia.org/wiki/Hoeffding%27s_inequality

    """
    num_of_lan = len(languages)
    if num_of_lan < 2:
        raise Exception("Need at least 2 languages to compare")

    n = np.log(2 / confidence) / (2 * width * width)
    # print("size of sample:" + str(int(n)))
    if samples is None:
        samples = []
        while len(samples) <= n:
            if len(samples) % 1000 == 0:
                sys.stdout.write('\r Creating words:  {}/100 done'.format(str(int((len(samples) / n) * 100))))
            samples.append(sampler(languages[0].alphabet))

        # sys.stdout.write('\r Creating words:  100/100 done \n')
    in_langs_lists = []
    # i = 0
    # sys.stdout.write('\r Creating bool lists for each lan:  {}/{} done'.format(i, num_of_lan))
    torch.cuda.empty_cache()
    if previous_answers is None:
        for lang in languages:
            if isinstance(lang, DFA):
                in_langs_lists.append([lang.is_word_in(w) for w in samples])
            else:
                rnn_bool_list = []
                batch_size = 1000
                n = int((len(samples) / batch_size))
                for i in range(n):
                    if i % 50 == 0:
                        print("done {}/{}".format(i, n))
                    batch = samples[i * batch_size:(i + 1) * batch_size]
                    batch_out = (lang.is_words_in_batch(batch))
                    rnn_bool_list.extend(batch_out)
                batch = samples[n * batch_size:len(samples)]
                batch_out = (lang.is_words_in_batch(batch) )
                rnn_bool_list.extend(batch_out)
                in_langs_lists.append(rnn_bool_list)
    else:
        in_langs_lists = previous_answers
        in_langs_lists.append([languages[2].is_word_in(w) for w in samples])
        # print(in_langs_lists)
    output = []
    for _ in range(num_of_lan):
        output.append([1] * num_of_lan)

    for lang1 in range(num_of_lan):
        for lang2 in range(num_of_lan):
            if lang1 == lang2:
                output[lang1][lang2] = 0
            elif output[lang1][lang2] == 1:
                output[lang1][lang2] = ([(in_langs_lists[lang1])[i] == (in_langs_lists[lang2])[i] for i in
                                         range(len(samples))].count(False)) / n

    print()
    return output, samples, in_langs_lists[0:2]


def model_check_random(language_inf, language_sup,  confidence=0.001, width=0.001):
    """


    :return:
    """
    start_time=time.time()
    n = np.log(2 / confidence) / (2 * width * width)
    alph = language_sup.alphabet
    print("num of words" + str(n))
    batch_size = 200
    for i in range(int(n / batch_size) + 1):
        if (time.time() -start_time > 600):
            return None
        batch = [random_word(alph) for _ in range(batch_size)]
        for x, y, w in zip(language_inf.is_words_in_batch(batch) , [language_sup.is_word_in(w) for w in batch],
                           batch):
            if x and (not y):
                # short_word = ""
                # for l in w:
                #     short_word+=l
                #     if language_inf.is_word_in(short_word) and (not language_sup.is_word_in(short_word)):
                #         return short_word
                return w
    return None
