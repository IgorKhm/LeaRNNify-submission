import csv
import datetime
import os
import time

import numpy as np

from dfa import DFA, random_dfa, save_dfa_as_part_of_model
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import RNNLanguageClasifier
from random_words import random_word

FIELD_NAMES = ["alph_len",

               "dfa_states", "dfa_final",
               "dfa_extract_states", "dfa_extract_final",
               "dfa_icml18_states", "dfa_icml18_final",

               "rnn_layers", "rnn_hidden_dim", "rnn_dataset_learning", "rnn_dataset_testing",
               "rnn_testing_acc", "rnn_val_acc", "rnn_time",

               "extraction_time",
               "extraction_time_icml18",

               "dist_rnn_vs_inter", "dist_rnn_vs_extr", "dist_rnn_vs_icml18",
               "dist_inter_vs_extr", "dist_inter_vs_icml18"]


def write_csv_header(filename, fieldnames=None):
    if fieldnames is None:
        fieldnames = FIELD_NAMES
    with open(filename, mode='a') as employee_file:
        writer = csv.DictWriter(employee_file, fieldnames=fieldnames)
        writer.writeheader()


def write_line_csv(filename, benchmark, fieldnames=None):
    if fieldnames is None:
        fieldnames = FIELD_NAMES
    with open(filename, mode='a') as benchmark_summary:
        writer = csv.DictWriter(benchmark_summary, fieldnames=fieldnames)
        writer.writerow(benchmark)


def minimize_dfa(dfa: DFA) -> DFA:
    teacher_pac = ExactTeacher(dfa)
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student)
    return student.dfa


def learn_dfa(dfa: DFA, benchmark, hidden_dim=-1, num_layers=-1, embedding_dim=-1, batch_size=-1,
              epoch=-1, num_of_examples=-1):
    if hidden_dim == -1:
        hidden_dim = len(dfa.states) * 20
    if num_layers == -1:
        num_layers = 2 + int(len(dfa.states) / 10)
    if embedding_dim == -1:
        embedding_dim = len(dfa.alphabet) * 2
    if epoch == -1:
        epoch = 30
    if batch_size == -1:
        batch_size = 20
    if num_of_examples == -1:
        num_of_examples = 250000

    start_time = time.time()
    model = RNNLanguageClasifier()
    model.train_a_lstm(dfa.alphabet, dfa.is_word_in, random_word,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       embedding_dim=embedding_dim,
                       batch_size=batch_size,
                       epoch=epoch,
                       num_of_examples=num_of_examples
                       )

    benchmark.update({"rnn_time": "{:.3}".format(time.time() - start_time),
                      "rnn_hidden_dim": hidden_dim,
                      "rnn_layers": num_layers,
                      "rnn_testing_acc": "{:.3}".format(model.test_acc),
                      "rnn_val_acc": "{:.3}".format(model.val_acc),
                      "rnn_dataset_learning": model.num_of_train,
                      "rnn_dataset_testing": model.num_of_test})

    print("time: {}".format(time.time() - start_time))
    return model


def learn_dfa_with_rnn(dfa: DFA, benchmark, dir_name=None):
    rnn = learn_dfa(dfa, benchmark)

    if float(benchmark["rnn_testing_acc"]) < 95:
        print("didn't learned the rnn well enough starting over")
        return
    else:
        save_dfa_as_part_of_model(dir_name, dfa, name="dfa")
        dfa.draw_nicely(name="dfa", save_dir=dir_name)
        rnn.save_lstm(dir_name)


def rand_benchmark(save_dir=None):
    dfa = DFA(0, {0}, {0: {0: 0}})

    full_alphabet = "abcdefghijklmnopqrstuvwxyz"

    alphabet = full_alphabet[0:5]
    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    while len(dfa.states) < 5:
        max_final_states = np.random.randint(5, 29)
        dfa_rand1 = random_dfa(alphabet, min_states=max_final_states, max_states=30, min_final=1,
                               max_final=max_final_states)
        dfa = minimize_dfa(dfa_rand1)

    benchmark.update({"dfa_states": len(dfa.states), "dfa_final": len(dfa.final_states)})

    print("DFA to learn {}".format(dfa))

    learn_dfa_with_rnn(dfa, benchmark, save_dir)

    return benchmark


def create_random_couples_of_dfa_rnn(num_of_bench=10, save_dir=None):
    print("Running benchmark without model checking with " + str(num_of_bench) + " number of benchmarks")
    if save_dir is None:
        creation_time = datetime.datetime.now().strftime("%d-%b-%Y_%H-%M")
        save_dir = "../models/random_bench_{}".format(creation_time)
        os.makedirs(save_dir)

    write_csv_header(save_dir + "/test.csv")
    for num in range(1, num_of_bench + 1):
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num))
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark)

    return save_dir ,creation_time
