import csv
import datetime
import os
import time

import numpy as np

from dfa import DFA, random_dfa, dfa_intersection, save_dfa_as_part_of_model,load_dfa_dot
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import RNNLanguageClasifier
from pac_teacher import PACTeacher
from random_words import confidence_interval_many, random_word, confidence_interval_subset

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
              epoch=-1,num_of_examples=-1):
    if hidden_dim == -1:
        hidden_dim = len(dfa.states) * 20
    if num_layers == -1:
        num_layers = 2 + int(len(dfa.states)/10)
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


def learn_and_check(dfa: DFA, benchmark, dir_name=None):
    rnn = learn_dfa(dfa, benchmark)

    if float(benchmark["rnn_testing_acc"]) < 80:
        print("didn't learned the rnn well enough starting over")
        return


    extracted_dfas = extract_dfa_from_rnn(rnn, benchmark, timeout=900)
    if dir_name is not None:
        save_dfa_as_part_of_model(dir_name, dfa, name="dfa")
        dfa.draw_nicely(name="dfa", save_dir=dir_name)
        rnn.save_lstm(dir_name)
        for extracted_dfa, name in extracted_dfas:
            if isinstance(name, DFA):
                save_dfa_as_part_of_model(dir_name, extracted_dfa, name=name)

    models = [dfa, rnn, extracted_dfas[0][0], extracted_dfas[1][0]]
    compute_distances_no_model_checking(models, benchmark, delta=0.005, epsilon=0.005)


def extract_dfa_from_rnn(rnn, benchmark, timeout=900):
    teacher_pac = PACTeacher(rnn)

    ###################################################
    # DFA extraction
    ###################################################
    print("Starting DFA extraction w/o model checking")
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.is_counter_example_in_batches = False
    teacher_pac.teach(student, timeout=timeout)
    benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})

    dfa_extract = minimize_dfa(student.dfa)
    print(student.dfa)
    benchmark.update({"dfa_extract_states": len(dfa_extract.states),
                      "dfa_extract_final": len(dfa_extract.final_states)})

    ###################################################
    # Doing DFA extraction acc. to icml18
    ###################################################
    # print("Starting DFA extraction acc to iclm18")
    # start_time = time.time()
    #
    # dfa_iclm18 = extract_iclm(rnn, time_limit=timeout, initial_split_depth=10)
    #
    # benchmark.update({"extraction_time_icml18": time.time() - start_time,
    #                   "dfa_icml18_states": len(dfa_iclm18.Q),
    #                   "dfa_icml18_final": len(dfa_iclm18.F)})
    #
    # print("Finished DFA extraction")
    #
    # return (dfa_extract, "dfa_extract"), (dfa_iclm18, "dfa_icml18")
    return [(dfa_extract, "dfa_extract")]

def compute_distances_no_model_checking(models, benchmark, epsilon=0.005, delta=0.001):
    print("Starting distance measuring")
    output, samples = confidence_interval_many(models, random_word, width=epsilon, confidence=delta)
    print("The confidence interval for epsilon = {} , delta = {}".format(delta, epsilon))
    print(output)

    benchmark.update({"dist_rnn_vs_inter": "{}".format(output[1][0]),
                      "dist_rnn_vs_extr": "{}".format(output[1][2])})

    benchmark.update({"dist_inter_vs_extr": "{}".format(output[0][2])})

    print("Finished distance measuring")


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

    learn_and_check(dfa, benchmark, save_dir)

    return benchmark


def run_rand_benchmarks_wo_model_checking(num_of_bench=30, save_dir=None):
    print("Running benchmark without model checking with "+ str(num_of_bench) +" number of benchmarks")
    if save_dir is None:
        save_dir = "../models/random_bench_{}".format(datetime.datetime.now().strftime("%d-%b-%Y_%H-%M-%S"))
        os.makedirs(save_dir)

    write_csv_header(save_dir + "/test.csv")
    for num in range(1, num_of_bench + 1):
        print("Running benchmark {}/{}:".format(num, num_of_bench))
        benchmark = rand_benchmark(save_dir + "/" + str(num))
        print("Summary for the {}th benchmark".format(num))
        print(benchmark)
        write_line_csv(save_dir + "/test.csv", benchmark)



def extract(dfa: DFA, benchmark,rnn, dir_name=None):

    extracted_dfas = extract_dfa_from_rnn(rnn, benchmark, timeout=900)
    if dir_name is not None:
        for extracted_dfa, name in extracted_dfas:
            if isinstance(extracted_dfa, DFA):
                save_dfa_as_part_of_model(dir_name, extracted_dfa, name=name+"-extracted_2")

    models = [dfa, rnn, extracted_dfas[0][0]]
    compute_distances_no_model_checking(models, benchmark, delta=0.005, epsilon=0.001)



def run_extraction_on_dir(dir):
    # print(" "+ str(num_of_bench) +" number of benchmarks")
    first_entry = True
    summary_csv = dir + "/extraxtion2.csv"
    for folder in os.walk(dir):
        if os.path.isfile(folder[0] + "/meta"):
            name = folder[0].split('/')[-1]
            rnn = RNNLanguageClasifier().load_lstm(folder[0])
            dfa = load_dfa_dot(folder[0]+"/dfa.dot")
            benchmark = {"name": name}
            extract(dfa, benchmark,rnn, folder[0])
            if first_entry:
                write_csv_header(summary_csv, benchmark.keys())
                first_entry = False
            write_line_csv(summary_csv, benchmark, benchmark.keys())
