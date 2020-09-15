import copy
import csv
import datetime
import os
import time

import numpy as np

from dfa import DFA, random_dfa, dfa_intersection, save_dfa_as_part_of_model, load_dfa_dot
from dfa_check import DFAChecker
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import RNNLanguageClasifier
from pac_teacher import PACTeacher
from random_words import confidence_interval_many, random_word, confidence_interval_subset, model_check_random

FIELD_NAMES = ["alph_len",

               "dfa_inter_states", "dfa_inter_final",
               'dfa_spec_states', 'dfa_spec_final',
               'dfa_extract_specs_states', "dfa_extract_specs_final",
               "dfa_extract_states", "dfa_extract_final",
               "dfa_icml18_states", "dfa_icml18_final",

               "rnn_layers", "rnn_hidden_dim", "rnn_dataset_learning", "rnn_dataset_testing",
               "rnn_testing_acc", "rnn_val_acc", "rnn_time",

               "extraction_time_spec", "extraction_mistake_during",
               "extraction_time", "mistake_time_after", "extraction_mistake_after",
               "extraction_time_icml18",

               "dist_rnn_vs_inter", "dist_rnn_vs_extr", "dist_rnn_vs_extr_spec", "dist_rnn_vs_icml18",
               "dist_inter_vs_extr", "dist_inter_vs_extr_spec", "dist_inter_vs_icml18",

               "dist_specs_rnn", "dist_specs_extract", "dist_specs_extract_w_spec", "statistic_checking_time"]


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


#
def learn_dfa(dfa: DFA, benchmark, hidden_dim=-1, num_layers=-1, embedding_dim=-1, batch_size=-1,
              epoch=-1, num_of_exm_per_length=-1, word_training_length=-1):
    if hidden_dim == -1:
        hidden_dim = len(dfa.states) * 6
    if num_layers == -1:
        num_layers = 3
    if embedding_dim == -1:
        embedding_dim = len(dfa.alphabet) * 2
    if num_of_exm_per_length == -1:
        num_of_exm_per_length = 15000
    if epoch == -1:
        epoch = 10
    if batch_size == -1:
        batch_size = 20
    if word_training_length == -1:
        word_training_length = len(dfa.states) + 5

    start_time = time.time()
    model = RNNLanguageClasifier()
    model.train_a_lstm(dfa.alphabet, dfa.is_word_in,
                       hidden_dim=hidden_dim,
                       num_layers=num_layers,
                       embedding_dim=embedding_dim,
                       batch_size=batch_size,
                       epoch=epoch,
                       num_of_exm_per_lenght=num_of_exm_per_length,
                       word_traning_length=word_training_length
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


def learn_target(target, alphabet, benchmark, hidden_dim=-1, num_layers=-1, embedding_dim=-1, batch_size=-1,
                 epoch=-1, num_of_examples=-1):
    if hidden_dim == -1:
        hidden_dim = 100
    if num_layers == -1:
        num_layers = 3
    if embedding_dim == -1:
        embedding_dim = len(alphabet) * 2
    if epoch == -1:
        epoch = 10
    if batch_size == -1:
        batch_size = 20
    if num_of_examples == -1:
        num_of_examples = 50000

    start_time = time.time()
    model = RNNLanguageClasifier()
    model.train_a_lstm(alphabet, target, random_word,
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


def learn_and_check(dfa: DFA, spec: [DFAChecker], benchmark, dir_name=None):
    rnn = learn_dfa(dfa, benchmark, epoch=3, num_of_exm_per_length=2000)

    extracted_dfas = check_rnn_acc_to_spec(rnn, spec, benchmark)
    if dir_name is not None:
        rnn.save_lstm(dir_name)
        for extracted_dfa, name in extracted_dfas:
            if isinstance(name, DFA):
                save_dfa_as_part_of_model(dir_name, extracted_dfa, name=name)
            # dfa_extract.draw_nicely(name="_dfa_figure", save_dir=dir_name)

    models = [dfa, rnn, extracted_dfas[0][0], extracted_dfas[1][0], extracted_dfas[2][0]]

    compute_distances(models, spec[0].specification, benchmark, delta=0.05, epsilon=0.05)


def check_rnn_acc_to_spec(rnn, spec, benchmark, timeout=900):
    teacher_pac = PACTeacher(rnn, epsilon=0.0005, delta=0.0005)
    student = DecisionTreeLearner(teacher_pac)

    print("Starting DFA extraction")
    ##################################################
    # Doing the model checking during a DFA extraction
    ###################################################
    print("Starting DFA extraction with model checking")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    counter = teacher_pac.check_and_teach(student, spec[0], timeout=timeout)
    benchmark.update({"during_time_spec": "{:.3}".format(time.time() - start_time)})
    dfa_extract_w_spec = student.dfa
    dfa_extract_w_spec = minimize_dfa(dfa_extract_w_spec)

    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_during": "NAN",
                          "dfa_extract_specs_states": len(dfa_extract_w_spec.states),
                          "dfa_extract_specs_final": len(dfa_extract_w_spec.final_states),
                          "dfa_extract_spec_mem_queries": rnn.num_of_membership_queries})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_mistake_during": counter,
                          "dfa_extract_specs_states": len(dfa_extract_w_spec.states),
                          "dfa_extract_specs_final": len(dfa_extract_w_spec.final_states),
                          "dfa_extract_spec_mem_queries": rnn.num_of_membership_queries})

    ###################################################
    # Doing the model checking after a DFA extraction
    ###################################################
    print("Starting DFA extraction w/o model checking")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student, timeout=timeout)
    # benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})

    print("Model checking the extracted DFA")
    counter = student.dfa.is_language_not_subset_of(spec[0].specification)
    if counter is not None:
        if not rnn.is_word_in(counter):
            counter = None

    benchmark.update({"mistake_time_extraction": "{:.3}".format(time.time() - start_time)})

    dfa_extract = minimize_dfa(student.dfa)
    if counter is None:
        print("No mistakes found ==> DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_after": "NAN",
                          "dfa_extract_states": len(dfa_extract.states),
                          "dfa_extract_final": len(dfa_extract.final_states),
                          "dfa_extract_mem_queries": rnn.num_of_membership_queries})
    else:
        print("Mistakes found ==> Counter example: {}".format(counter))
        benchmark.update({"extraction_mistake_after": counter,
                          "dfa_extract_states": len(dfa_extract.states),
                          "dfa_extract_final": len(dfa_extract.final_states),
                          "dfa_extract_mem_queries": rnn.num_of_membership_queries})

    ###################################################
    # Doing the model checking acc. of a sup lang extraction
    ###################################################
    # print("Starting DFA extraction super w/o model checking")
    # rnn.num_of_membership_queries = 0
    # start_time = time.time()
    # student = DecisionTreeLearner(teacher_pac)
    # teacher_pac.teach_a_superset(student, timeout=timeout)
    # # benchmark.update({"extraction_super_time": "{:.3}".format(time.time() - start_time)})
    #
    # print("Model checking the extracted DFA")
    # counter = student.dfa.is_language_not_subset_of(spec[0].specification)
    # if counter is not None:
    #     if not rnn.is_word_in(counter):
    #         counter = None
    #
    # benchmark.update({"mistake_time_super": "{:.3}".format(time.time() - start_time)})
    #
    # dfa_extract_super = minimize_dfa(student.dfa)
    # if counter is None:
    #     print("No mistakes found ==> DFA learned:")
    #     print(student.dfa)
    #     benchmark.update({"extraction_super_mistake_after": "NAN",
    #                       "dfa_extract_super_states": len(dfa_extract.states),
    #                       "dfa_extract_super_final": len(dfa_extract.final_states),
    #                       "dfa_extract_super_mem_queries": rnn.num_of_membership_queries})
    # else:
    #     print("Mistakes found ==> Counter example: {}".format(counter))
    #     benchmark.update({"extraction_super_mistake_after": counter,
    #                       "dfa_extract_super_states": len(dfa_extract.states),
    #                       "dfa_extract_super_final": len(dfa_extract.final_states),
    #                       "dfa_extract_super_mem_queries": rnn.num_of_membership_queries})
    #
    # print("Finished DFA extraction")

    ###################################################
    # Doing the model checking randomly
    ###################################################
    print("starting rand model checking")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    counter = model_check_random(rnn, spec[0].specification, width=0.005, confidence=0.005)
    if counter is None:
        counter = "NAN"
    benchmark.update({"mistake_time_rand": "{:.3}".format(time.time() - start_time),
                      "mistake_rand": counter,
                      "rand_num_queries": rnn.num_of_membership_queries})

    print(benchmark)
    return (dfa_extract_w_spec, "dfa_extract_W_spec"), \
           (dfa_extract, "dfa_extract")  # , \
    # (dfa_extract_super, "dfa_extract_super")


def check_rnn_acc_to_spec_only_mc(rnn, spec, benchmark, timeout=900, delta=0.0005, epsilon=0.0005):
    teacher_pac = PACTeacher(rnn, epsilon=epsilon, delta=delta)
    student = DecisionTreeLearner(teacher_pac)

    ##################################################
    # Doing the model checking PDV
    ###################################################
    print("---------------------------------------------------\n"
          "------Starting property-directed verification------\n"
          "---------------------------------------------------\n")

    rnn.num_of_membership_queries = 0
    start_time = time.time()
    counter_extract_w_spec = teacher_pac.check_and_teach(student, spec[0], timeout=timeout)
    benchmark.update({"PDV_time": "{:.3}".format(time.time() - start_time)})
    dfa_extract_w_spec = student.dfa
    dfa_extract_w_spec = minimize_dfa(dfa_extract_w_spec)

    if counter_extract_w_spec is None:
        print("Using PDV no mistakes found")
        print("DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_PDV": "NAN",
                          "dfa_PDV_states": len(dfa_extract_w_spec.states),
                          "dfa_PDV_final": len(dfa_extract_w_spec.final_states),
                          "PDV_mem_queries": rnn.num_of_membership_queries})
    else:
        print("Using PDV Mistakes found ==> Counter example: {}".format(counter_extract_w_spec))
        print("DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_PDV": counter_extract_w_spec,
                          "dfa_PDV_states": len(dfa_extract_w_spec.states),
                          "dfa_PDV_final": len(dfa_extract_w_spec.final_states),
                          "PDV_mem_queries": rnn.num_of_membership_queries})
    print("Finished PDV in {} sec".format(benchmark["PDV_time"]))

    ##################################################
    # Doing the model checking AAMC
    ###################################################
    print("\n---------------------------------------------------\n"
          "-Starting Automaton Abstraction and Model Checking-\n"
          "---------------------------------------------------\n")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student, timeout=timeout)

    counter = student.dfa.is_language_not_subset_of(spec[0].specification)
    if counter is not None:
        if not rnn.is_word_in(counter):
            counter = None

    benchmark.update({"time_AAMC": "{:.3}".format(time.time() - start_time)})

    dfa_extract = minimize_dfa(student.dfa)
    if counter is None:
        print("Using AAMC no mistakes found ")
        print("DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_AAMC": "NAN",
                          "dfa_AAMC_states": len(dfa_extract.states),
                          "dfa_AAMC_final": len(dfa_extract.final_states),
                          "AAMC_mem_queries": rnn.num_of_membership_queries})
    else:
        print("Using AAMC Mistakes found ==> Counter example: {}".format(counter))
        print("DFA learned:")
        print(student.dfa)
        benchmark.update({"extraction_mistake_AAMC": counter,
                          "dfa_AAMC_states": len(dfa_extract.states),
                          "dfa_AAMC_final": len(dfa_extract.final_states),
                          "AAMC_mem_queries": rnn.num_of_membership_queries})

    print("Finished AAMC in {} sec".format(benchmark["time_AAMC"]))

    #################################################
    # Doing the model checking randomly
    ##################################################
    print("\n---------------------------------------------------\n"
          "---------Starting Statistical Model Checking-------\n"
          "---------------------------------------------------\n")
    rnn.num_of_membership_queries = 0
    start_time = time.time()
    counter = model_check_random(rnn, spec[0].specification, width=epsilon, confidence=delta, timeout=timeout)
    if counter is None:
        print("Using SMC no mistakes found")
        counter = "NAN"
    else:
        print("Using SMC Mistakes found ==> Counter example: {}".format(counter))

    benchmark.update({"time_SMC": "{:.3}".format(time.time() - start_time),
                      "mistake_SMC": counter,
                      "SMC_num_queries": rnn.num_of_membership_queries})

    print("Finished SMC in {} sec".format(benchmark["time_SMC"]))

    return dfa_extract_w_spec, counter_extract_w_spec


def extract_dfa_from_rnn(rnn, benchmark, timeout=900):
    rnn.num_of_membership_queries = 0
    teacher_pac = PACTeacher(rnn)

    ###################################################
    # DFA extraction
    ###################################################
    print("Starting DFA extraction w/o model checking")
    start_time = time.time()
    student = DecisionTreeLearner(teacher_pac)
    teacher_pac.teach(student, timeout=timeout)
    benchmark.update({"extraction_time": "{:.3}".format(time.time() - start_time)})

    dfa_extract = minimize_dfa(student.dfa)
    print(student.dfa)
    benchmark.update({"dfa_extract_states": len(dfa_extract.states),
                      "dfa_extract_final": len(dfa_extract.final_states),
                      "num_of_mem_quarries_extracted": rnn.num_of_membership_queries})

    return dfa_extract


def compute_distances(models, dfa_spec, benchmark, epsilon=0.005, delta=0.001):
    print("Starting distance measuring")
    output, samples = confidence_interval_many(models, random_word, width=epsilon, confidence=delta)
    print("The confidence interval for epsilon = {} , delta = {}".format(delta, epsilon))
    print(output)

    benchmark.update({"dist_rnn_vs_inter": "{}".format(output[1][0]),
                      "dist_rnn_vs_extr_spec": "{}".format(output[1][2]),
                      "dist_rnn_vs_extr": "{}".format(output[1][3]),
                      "dist_rnn_vs_icml18": "{}".format(output[1][4])})

    benchmark.update({"dist_inter_vs_extr_spec": "{}".format(output[0][2]),
                      "dist_inter_vs_extr": "{}".format(output[0][3]),
                      "dist_inter_vs_icml18": "{}".format(output[0][4])})

    start_time = time.time()
    a, samples = confidence_interval_subset(models[1], dfa_spec, confidence=epsilon, width=delta)
    benchmark.update({"statistic_checking_time": time.time() - start_time})
    b, _ = confidence_interval_subset(models[2], dfa_spec, samples, epsilon, delta)
    c, _ = confidence_interval_subset(models[3], dfa_spec, samples, epsilon, delta)
    benchmark.update(
        {"dist_specs_rnn": "{}".format(a),
         "dist_specs_extract_w_spec": "{}".format(b),
         "dist_specs_extract": "{}".format(c)})

    print("Finished distance measuring")


def rand_benchmark(save_dir=None):
    dfa_spec, dfa_inter = DFA(0, {0}, {0: {0: 0}}), DFA(0, {0}, {0: {0: 0}})

    full_alphabet = "abcdefghijklmnopqrstuvwxyz"

    alphabet = full_alphabet[0:np.random.randint(4, 5)]
    benchmark = {}
    benchmark.update({"alph_len": len(alphabet)})

    while len(dfa_inter.states) < 5 or len(dfa_spec.states) < 2 or (len(dfa_inter.states) > 25):
        dfa_rand1 = random_dfa(alphabet, min_states=10, max_states=15, min_final=2, max_final=10)
        dfa_rand2 = random_dfa(alphabet, min_states=5, max_states=7, min_final=4, max_final=5)

        dfa_inter = minimize_dfa(dfa_intersection(dfa_rand1, dfa_rand2))
        dfa_spec = minimize_dfa(dfa_rand2)

    benchmark.update({"dfa_inter_states": len(dfa_inter.states), "dfa_inter_final": len(dfa_inter.final_states),
                      "dfa_spec_states": len(dfa_spec.states), "dfa_spec_final": len(dfa_spec.final_states)})

    if save_dir is not None:
        save_dfa_as_part_of_model(save_dir, dfa_inter, name="dfa_intersection")
        dfa_inter.draw_nicely(name="intersection_dfa_figure", save_dir=save_dir)

        save_dfa_as_part_of_model(save_dir, dfa_spec, name="dfa_spec")
        dfa_spec.draw_nicely(name="spec_dfa_figure", save_dir=save_dir)

    print("DFA to learn {}".format(dfa_inter))
    print("Spec to learn {}".format(dfa_spec))

    learn_and_check(dfa_inter, [DFAChecker(dfa_spec)], benchmark, save_dir)

    return benchmark


def run_rand_benchmarks(num_of_bench=10, save_dir=None):
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


def check_folder_of_rand(folder):
    timeout = 600
    first_entry = True
    summary_csv = folder + "/summary_model_checking_second_try.csv"
    for folder in os.walk(folder):
        if os.path.isfile(folder[0] + "/meta"):
            name = folder[0].split('/')[-1]
            rnn = RNNLanguageClasifier().load_lstm(folder[0])
            dfa = load_dfa_dot(folder[0] + "/dfa.dot")
            i = 1
            for dfa_spec in from_dfa_to_sup_dfa_gen(dfa):
                dfa_spec.save(folder[0] + "/spec_second_" + str(i))
                benchmark = {"name": name, "spec_num": str(i),
                             "spec_states": len(dfa_spec.states),
                             "spec_fin": len(dfa_spec.final_states)}
                check_rnn_acc_to_spec(rnn, [DFAChecker(dfa_spec)], benchmark, timeout)
                if first_entry:
                    write_csv_header(summary_csv, benchmark.keys())
                    first_entry = False
                write_line_csv(summary_csv, benchmark, benchmark.keys())
                i += 1


def from_dfa_to_sup_dfa_gen(dfa: DFA, tries=5):
    not_final_states = [state for state in dfa.states if state not in dfa.final_states]
    if len(not_final_states) == 1:
        return

    created_dfas = []
    for _ in range(tries):
        s = np.random.randint(1, len(not_final_states))
        new_final_num = np.random.choice(len(not_final_states), size=s, replace=False)
        new_final = [not_final_states[i] for i in new_final_num]
        dfa_spec = DFA(dfa.init_state, dfa.final_states + new_final, dfa.transitions)
        dfa_spec = minimize_dfa(dfa_spec)

        if dfa_spec in created_dfas:
            continue
        created_dfas.append(dfa_spec)
        yield dfa_spec


def flawed_flow_cross_product(counter, dfa_extracted, dfa_spec, flawed_flow, rnn):
    s1, s2 = dfa_extracted.init_state, dfa_spec.init_state
    i = 0
    for ch in counter:
        loops = loop_from_initial(dfa_extracted, dfa_spec, s1, s2)
        if len(loops) != 0:
            for loop in loops:
                if check_for_loops(counter[0:i], loop, counter[i:len(counter)], dfa_spec, rnn, flawed_flow):
                    return
        s1, s2 = dfa_extracted.next_state_by_letter(s1, ch), dfa_spec.next_state_by_letter(s2, ch)
        i += 1


def loop_from_initial(dfa1, dfa2, s1, s2):
    loops = []
    visited = [(s1, s2)]
    front = [(s1, s2, tuple())]
    while len(front) != 0:
        s1, s2, w = front.pop()
        for ch in dfa1.alphabet:
            q1, q2 = dfa1.next_state_by_letter(s1, ch), dfa2.next_state_by_letter(s2, ch)
            if (q1, q2) not in visited:
                visited.append((q1, q2))
                front.append((q1, q2, w + tuple(ch)))
            elif (q1, q2) == visited[0]:
                loops.append(w + tuple(ch))
    return loops


def check_for_loops(prefix, loop, suffix, dfa_spec, rnn, flawed_flow):
    count = 0
    preword = prefix
    for _ in range(100):
        if not dfa_spec.is_word_in(preword + suffix) and rnn.is_word_in(preword + suffix):
            count = count + 1
        preword = preword + loop
    if count > 20:
        print("found faulty flow:")
        print("\t prefix:{},\n\t loop:{},\n\t suffix:{}".format(prefix, loop, suffix))
        flawed_flow.append((prefix, loop, suffix, count))
        return True
    else:
        return False


def rand_pregenerated_benchmarks(check_flows=True, timeout=600, delta=0.0005, epsilon=0.0005):
    print("Start random benchmarks")
    first_entry = True
    folder = "../models/rand"
    summary_csv = "../results/summary_rand_model_checking.csv"
    i = 1
    for folder in os.walk(folder):
        if os.path.isfile(folder[0] + "/meta"):
            name = folder[0].split('/')[-1]
            print("Loading RNN in :\"{}\"".format(folder[0]))
            rnn = RNNLanguageClasifier().load_lstm(folder[0])
            if i == 1:
                i = 2
                continue
            # Loads specification dfa in the folder and checks whether
            # the rnn is compliant.
            for file in os.listdir(folder[0]):
                if 'spec_second_' in file:
                    dfa_spec = load_dfa_dot(folder[0] + "/" + file)
                    benchmark = {"name": name, "spec_num": file}

                    print("\n#####################################################\n"
                          "# Starting verification according to PDV,AAMC and SMC\n"
                          "# for the specification: {} \n".format(benchmark["spec_num"]) +
                          "# with epsilon = {} and delta = {}   \n".format(epsilon, delta) +
                          "#####################################################\n")

                    dfa_extracted, counter = check_rnn_acc_to_spec_only_mc(rnn, [DFAChecker(dfa_spec)], benchmark,
                                                                           timeout, epsilon=epsilon, delta=delta)

                    # if found mistake and needs to check for faulty flaws
                    # do the following:
                    if check_flows:
                        flawed_flows = []
                        if counter is not None:
                            print("---------------------------------------------------\n"
                                  "-------------Checking for faulty flows-------------\n"
                                  "---------------------------------------------------\n")
                            flawed_flows = []
                            flawed_flow_cross_product(counter, dfa_extracted, dfa_spec, flawed_flows, rnn)
                        benchmark.update({"flawed_flows": flawed_flows})

                    if first_entry:
                        write_csv_header(summary_csv, benchmark.keys())
                        first_entry = False
                    write_line_csv(summary_csv, benchmark, benchmark.keys())

                    print("\n#####################################################\n"
                          "# Done - verification according to PDV,AAMC and SMC  \n"
                          "# for the specification: {} \n".format(benchmark["spec_num"]) +
                          "#####################################################\n")
