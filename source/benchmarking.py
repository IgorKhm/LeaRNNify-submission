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
from rand_dfa_rnn import create_random_couples_of_dfa_and_rnn
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
    folder_main = "../models/rand"
    if not os.path.isdir("../results/"):
        os.makedirs("../results/")
    summary_csv = "../results/summary_rand_model_checking.csv"

    for folder in os.walk(folder_main):
        if os.path.isfile(folder[0] + "/meta"):
            name = folder[0].split('/')[-1]
            print("Loading RNN in :\"{}\"".format(folder[0]))
            rnn = RNNLanguageClasifier().load_lstm(folder[0])
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


def generate_rand_spec_and_check_them(folder=None, check_flows=True, timeout=600, delta=0.0005, epsilon=0.0005):
    first_entry = True
    folder, creation_time = create_random_couples_of_dfa_and_rnn(save_dir=folder, num_of_bench=3)
    if not os.path.isdir("../results/"):
        os.makedirs("../results/")
    summary_csv = "../results/summary_model_checking_rand_{}.csv".format(creation_time)
    for folder in os.walk(folder):
        if os.path.isfile(folder[0] + "/meta"):
            name = folder[0].split('/')[-1]
            rnn = RNNLanguageClasifier().load_lstm(folder[0])
            dfa = load_dfa_dot(folder[0] + "/dfa.dot")
            spec_num = 1
            for dfa_spec in from_dfa_to_sup_dfa_gen(dfa):
                dfa_spec.save(folder[0] + "/spec_second_" + str(spec_num))
                benchmark = {"name": name, "spec_num": spec_num}

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
                spec_num += 1
