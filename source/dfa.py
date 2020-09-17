import os
import string

import numpy as np
from IPython.display import Image
from IPython.display import display

import graphviz as gv
import functools

digraph = functools.partial(gv.Digraph, format='png')
graph = functools.partial(gv.Graph, format='png')
separator = "_"


class DFA:
    def __init__(self, init_state, final_states, transitions):
        self.final_states = final_states
        self.init_state = init_state
        self.transitions = transitions
        self.states = transitions.keys()
        self.alphabet = list(transitions[init_state].keys())
        self.current_state = self.init_state

    def is_word_in(self, word):
        state = self.init_state
        for letter in word:
            state = self.transitions[state][letter]
        return state in self.final_states

    def next_state_by_letter(self, state, letter):
        next_state = self.transitions[state][letter]
        return next_state

    def is_final_state(self, state):
        return state in self.final_states

    def is_word_letter_by_letter(self, letter, reset=False):
        if reset:
            self.current_state = self.init_state

        self.current_state = self.next_state_by_letter(self.current_state, letter)
        return self.current_state in self.final_states

    def reset_current_to_init(self):
        self.current_state = self.init_state

    def equivalence_with_counterexample(self, other):
        """
        Check if equal to another DFA by traversing their cross product.
        If equal returns None, otherwise returns a counter example.

        """
        if self.is_word_in(tuple()) != other.is_word_in(tuple()):
            return tuple()

        cross_states = {(self.init_state, other.init_state): (tuple(), None)}
        to_check = [(self.init_state, other.init_state)]
        alphabet = self.alphabet

        while len(to_check) != 0:
            s1, s2 = to_check.pop(0)
            for l in alphabet:
                q1 = self.next_state_by_letter(s1, l)
                q2 = other.next_state_by_letter(s2, l)

                if (q1 in self.final_states) != (q2 in other.final_states):
                    counter_example = tuple([l])
                    q1, q2 = s1, s2

                    while (q1 != self.init_state) | (q2 != other.init_state):
                        l, q1, q2 = cross_states.get((q1, q2))
                        counter_example = tuple([l]) + counter_example
                    return counter_example

                if cross_states.get((q1, q2)) is None:
                    to_check.append((q1, q2))
                    cross_states.update({(q1, q2): (l, s1, s2)})
        return None

    def is_language_not_subset_of(self, other):
        """
        Checks whether this DFA's language is a subset of the language of other (checking A \cap A'^c != 0),
        by traversing the cross product of self and the complimentary of other.
        If equal returns None, otherwise returns a counter example.
        """
        if self.is_word_in(tuple()) & (not other.is_word_in(tuple())):
            return tuple()

        cross_states = {(self.init_state, other.init_state): (tuple(), None)}
        to_check = [(self.init_state, other.init_state)]
        alphabet = self.alphabet

        while len(to_check) != 0:
            s1, s2 = to_check.pop(0)
            for l in alphabet:
                q1 = self.next_state_by_letter(s1, l)
                q2 = other.next_state_by_letter(s2, l)

                if (q1 in self.final_states) and (q2 not in other.final_states):
                    counter_example = tuple([l])
                    q1, q2 = s1, s2

                    while (q1 != self.init_state) | (q2 != other.init_state):
                        l, q1, q2 = cross_states.get((q1, q2))
                        counter_example = tuple([l]) + counter_example
                    return counter_example

                if cross_states.get((q1, q2)) is None:
                    to_check.append((q1, q2))
                    cross_states.update({(q1, q2): (l, s1, s2)})
        return None

    def save(self, filename):
        with open(filename + ".dot", "w") as file:
            file.write("digraph g {\n")
            file.write('__start0 [label="" shape="none]\n')

            for s in self.states:
                if s in self.final_states:
                    shape = "doublecircle"
                else:
                    shape = "circle"
                file.write('{} [shape="{}" label="{}"]\n'.format(s, shape, s))

            file.write('__start0 -> {}\n'.format(self.init_state))

            for s1 in self.transitions.keys():
                tran = self.transitions[s1]
                for letter in tran.keys():
                    file.write('{} -> {}[label="{}"]\n'.format(s1, tran[letter], letter))
            file.write("}\n")

    def __eq__(self, other):
        if self.is_word_in("") != other.is_word_in(""):
            return ""

        cross_states = {(self.init_state, other.init_state): ("", None)}
        to_check = [(self.init_state, other.init_state)]
        alphabet = self.alphabet

        while len(to_check) != 0:
            s1, s2 = to_check.pop(0)
            for l in alphabet:
                q1 = self.next_state_by_letter(s1, l)
                q2 = other.next_state_by_letter(s2, l)

                if (q1 in self.final_states) != (q2 in other.final_states):
                    return False

                if cross_states.get((q1, q2)) is None:
                    to_check.append((q1, q2))
                    cross_states.update({(q1, q2): (l, s1, s2)})
        return True

    def __repr__(self):
        return str(len(self.states)) + " states, " + str(len(self.final_states)) + " final states and " + \
               str(len(self.alphabet)) + " letters."

    # Stole the following function(draw_nicely) from lstar algo: https://github.com/tech-srl/lstar_extraction
    def draw_nicely(self, force=False, maximum=60,
                    name="dfa", save_dir="img/"):  # todo: if two edges are identical except for letter, merge them
        # and note both the letters
        if (not force) and len(self.transitions) > maximum:
            return

        # suspicion: graphviz may be upset by certain sequences, avoid them in nodes
        label_to_number_dict = {False: 0}  # false is never a label but gets us started

        def label_to_numberlabel(label):
            max_number = max(label_to_number_dict[l] for l in label_to_number_dict)
            if not label in label_to_number_dict:
                label_to_number_dict[label] = max_number + 1
            return str(label_to_number_dict[label])

        def add_nodes(graph, nodes):  # stolen from http://matthiaseisen.com/articles/graphviz/
            for n in nodes:
                if isinstance(n, tuple):
                    graph.node(n[0], **n[1])
                else:
                    graph.node(n)
            return graph

        def add_edges(graph, edges):  # stolen from http://matthiaseisen.com/articles/graphviz/
            for e in edges:
                if isinstance(e[0], tuple):
                    graph.edge(*e[0], **e[1])
                else:
                    graph.edge(*e)
            return graph

        g = digraph()
        g = add_nodes(g, [(label_to_numberlabel(self.init_state),
                           {'color': 'green' if self.init_state in self.final_states else 'black',
                            'shape': 'hexagon', 'label': 'start'})])
        states = list(set(self.states) - {self.init_state})
        g = add_nodes(g, [(label_to_numberlabel(state), {'color': 'green' if state in self.final_states else 'black',
                                                         'label': str(i)})
                          for state, i in zip(states, range(1, len(states) + 1))])

        #
        # g = add_nodes(g, [(str(self.init_state),
        #                    {'color': 'green' if self.init_state in self.final_states else 'black',
        #                     'shape': 'hexagon', 'label': 'start'})])
        # states = list(set(self.states) - {self.init_state})
        # g = add_nodes(g, [(str(state), {'color': 'green' if state in self.final_states else 'black',
        #                                 'label': str(i)})
        #                   for state, i in zip(states, range(1, len(states) + 1))])

        def group_edges():
            def clean_line(line, group):
                line = line.split(separator)
                line = sorted(line) + ["END"]
                in_sequence = False
                last_a = ""
                clean = line[0]
                if line[0] in group:
                    in_sequence = True
                    first_a = line[0]
                    last_a = line[0]
                for a in line[1:]:
                    if in_sequence:
                        if a in group and (ord(a) - ord(last_a)) == 1:  # continue sequence
                            last_a = a
                        else:  # break sequence
                            # finish sequence that was
                            if (ord(last_a) - ord(first_a)) > 1:
                                clean += ("-" + last_a)
                            elif not last_a == first_a:
                                clean += (separator + last_a)
                            # else: last_a==first_a -- nothing to add
                            in_sequence = False
                            # check if there is a new one
                            if a in group:
                                first_a = a
                                last_a = a
                                in_sequence = True
                            if not a == "END":
                                clean += (separator + a)
                    else:
                        if a in group:  # start sequence
                            first_a = a
                            last_a = a
                            in_sequence = True
                        if not a == "END":
                            clean += (separator + a)
                return clean

            edges_dict = {}
            for state in self.states:
                for a in self.alphabet:
                    edge_tuple = (label_to_numberlabel(state), label_to_numberlabel(self.transitions[state][a]))
                    # print(str(edge_tuple)+"    "+a)
                    if not edge_tuple in edges_dict:
                        edges_dict[edge_tuple] = a
                    else:
                        edges_dict[edge_tuple] += separator + a
                    # print(str(edge_tuple)+"  =   "+str(edges_dict[edge_tuple]))
            for et in edges_dict:
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_lowercase)
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_uppercase)
                edges_dict[et] = clean_line(edges_dict[et], "0123456789")
                edges_dict[et] = edges_dict[et].replace(separator, ",")
            return edges_dict

        edges_dict = group_edges()
        g = add_edges(g, [(e, {'label': edges_dict[e]}) for e in edges_dict])
        # print('\n'.join([str(((str(state),str(self.delta[state][a])),{'label':a})) for a in self.alphabet for state in
        #                  self.Q]))
        # g = add_edges(g,[((label_to_numberlabel(state),label_to_numberlabel(self.delta[state][a])),{'label':a})
        #                  for a in self.alphabet for state in self.Q])
        # display(Image(filename=g.render(filename='img/automaton')))

        g.render(filename=save_dir + "/" + name)


def random_dfa(alphabet, min_states=10, max_states=100, min_final=1, max_final=10) -> DFA:
    states = np.random.randint(min_states, max_states)
    num_of_final = np.random.randint(min_final, max_final)
    initial_state = np.random.randint(1, states)

    final_states = np.random.choice(states, size=num_of_final, replace=False)

    transitions = {}
    for s in range(states):
        tran = {}
        for l in alphabet:
            q = np.random.randint(1, states)
            tran.update({l: q})
        transitions.update({s: tran})

    return DFA(initial_state, final_states.tolist(), transitions)



def load_dfa_dot(filename: str) -> DFA:
    mode = "init"
    transitions = {}
    states = []
    final_states = []

    with open(filename, "r") as file:
        for line in file.readlines():
            if mode == "init":
                if "__start0" in line:
                    mode = "states"
                    continue
            elif mode == "states":
                # reading states:
                if "__start0" in line:
                    initial_state = line.split(" -> (")[1]
                    initial_state = tuple([''])
                    mode = "transitions"
                    continue
                new_state = (line.split(") [")[0]).replace("\'", "")
                new_state = new_state.replace("(", '')
                new_state = new_state.replace(" ", '')
                new_state = new_state.split(",")
                new_state = tuple(new_state)
                transitions.update({new_state: {}})
                states.append(new_state)
                if "doublecircle" in line:
                    final_states.append(new_state)
            elif mode == "transitions":
                if line == '}\n':
                    break
                line = line.replace(" -> ", ";")
                line = line.replace('[label="', ";")
                line = line.replace('"]\n', "")
                split = line.split(";")
                state_from = tuple(((((split[0].replace('(', '')).replace(')', '')).replace('\'', '')).replace(' ', '')).split(","))
                state_to = tuple(((((split[1].replace('(', '')).replace(')', '')).replace('\'', '')).replace(' ', '')).split(","))
                t = transitions[state_from]
                t.update({split[2]: state_to})
    return DFA(initial_state, final_states, transitions)

def load_dfa_dot_TN(filename):
    mode = "init"
    transitions = {}
    states = []
    final_states = []

    with open(filename, "r") as file:
        for line in file.readlines():
            if mode == "init":
                if "__start0" in line:
                    mode = "states"
                    continue
            elif mode == "states":
                # reading states:
                if "__start0" in line:
                    initial_state = line.split(" -> ")[1]
                    initial_state = initial_state[:len(initial_state) - 1]
                    mode = "transitions"
                    continue
                new_state = line.split(" ")[0]
                transitions.update({new_state: {}})
                states.append(new_state)
                if "doublecircle" in line:
                    final_states.append(new_state)
            elif mode == "transitions":
                if line == '}\n':
                    break
                line = line.replace(" -> ", ";")
                line = line.replace('[label="', ";")
                line = line.replace('"]\n', "")
                split = line.split(";")
                t = transitions[split[0]]
                t.update({split[2]: split[1]})
    return DFA(initial_state, final_states, transitions)


def save_dfa_as_part_of_model(dir_name, dfa: DFA, name="dfa", force_overwrite=False):
    """
    Saves as a part of model, i.e. in a folder with an RNN and other information
    """
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    elif os.path.exists(dir_name + "/"+name+".dot") & (not force_overwrite):
        if input("the save {} exists. Enter y if you want to overwrite it.".format(name)) != "y":
            return
    dfa.save(dir_name + "/" + name)


def dfa_intersection(model1: DFA, model2: DFA) -> DFA:
    if model1.alphabet != model2.alphabet:
        raise Exception("The two DFAs have different alphabets")

    new_states = [(state1, state2) for state1 in model1.states for state2 in model2.states]
    new_init_states = (model1.init_state, model2.init_state)
    new_final_states = [(state1, state2) for state1 in model1.final_states for state2 in model2.final_states]
    new_transitions = {}
    for (state1, state2) in new_states:
        new_transitions[(state1, state2)] = {
            letter: (model1.transitions[state1][letter], model2.transitions[state2][letter]) for letter in
            model1.alphabet}

    new_dfa = DFA(new_init_states, new_final_states, new_transitions)

    return new_dfa


class DFANoisy(DFA):
    def __init__(self, init_state, final_states, transitions, mistake_prob=0.01):
        super().__init__(init_state, final_states, transitions)
        self.mistake_prob = mistake_prob
        self.known_mistakes = {}

    def is_word_in(self, word):
        if word in self.known_mistakes:
            return self.known_mistakes[word]

        state = self.init_state
        for letter in word:
            state = self.transitions[state][letter]
        label = state in self.final_states

        if np.random.randint(0, int(1 / self.mistake_prob)) == 0:
            self.known_mistakes.update({word: not label})
            return not label
        else:
            self.known_mistakes.update({word: label})
            return label
