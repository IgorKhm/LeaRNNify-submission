import cProfile
import os
import time
import csv
import numpy as np


from dfa import *
from dfa_check import DFAChecker
from model_checker import * 
from exact_teacher import ExactTeacher
from learner_decison_tree import DecisionTreeLearner
from modelPadding import RNNLanguageClasifier
from pac_teacher import PACTeacher
from random_words import random_nonempty_word
from temporal_networks import *


#For Temporal Networks 


error = 0.0005
confidence = 0.0005



class Data:
	def __init__(self, filename=None, alphabet=None):	
		self.filename = filename
		self.alphabet = alphabet
		if filename!=None:
			self.read_from_file()
		
	
	def read_from_file(self):
		self.word_list = []
		self.label_list = []

		with open(self.filename, 'r') as file:
			mode=0
			while True:
				line=file.readline()
				if not line:
					break
				if line == '---\n':
					mode+=1
					continue

				if mode==0:
					line=line.split(',')
					line=line[:-1] + [(line[-1].strip('\n'))]
					self.word_list.append(tuple(line))
					self.label_list.append(True)
				if mode==1:
					line=line.split(',')
					line=line[:-1] + [(line[-1].strip('\n'))]
					self.word_list.append(tuple(line))
					self.label_list.append(False)
				if mode==2:
					self.alphabet=line.split(',')





#################
#Recording tests#
#################
FIELD_NAMES_RNN_TRAINING = [ "Model name",
				"LSTM learning time", "LSTM hidden dimension", "LSTM layers", 
				"LSTM test accuracy", "LSTM value accuracy", "LSTM dataset learning",
				"LSTM dataset testing"]

FIELD_NAMES_MODEL_CHECKING = ["Model name", "Technique",
								"Counterexample", "Specification", "Extracted DFA size", "Total Time"]
RNN_train_info = {f: None for f in FIELD_NAMES_RNN_TRAINING}
Model_Check_info = {f: None for f in FIELD_NAMES_MODEL_CHECKING}


def write_csv_header():
	
	filename = "../models/trained_rnn_info.csv"
	FIELD_NAMES = FIELD_NAMES_RNN_TRAINING
	with open(filename, mode='w') as employee_file:
		writer = csv.DictWriter(employee_file, fieldnames=FIELD_NAMES)
		writer.writeheader()
	

	filename = "../models/learned_dfa_info.csv"
	FIELD_NAMES = FIELD_NAMES_MODEL_CHECKING
	with open(filename, mode='w') as employee_file:
		writer = csv.DictWriter(employee_file, fieldnames=FIELD_NAMES)
		writer.writeheader()


def write_line_csv(purpose):
	if purpose=="RNN_training":
		filename = "../models/trained_rnn_info.csv"
		FIELD_NAMES = FIELD_NAMES_RNN_TRAINING
		info = RNN_train_info

	if purpose=="Model_Checking":
		filename = "../models/learned_dfa_info.csv"
		FIELD_NAMES = FIELD_NAMES_MODEL_CHECKING
		info = Model_Check_info
		print(info)

	with open(filename, mode='a') as benchmark_summary:
		writer = csv.DictWriter(benchmark_summary, fieldnames=FIELD_NAMES)
		writer.writerow(info)



###############
#Training RNNs#
###############
def train_and_save_rnn(train_data, test_data, alphabet, embedding_dim=10, hidden_dim =30, num_layers=2, batch_size=1000, epoch=1, save_dir="."):
	
	last_noted_time = time.time()

	model = RNNLanguageClasifier()
	model.train_a_lstm_dataset(hidden_dim=hidden_dim,
					   train_data=train_data,
					   test_data=test_data,
					   alphabet=alphabet,	
					   num_layers=num_layers,
					   embedding_dim=embedding_dim,
					   batch_size=batch_size,
					   epoch=epoch,
					   save_dir=save_dir)
	training_time = time.time() - last_noted_time

	RNN_train_info.update({"Model name": "{}".format(save_dir),
					  "LSTM learning time": "{:.4f}".format(training_time),
					  "LSTM hidden dimension": hidden_dim,
					  "LSTM layers": num_layers,
					  "LSTM test accuracy": "{:.4f}".format(model.test_acc),
					  "LSTM value accuracy": "{:.4f}".format(model.val_acc),
					  "LSTM dataset learning": len(train_data.label_list),
					  "LSTM dataset testing": model.num_of_test})
	
	write_line_csv("RNN_training")


################
#Verifying RNNs#
################
def verification_methods(rnn, spec_tuple, rnn_model_name, TO=600):

	#3 methods for verification
	method = 0
	while method<3:
		for k in Model_Check_info.keys():
			Model_Check_info[k]=None
		
		Model_Check_info.update({"Model name": rnn_model_name,
							"Specification": spec_tuple[0]})
		
		if method==0:
			Model_Check_info.update({"Technique": "AAMC"})
			print("Performing AAMC")
			abstract_and_check(rnn, spec_tuple, rnn_model_name, TO)

		if method==1:
			Model_Check_info.update({"Technique": "PDV"})
			print("Performing PDV")
			learn_and_check(rnn, spec_tuple, rnn_model_name, TO)
		'''
		if method==2:
			Model_Check_info.update({"Technique": "SMC"})
			print("Performing SMC")
			statistical_check(rnn, spec_tuple, rnn_model_name, TO)
		'''

		write_line_csv("Model_Checking")
		method+=1


################################
#Property Directed Verification#
################################
def learn_and_check(rnn, spec_tuple, rnn_model_name, TO):


	teacher_pac = PACTeacher(rnn) 
	student = DecisionTreeLearner(teacher_pac)
	checker = DFAChecker(spec_tuple[0])

	print("------------Starting DFA extraction---------------")
	last_noted_time = time.time()
	counter = teacher_pac.check_and_teach(student, checker, TO)
	total_time = time.time() - last_noted_time
	Model_Check_info.update({"Technique": "PDV", "Total Time": total_time})

	if counter is None:
		print("No mistakes found ==> DFA learned:")
		Model_Check_info.update({"Counterexample": None,
						  "Extracted DFA size": len(student.dfa.states)})
	else:
		print("Mistakes found ==> Counter example: {}".format(counter))
		Model_Check_info.update({"Counterexample": counter,
						  "Specification": spec_tuple[0],
						  "Extracted DFA size": len(student.dfa.states)})
	return student.dfa
		


############################
#Statistical Model Checking#
############################
def statistical_check(rnn, spec_tuple, rnn_model_name, TO):

	alph = rnn.alphabet
	test_size = np.log(2 / confidence) / (2 * error * error)#change the confidence and width according to chernoff_hoeding
	batch_size = 200
	checker_time = 0
	counter = None


	last_noted_time = time.time()
	for i in range(int(test_size // batch_size) + 1):
			
		batch = [random_nonempty_word(alph) for i in range(batch_size)]
		for x, y, w in zip(rnn.is_words_in_batch(batch), [spec_tuple[0].is_word_in(w) for w in batch],
						   batch):
			
			checker_time = time.time() - last_noted_time
			if checker_time > TO:
				print("STATMC timed out!!!")
				break

			if x and (not y):
				counter = w
				break
		else:
			continue
		break

	total_time = time.time() - last_noted_time
	Model_Check_info.update({"Model name": rnn_model_name,
							"Extracted DFA size": None,
							"Total Time": total_time,
							"Technique": "Statistical"})
	
	if counter is None:
		print("Checked on test suite ==> No mistakes found:")
		Model_Check_info.update({"Counterexample": None,
						  "Specification": spec_tuple[0]})
	else:
		print("Checked on test suite ==> Mistakes found ==> Counter example: {}".format(counter))
		Model_Check_info.update({"Counterexample": counter,
						  "Specification": spec_tuple[0]})


	return None




#######################
#Automaton Abs. and MC#
#######################
def abstract_and_check(rnn, spec_tuple, rnn_model_name, TO):
	

	teacher_pac = PACTeacher(rnn) 
	student = DecisionTreeLearner(teacher_pac)


	
	print("------------Starting DFA extraction---------------")
	
	last_noted_time = time.time()
	teacher_pac.teach(student, TO)
	checker = DFAChecker(spec_tuple[0])
	counter = checker.check_for_counterexample(student.dfa)
	if not rnn.is_word_in(counter):
		print("This check failed but counter", counter)
		counter = None
	total_time = time.time() - last_noted_time

	Model_Check_info.update({"Model name": rnn_model_name,
	 						"Technique": "Abstraction-based",
	 						"Total Time": total_time})


	if counter is None:
		print("DFA learned ==> No mistakes found:")
		Model_Check_info.update({"Counterexample": None,
						  "Specification": spec_tuple[0],
						  "Extracted DFA size": len(student.dfa.states)})
	else:
		print("DFA learned ==> Mistakes found ==> Counter example: {}".format(counter))
		Model_Check_info.update({"Counterexample": counter,
						  "Specification": spec_tuple[0],
						  "Extracted DFA size": len(student.dfa.final_states)})
	

	return student.dfa



def check_contact_sequence_RNN():
	
	write_csv_header()

	foldername = '../models/GeneratedData'
	dirs = ["scc2034_kilifi_all_contacts_across_households",
		"tij_InVS",
		"highschool_2011",
		"detailed_list_of_contacts_Hospital",
		"scc2034_kilifi_all_contacts_within_households",
		"tij_SFHH",
		"tij_InVS15"]


	for d in dirs:
		data_dir = foldername+'/'+d
		d_train=Data(filename=data_dir+'/train.data')
		d_test=Data(filename=data_dir+'/test.data')	
		spec_dfa = load_dfa_dot_TN(filename=data_dir+'/spec_dfa.dot')
		print("Starting test for filename:", data_dir)
			
		rnn = RNNLanguageClasifier()
		rnn.load_lstm(data_dir)
	

		last_noted_time = time.time()
		print("----------Verification starting----------")
		
		verification_methods(rnn=rnn, spec_tuple=(spec_dfa, 'DFA'), rnn_model_name=d)

		print("----------Verification ended----------. \nTotal time {}".format(time.time()-last_noted_time))


def train_contact_sequence_RNN():

	write_csv_header()

	foldername = '../models/GeneratedData'
	dirs = ["scc2034_kilifi_all_contacts_across_households",
		"tij_InVS",
		"highschool_2011",
		"detailed_list_of_contacts_Hospital",
		"scc2034_kilifi_all_contacts_within_households",
		"tij_SFHH",
		"tij_InVS15"]

	for d in dirs:
		data_dir = foldername+'/'+d
		d_train=Data(filename=data_dir+'/train.data')
		d_test=Data(filename=data_dir+'/test.data')	
		spec_dfa = load_dfa_dot_TN(filename=data_dir+'/spec_dfa.dot')
		print("Starting test for filename:", data_dir)
			

		
		##############
		#RNN training#
		##############
		#tune hyperparameters if necessary
		hidden_dim = min(max(100, len(spec_dfa.states)), 500) 
		num_layers = min(max(2, len(spec_dfa.states)//100), 5)
		
		#uncomment this in order to train RNN on contact sequence data
		train_and_save_rnn(train_data=d_train, 
							test_data=d_test, 
							alphabet=d_train.alphabet,
							hidden_dim=hidden_dim, 
							num_layers=num_layers, 
							save_dir=data_dir)


check_contact_sequence_RNN()
'''
if __name__ == "__main__":
	main()
 '''
