import numpy as np
import random
import math
import argparse
import csv
import sys
import os
import time
from dfa import *

#Probabilty with which same node is repeated in a time resepecting path
epsilon = 0.01


#very interesting temporal graphs
#datasets taken from http://www.sociopatterns.org/datasets/
class TemporalGraph:
	def __init__(self, filename=None):
		self.nodes = [] #list of ids
		self.temporal_edges = {} #adjacency list of edges with time labels
		self.edges = {}
		self.events= {}
		self.filename = filename
		if filename!=None:
			self.read_from_file(filename)

	def read_from_file(self, filename=None):
		with open(filename) as csvfile:
			reader = csv.reader(csvfile)
			self.start_time = float('inf')
			self.end_time= -float('inf')
			for row in reader:
				time_label=int(row[0])
				v1=str(row[1])
				v2=str(row[2])
				if time_label<self.start_time:
					self.start_time =time_label
				if time_label>self.end_time:
					self.end_time=time_label
				if v1 not in self.nodes:
					self.nodes.append(v1)
				if v2 not in self.nodes:
					self.nodes.append(v2)
				try:
					self.events[time_label].append((v1,v2))
				except:
					self.events[time_label]=[(v1,v2)]
				try:
					self.temporal_edges[v1].append((v2,time_label))
					if v2 not in self.edges[v1]:
						self.edges[v1].append(v2)
				except KeyError:
					self.temporal_edges[v1] = [(v2,time_label)]
					self.edges[v1]=[v2]
				try:
					self.temporal_edges[v2].append((v1,time_label))
					if v1 not in self.edges[v2]:
						self.edges[v2].append(v1)
				except KeyError:
					self.temporal_edges[v2] = [(v1,time_label)]
					self.edges[v2]=[v1]

	
	#generates a valid path starting from the start_time
	def generate_valid_path(self, max_path_length):
		curr_time = np.random.choice(list(self.events.keys()))
		curr_edge = self.events[curr_time][np.random.choice(len(self.events[curr_time]))]

		path=[curr_edge[0], curr_edge[1]]
		prev_node = curr_edge[0]
		curr_node =curr_edge[1]

		for _ in range(max_path_length-1):

			#Heuristic to have paths to avoid repetation of nodes
			poss_nodes, prev_nodes=[], []
			for i in self.temporal_edges[curr_node]:
				if (i[1]>=curr_time and i[0]!=prev_node):
					poss_nodes.append(i)
				if (i[1]>curr_time and i[0]==prev_node):
					prev_nodes.append(i)

			prev_node = curr_node
			
			
			if poss_nodes==[] and prev_nodes==[]:
				break
			
			if prev_nodes==[]:
				(curr_node, curr_time) = poss_nodes[np.random.choice(len(poss_nodes))]
			
			elif poss_nodes==[]:
				(curr_node, curr_time) = prev_nodes[np.random.choice(len(prev_nodes))]
			
			else:
				coin_toss = np.random.choice([0,1], p=[1-epsilon, epsilon])
				if coin_toss == 0:
					(curr_node, curr_time) = poss_nodes[np.random.choice(len(poss_nodes))]
				else:
					(curr_node, curr_time) = prev_nodes[np.random.choice(len(prev_nodes))]
			path.append(curr_node)

		return path

		


	#generates an invalid path which contains at least one edge break or temporal break
	def generate_invalid_path(self, max_path_length):
		breaks=0 
		count=0
		curr_time=np.random.choice(list(self.events.keys()))
		curr_edge = self.events[curr_time][np.random.choice(len(self.events[curr_time]))]

		path=[curr_edge[0]]
		curr_node=curr_edge[0]
		
		while count<max_path_length:
			count+=1
			if count==1:
				break_type = np.random.choice([1,2,3], p=[1/3,1/3,1/3])
			else:
				break_type = np.random.choice([0,1,2,3], p=[1/4,1/4,1/4,1/4])

			#temporal break
			if break_type == 0:
				breaks = 1
				try:
					min_time = min([i[1] for i in self.temporal_edges[curr_node] if i[0]==prev_node])
				except:
					min_time = curr_time
				poss_nodes = [i for i in self.temporal_edges[curr_node] if i[1] < min_time]
				if poss_nodes==[]:
					count-=1
					continue
				prev_node = curr_node
				(curr_node, curr_time) = poss_nodes[np.random.choice(len(poss_nodes))]
				path.append(curr_node)

			#edge break
			if break_type == 1:
				breaks = 1
				random_node = np.random.choice([node for node in self.nodes if node not in self.edges[curr_node]])
				poss_nodes = self.temporal_edges[random_node]
				if poss_nodes==[]:
					count-=1
					continue
				prev_node = curr_node
				(curr_node, curr_time) = poss_nodes[np.random.choice(len(poss_nodes))]
				path.append(curr_node)
			
			#no break
			if break_type == 2:

				poss_nodes = [i for i in self.temporal_edges[curr_node] if i[1]>=curr_time]
				if poss_nodes==[]:
					count-=1
					continue
				prev_node = curr_node
				(curr_node, curr_time) = poss_nodes[np.random.choice(len(poss_nodes))]
				path.append(curr_node)
			
			#node repetation 
			if break_type==3:
				breaks = 1
				prev_node = curr_node
				path.append(curr_node)



		#keeps trying to generate paths with breaks
		if breaks==0:
			self.generate_invalid_path(max_path_length)
		
		return path

	#take the minimum possible time at every instance will result in a valid path 
	def is_valid_path(self, path):
		curr_node = path[0]
		curr_time = min([i[1] for i in self.temporal_edges[curr_node]])
		
		for i in range(1,len(path)):
			next_node = path[i]
			if next_node not in self.edges[curr_node]:
				return False

			poss_nodes = [i for i in self.temporal_edges[curr_node] if i[0]==next_node and i[1]>=curr_time]
			if poss_nodes==[]:
				return False

			curr_node = next_node
			curr_time = min([i[1] for i in poss_nodes])
		return True


	def is_possible_path(self, path):
		curr_node = path[0]
		
		for i in range(1,len(path)):
			next_node = path[i]
			if next_node not in self.edges[curr_node]:
				return False
			curr_node = next_node
		
		return True

	def generate_spec_dfa(self, filename):
		alphabet = self.nodes

		transitions = {'init':{letter:letter for letter in alphabet}}
		final_states = [letter for letter in alphabet]
		
		for state in alphabet:
			state_transtion ={}
			for letter in alphabet:
				if letter in self.edges[state]:
					state_transtion.update({letter:letter})
				else:
					state_transtion.update({letter:'dead'})
			transitions.update({state:state_transtion})

		transitions.update({'dead':{letter:'dead' for letter in alphabet}})

		spec_dfa = DFA('init', final_states, transitions)

		os.makedirs(os.path.dirname(filename), exist_ok=True)
		save_dfa_dot(filename=filename , dfa=spec_dfa)



	def create_temporal_dataset(self, no_of_examples, save_file_name):
		outputfile = save_file_name
		print("Creating sample for file {} .... ".format(save_file_name))
		print("Number of examples in the sample {}".format(no_of_examples))
		pos_list, neg_list=[],[]
		for i in range(no_of_examples):
			completion_percent = int(float(i/no_of_examples)*100)
			path_length = np.random.choice([i for i in range(5,16)])
			sys.stdout.write('\r Completed {} {}/100'.format("#"*completion_percent, completion_percent))
			if i<no_of_examples//2:
				pos_list.append(self.generate_valid_path(path_length))
			else:
				neg_list.append(self.generate_invalid_path(path_length))

		os.makedirs(os.path.dirname(outputfile), exist_ok=True)
		with open(outputfile, 'w') as file:
				line_list=[]
				for word in pos_list:
					line = ''
					for letter in word:
						line += str(letter)+',' 
					line_list.append(line[:-1]+'\n')
				line_list.append('---\n')
				for word in neg_list:
					line = ''
					for letter in word:
						line += str(letter)+','
					line_list.append(line[:-1]+'\n')
				
				line_list.append('---\n')
				line=''
				for node in self.nodes:
					line+=str(node)+','
				line_list.append(line[:-1])

				file.writelines(line_list)



def main():

	#creating dataset for temporal paths




	temporal_data_dir = "../TemporalNetworks"
	root, dirs, files = list(os.walk(temporal_data_dir))[0]

	for file in files:
		if file.endswith('.csv'):
			#print("Creating Data for filename", file)
			g=TemporalGraph(temporal_data_dir+'/'+file)
			no_of_examples = 0
			no_of_edges = 0
			for i in g.temporal_edges.keys():
				no_of_examples += len(g.temporal_edges[i])
				no_of_edges += len(g.edges[i])
			save_file_name = '../TemporalNetworks/GeneratedData/'+file.split('.')[0]


			print(file)
			no_of_examples = 2*max(50000, no_of_examples)
			g.create_temporal_dataset(no_of_examples=(no_of_examples), save_file_name = save_file_name+'/train.data' )
			g.create_temporal_dataset(no_of_examples=(no_of_examples//5), save_file_name = save_file_name+'/test.data' )
			g.generate_spec_dfa(filename=save_file_name+'/spec_dfa.dot')	
	


if __name__ == "__main__":
	main()

