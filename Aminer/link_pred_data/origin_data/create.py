import os
import networkx as nx
import numpy as np

num_a = 16604
num_p = 12455

# new_list = []

# G = nx.Graph()
# i=0
# f = open("author_class.txt",'r',encoding="UTF-8")
# for line in f:
# 	c = line
# 	G.add_node(i)
# 	G.node[i]["class"] = int(c)
# 	i=i+1
# f.close()

# f = open("paper_class.txt",'r',encoding="UTF-8")
# for line in f:
# 	c = line
# 	G.add_node(i)
# 	G.node[i]["class"] = int(c)
# 	i=i+1
# f.close()

# # print(np.sort(G.nodes()))
# # print(len(G.nodes()))

# f = open("Aminer_author2paper.txt",'r',encoding="UTF-8")
# for line in f:
# 	u, v = line.strip().split(' ')
# 	# new_list.append([str(int(u))+" "+str(int(v)+num_a)+"\n"])
# 	G.add_edge(int(u),int(v)+num_a)
# f.close()

# f = open("Aminer_coauthor.txt",'r',encoding="UTF-8")
# for line in f:
# 	u, v = line.strip().split(' ')
# 	# new_list.append([str(int(u))+" "+str(int(v))+"\n"])
# 	G.add_edge(int(u),int(v))
# f.close()

# f = open("Aminer_citation.txt",'r',encoding="UTF-8")
# for line in f:
# 	u, v = line.strip().split(' ')
# 	# new_list.append([str(int(u)+num_a)+" "+str(int(v)+num_a)+"\n"])
# 	G.add_edge(int(u)+num_a,int(v)+num_a)
# f.close()

# print(np.sort(G.nodes()))
# print(len(G.nodes()))

# isolate_node = []
# for i in G.nodes():
# 	if (G.degree(i)) == 0:
# 		# G.remove_node(i)
# 		isolate_node.append(i)

# for i in isolate_node:
# 	G.remove_node(i)

# print(np.sort(G.nodes()))
# print(len(G.nodes()))

# def graph_node_resort(graph):
#     g = nx.Graph()
#     g_nodes = list(graph.nodes())
#     g_edges = list(graph.edges())
#     add_edges_list = []
#     g.add_nodes_from(list(range(0,len(g_nodes))))

#     j=0
#     for i in G.nodes():
#     	g.node[j]["class"] = G.node[i]["class"]
#     	j=j+1

#     for u,v,_ in (graph.edges.data()):
#         # print(u,v)
#         add_edges_list.append((g_nodes.index(u),g_nodes.index(v)))
#     g.add_edges_from(add_edges_list)
#     return g

# clean_G = graph_node_resort(G)

# print(clean_G.node[0])
# print(np.sort(clean_G.nodes()))
# print(len(clean_G.nodes()))

# Save graph 
# nx.write_gml(clean_G, "./clean_G.gml") 

# Read graph
G = nx.read_gml('./clean_G.gml') 
print(len(G.nodes()))

# get the link_pred data

import random
from random import choice

# all_graph = open("Aminer_Graph.txt", 'w', encoding="UTF-8")
# for u,v,_ in G.edges.data():
# 	all_graph.writelines(u+" "+v+"\n")
# all_graph.close()

# isolate_node = []
# with open("isolate.txt","r",encoding="UTF-8") as f:
# 	for line in f:
# 		isolate_node.append(int(line)-num_a)

# i = 0
# paper_class = open("paper_class.txt", 'w', encoding="UTF-8")
# with open("../data/paper_class.txt", 'r', encoding="UTF-8") as f:
# 	for line in f:
# 		if i not in isolate_node:
# 			paper_class.writelines(line)
# 		i=i+1

# i=0
# paper_title = open("paper_title.txt", 'w', encoding="UTF-8")
# with open("../data/paper_title.txt", 'r', encoding="UTF-8") as f:
# 	for line in f:
# 		if i not in isolate_node:
# 			paper_title.writelines(line)
# 		i=i+1

# i=0
# paper_vectors = open("paper_vectors.txt", 'w', encoding="UTF-8")
# with open("../data/paper_vectors.txt", 'r', encoding="UTF-8") as f:
# 	for line in f:
# 		if i not in isolate_node:
# 			paper_vectors.writelines(line)
# 		i=i+1

# ap = open("Aminer_author2paper.txt", 'w', encoding="UTF-8")
# pp = open("Aminer_citation.txt", 'w', encoding="UTF-8")

# for u,v,_ in G.edges.data():
# 	if int(u)<num_a and int(v)>=num_a:
# 		s = str(u)+" "+str(int(v)-num_a)+"\n"
# 		print(s)
# 		ap.writelines(s)
# 	elif int(u)>=num_a and int(v)>=num_a:
# 		s = str(int(u)-num_a)+" "+str(int(v)-num_a)+"\n"
# 		print(s)
# 		pp.writelines(s)
# 	# print(u,v)
# ap.close()
# pp.close()

# print(np.sort(G.nodes()))
# print(len(G.nodes()))

# with open("Aminer_Graph.txt","w",encoding="UTF-8") as f:
# 	for i in new_list:
# 		f.writelines(i)
# 		print(i)
	# f.writelines(new_list[0])
	# f.writelines(new_list[1])