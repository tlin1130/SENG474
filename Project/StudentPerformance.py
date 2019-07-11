# SENG 474 Data Mining Project
# Team Members: 
# Han-wei Lin, Andrew Yang, Kibae Kim
# 
# StudentPerformance.py builds a decision tree classification model
# Training dataset: student-mat.csv from the UCI machine learning repository
#
# Citation:
# P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance.
# In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) 
# pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
#
#

import sys
import csv
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import pydotplus as pydot
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
from random import randint

# global variables
feature_names = None
class_names = ['0','50','55','60','65','70','75','80','85','90','95','100','20','25','30','35','40','45']
student_records = []
student_targets = []
total_accuracy = 0
total_average_error = 0;
total_binary_correct = 0

# preprocess csv file of dataset
def preprocessData(filename):

	global feature_names
	global student_records
	global student_targets

	# read from csv file
	with open(filename) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=';')
		
		i = 0
		for row in readCSV:
			if i == 0:
				feature_names = row
				feature_names.remove('G1')
				feature_names.remove('G2')
				feature_names.remove('G3')

				i = 1
			else:
				# discard attr 31 & 32
				record = row[0:30]
				target = row[32]
				student_records.append(record)
				student_targets.append(target)

	# change attribute values to numeric values
	for x in range(0,395):
		# school
		if student_records[x][0] == 'GP':
			student_records[x][0] = '0'
		else:
			student_records[x][0] = '1'
		# sex
		if student_records[x][1] == 'F':
			student_records[x][1] = 0
		else:
			student_records[x][1] = 1
		# address
		if student_records[x][3] == 'U':
			student_records[x][3] = 0
		else:
			student_records[x][3] = 1
		# famsize
		if student_records[x][4] == 'LE3':
			student_records[x][4] = 0
		else:
			student_records[x][4] = 1
		# pstatus
		if student_records[x][5] == 'T':
			student_records[x][5] = 0
		else:
			student_records[x][5] = 1
		# schoolsup
		if student_records[x][15] == 'yes':
			student_records[x][15] = 1
		else:
			student_records[x][15] = 0
		# famsup
		if student_records[x][16] == 'yes':
			student_records[x][16] = 1
		else:
			student_records[x][16] = 0
		# paid
		if student_records[x][17] == 'yes':
			student_records[x][17] = 1
		else:
			student_records[x][17] = 0
		# activities
		if student_records[x][18] == 'yes':
			student_records[x][18] = 1
		else:
			student_records[x][18] = 0
		# nursey
		if student_records[x][19] == 'yes':
			student_records[x][19] = 1
		else:
			student_records[x][19] = 0
		# higher
		if student_records[x][20] == 'yes':
			student_records[x][20] = 1
		else:
			student_records[x][20] = 0
		# internet
		if student_records[x][21] == 'yes':
			student_records[x][21] = 1
		else:
			student_records[x][21] = 0
		# romantic
		if student_records[x][22] == 'yes':
			student_records[x][22] = 1
		else:
			student_records[x][22] = 0
		# nominal attributes (using binary split)
		# guardian
		if student_records[x][11] == 'other':
			student_records[x][11] = 1
		else:
			student_records[x][11] = 0
		#Mjob
		if student_records[x][8] == 'teacher':
			student_records[x][8] = 1
		else:
			student_records[x][8] = 0
		#Fjob
		if student_records[x][9] == 'teacher':
			student_records[x][9] = 1
		else:
			student_records[x][9] = 0
		# reason
		if student_records[x][10] == 'course' or student_records[x][10] == 'reputation':
			student_records[x][10] = 1
		else:
			student_records[x][10] = 0


# train decision tree classifier
def trainDecisionTreeClassifier(X,Y,m):
	clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=m)
	clf = clf.fit(X, Y)
	return clf

# generate graph.pdf which contains graphical representation of trained decision tree 
def generateGraph(clf,f,c):
	dot_data = tree.export_graphviz(clf, out_file=None,feature_names=f,class_names=c)
	graph = graphviz.Source(dot_data)
	graph.render("graph")

if __name__ == '__main__':
	

	filename_math = "student-mat.csv"
	preprocessData(filename_math)

	test = 1
	while test < 11:
		student_records_train = []
		student_targets_train = []

		# random select instances for training
		for i in range(0,300):
			index = randint(0,394)
			#print index
			record = student_records[index]
			target = student_targets[index]
			student_records_train.append(record)
			student_targets_train.append(target)

		student_records_test = []
		student_targets_test = []

		# random select instances for testing
		for i in range(0,100):
			index = randint(0,394)
			#print index
			record = student_records[index]
			target = student_targets[index]
			student_records_test.append(record)
			student_targets_test.append(target) 

		clf = trainDecisionTreeClassifier(student_records_train,student_targets_train,100)
		results = clf.predict(student_records_test)

		

		total_difference = 0

		# compute correct predictions
		correct = 0
		for i in range(0,100):
			prediction = results[i]
			true = student_targets_test[i]
			if (prediction == true):
				correct = correct + 1 
			elif (prediction != true):
				difference = abs(int(true) - int(prediction))
				total_difference = total_difference + difference

		average_difference = total_difference / (100 - correct)
		average_difference = average_difference

		print 'Test' + str(test)
		print '_____ Multi-class predictions _____'
		print 'Accuracy = ' + str(correct) + '%'
		print 'Average error = ' + str(average_difference) +  ' grades'
		total_accuracy = total_accuracy + correct
		total_average_error = total_average_error + average_difference

		f = 0
		p = 1
		binary_correct = 0 
		binary_results = []
		binary_true = []
		for i in range(0,100):
			if (results[i] < 10):
				binary_results.append(f)
			else:	
				binary_results.append(p)

		for i in range(0,100):
			if (student_targets_test[i] < 10):
				binary_true.append(f)
			else:
				binary_true.append(p)

		for i in range(0,100):
			if (binary_results[i] == binary_true[i]):
				binary_correct = binary_correct + 1

		print '_____ Binary-class predictions _____'
		print 'Accuracy = ' + str(binary_correct) + '%'
		total_binary_correct = total_binary_correct + binary_correct

		test = test + 1


	average_a = total_accuracy / 10
	average_e = total_average_error / 10
	average_b = total_binary_correct / 10
	print '========================='
	print '_____ Multi-class predictions summary_____'
	print 'Average accuracy = ' + str(average_a) + '%'
	print 'Average of average error per test = ' + str(average_e) + ' grades' 
	print '_____ Binary-class predictions summary_____'
	print 'Average accuracy = ' + str(average_b) + '%'


	generateGraph(clf, feature_names,class_names)

