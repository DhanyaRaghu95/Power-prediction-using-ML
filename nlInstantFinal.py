import neurolab as nl
import numpy as np
import math
import random
import string
import csv
import copy
import pandas
import pylab as pl


def load_dataset():
    global inputData
    inputData1 = []
    output=[]
    with open("finalData.csv", "rb") as csvfile:
        creader = csv.reader(csvfile, delimiter=",")
        next(creader)
        # for temp in creader:
        temp = list(creader)
        # print len(temp[0])
        '''for index,each in enumerate(temp):
        	for i in range(len(each)):
        		if temp[index][i]=="":
        			print "true",i,index,temp[index][i]
        			exit(0)'''
        for each in temp:
        	#print type(each[0]),type(each),each[0]
        	inputData1.append([float(i) for i in each])
        #print "INPUTE: ",len(inputData1),type(inputData1[0][0]),len(inputData1[0])
        # exit(0)
        input=[]
        inputData = copy.deepcopy(inputData1)
        for i in range(len(inputData)):
        	input.append(inputData[i][:5] + inputData[i][29:])
        #removing 0th column
        for each in input:
            #each.remove(each[7])
            each.remove(each[0])
        input2 = np.asarray(input).reshape(len(input),52)
        #print input2[0]
        #exit(0)
        
        input = input2[:5000]
        #print input[0],input.shape
        #print(len(inputData[0]), len(inputData), inputData[0])
        # append the rest of the outputs
        for each in temp:
            output.append(float(each[0]))
        '''output = output[10:]
        for i in range(10):
        	output.append(0)
        print len(output)
        #exit(0)'''
        output = np.asarray(output).reshape(len(output),1)
        tarN = output
        
        #print tarN[:2],tarN.shape
        #exit(0)
        target = tarN[:5000]
        #print "input",input[:2],len(input),len(input[0])
        
        #print "target",target[:20],len(target),(target[9])
        
        ran=[]
        for i in range(52):
        	ran.append([0,1.1])
        #print "Ran",ran[:3],len(ran),len(ran[0])
        net = nl.net.newff(ran, [17,1])
        #print "shape:",input.shape[1],net.ci
        error = net.train(input, target, epochs=500, show=100)
        net.save('instMain.net')
        storedNet = nl.load('instMain.net')
	    #outN = storedNet.sim(inp)
        acc = 0
        err = 0
        count=0
        for i in range(3000,15300):
        	count+=1
        	p = storedNet.sim([input2[i]])
        	#print "pred",p,tarN[i]
        	if(abs(p - tarN[i])/float(tarN[i]))>0.1:
        		err+=1
        	else:
        		acc+=1
        print "accuracy", (float(acc) / count) * 100
        
load_dataset()
