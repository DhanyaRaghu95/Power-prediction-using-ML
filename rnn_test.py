import csv
import neurolab as nl
import numpy as np
import copy

# test the pre-trained RNN 

def load_dataset():
    global inputData
    inputData1 = []
    output = []
    global over_pred
    global under_pred
    over_pred = 0
    under_pred = 0
    #ran = []
    with open("finalData.csv", "rb") as csvfile:
        creader = csv.reader(csvfile, delimiter=",")
        next(creader)

        temp = list(creader)

        for each in temp:
            #print "each:",each
            inputData1.append([float(i) for i in each])
		
        inputData = copy.deepcopy(inputData1)
        
        for each in inputData:
            each.remove(each[0])

        print "inp", len(inputData[0])
        #print "len of inputData: ",len(inputData)
        inpN = np.asarray(inputData).reshape(len(inputData), len(inputData[0]))
        input = inpN[:4000]
        print "SHAPE:",input.shape
        #input = inpN[:5000]
        for each in temp:
            output.append(float(each[0]))
        #some seconds in advance
        output = output[10:]
        for i in range(1,10):
            output[-i] = 0
        tarN = np.asarray(output).reshape(len(output), 1)
        target =tarN[:4000]
        ran = []
        for i in range(76):
        	ran.append([0, 1.1]) #this was 1.1 before

# load the neural net
    storedNet = nl.load('90.03.net')
    output = storedNet.sim(inpN[5000:14300])
    print "pred",output,output.shape
    
    print "test",output[:10],tarN[5000:5011]

load_dataset()	
