import csv
import neurolab as nl
import numpy as np
import copy

def load_dataset():
    global inputData
    inputData1 = []
    output = []
    global over_pred
    global under_pred
    over_pred = 0
    under_pred = 0
    #ran = []
    with open("output.csv", "rb") as csvfile:
        creader = csv.reader(csvfile, delimiter=",")
        next(creader)

        temp = list(creader)

        for each in temp:
            inputData1.append([float(i) for i in each])
		
        inputData = copy.deepcopy(inputData1)
        
        for each in inputData:
            each.remove(each[0])

        print "inp", len(inputData[0])
        inpN = np.asarray(inputData).reshape(len(inputData), len(inputData[0]))
        input = inpN[:5000]
        print "SHAPE:",input.shape
        #input = inpN[:5000]
        for each in temp:
            output.append(float(each[0]))
        #some seconds in advance
        output = output[10:]
        for i in range(1,10):
            output[-i] = 0
        tarN = np.asarray(output).reshape(len(output), 1)
        target =tarN[:5000]
        ran = []
        for i in range(27):
        	ran.append([0, 1.1]) #this was 1.1 before

	# Create network with 2 layers
	net = nl.net.newelm(ran,[10,1], [nl.trans.TanSig(),nl.trans.TanSig()])
	# Set initialized functions and init
	net.layers[0].initf = nl.init.InitRand([0,1], 'wb')
	net.init()
	# Train network
	net.trainf = nl.train.train_rprop
	for i in range(0,5000,20):
		if i!=0:
			storedNet = nl.load('powerall.net')
		input_temp = input[i:i+20]
		target_temp = target[i:i+20]
		error = net.train(input_temp, target_temp, epochs=1000, show=100,goal = 0.001)
		net.save('powerall.net')
	# Simulate network
	count = 0
    acc = 0
    err = 0
    storedNet = nl.load('powerall.net')
    for i in range(5000,14391):
            count += 1
            p = storedNet.sim([inpN[i]])
            if (abs(p - tarN[i])/float(tarN[i])) > 0.1:
                err += 1
            else:
                acc += 1
    print "accuracy", (float(acc) / count) * 100

load_dataset()	
