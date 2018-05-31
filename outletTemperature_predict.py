import neurolab as nl
import numpy as np
import csv
import copy


def load_dataset():
    global inputData
    inputData1 = []
    output = []
    with open("outletTempSir.csv", "rb") as csvfile:
        creader = csv.reader(csvfile, delimiter=",")
        next(creader)
        # for temp in creader:
        temp = list(creader)
        print len(temp[1]), "temp", temp[1]
        # print len(temp[0])
        for each in temp:
            inputData1.append([float(i) for i in each])
        # print "INPUTE: ",len(inputData1),type(inputData1[0][0]),len(inputData1[0])
        # exit(0)
        inputData = copy.deepcopy(inputData1)

        # removing 0th column
        for each in inputData:
            # each.remove(each[7])
            each.remove(each[7])
        print "inp", len(inputData[0])
        input2 = np.asarray(inputData).reshape(len(inputData), 34)
        normf = nl.tool.Norm(input2)
        inpN = normf(input2)

        input = inpN[:3000]
        print input[:2], input.shape
        # print(len(inputData[0]), len(inputData), inputData[0])
        # append the rest of the outputs
        for each in temp:
            output.append(float(each[7]))
        output = np.asarray(output).reshape(len(output), 1)
        normf = nl.tool.Norm(output)
        tarN = normf(output)
        print tarN[:2], tarN.shape
        target = tarN[:3000]
        # print "input",input[:2],len(input),len(input[0])

        print "target", target[:20], len(target), len(target[0])

        ran = []
        for i in range(34):
            ran.append([0, 1.1])
        # print "Ran",ran[:3],len(ran),len(ran[0])
        net = nl.net.newff(ran, [9, 1])
        # print "shape:",input.shape[1],net.ci
        error = net.train(input, target, epochs=500, show=100)
        net.save('powerall.net')
        storedNet = nl.load('powerall.net')
        # outN = storedNet.sim(inp)
        acc = 0
        err = 0
        count = 0
        for i in range(3000, 11000):
            count += 1
            p = storedNet.sim([inpN[i]])
            print "pred", p, tarN[i]
            if (abs(p - tarN[i])) > 0.1:
                err += 1
            else:
                acc += 1
        print "accuracy", (float(acc) / count) * 100

        # print error,"error"'''


def dummy():
    net = nl.net.newff([[0, 1], [0, 1]], [4, 1])
    input1 = [[10, 20], [10, 12], [12, 220], [11, 122]]
    output = [0, 1, 1, 0]
    input = np.asarray(input1).reshape(4, 2)
    target = np.asarray(output).reshape(4, 1)
    size = len(input)
    normf = nl.tool.Norm(input)
    inpN = normf(input)
    print inpN, inpN.shape
    normf = nl.tool.Norm(target)
    tarN = normf(target)
    print tarN, tarN.shape
    # print input,len(input),len(input[0])
    # print target,len(target),len(target[0])
    '''net = nl.net.newff([[0,1], [0,1]], [4, 1])
    print input,target,"!!!!"
    print "shape:",input.shape[1],net.ci
    error = net.train(input, target, show=15)
    #net.sim([[0,0]])
    print error,"error"
    print "pred",net.sim([[0,1]])'''


# dummy()
load_dataset()
