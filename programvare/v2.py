#-----------------NN for gjenkjenning av enkle bilder, utrent nett --> kost paa 2.68. 
#-----------------er langt ifra optimalisert, men skal gi forstaaelse

import math
import simplejson as json
import numpy
from pprint import pprint
import random

PICS_LOAD = 5

class NeuralNetwork(object):
	def __init__(self):
		self.dimInput 	= 64
		self.dimL1 		= 16
		self.dimL2 		= 16
		self.dimOutput 	= 8
		self.batchSize  = 64
		self.set()

	def set(self):
		self.Input 	= [0 for x in range(self.dimInput)]
		self.L1 	= [0 for x in range(self.dimL1)]
		self.L2 	= [0 for x in range(self.dimL2)]
		self.Output = [0 for x in range(self.dimOutput)]

		self.W1	= [[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2	= [[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3	= [[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]

		self.L1Change 	= [0 for x in range(self.dimL1)]
		self.L2Change 	= [0 for x in range(self.dimL2)]
		self.OutputChange = [0 for x in range(self.dimOutput)]

		self.W1Change = [[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2Change = [[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3Change = [[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]

		self.W1TotalChange = [[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2TotalChange = [[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3TotalChange = [[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]

	def mod(self, dimInput, dimL1, dimL2, dimOutput):
		self.dimInput 	= dimInput
		self.dimL1 		= dimL1
		self.dimL2 		= dimL2
		self.dimOutput 	= dimOutput
		self.set()

	def shuffleWeights(self):
		for a in range(self.dimL1): 
			for b in range(self.dimInput):
				self.W1[b][a] = random.uniform(-1, 1)
		for a in range(self.dimL2): 
			for b in range(self.dimL1):
				self.W2[b][a] = random.uniform(-1, 1)
		for a in range(self.dimOutput): 
			for b in range(self.dimL2):
				self.W3[b][a] = random.uniform(-1, 1)

	def forwProp(self):
		for a in range(self.dimInput):
			self.Input[a] = float(self.Input[a]) / 255

		tmp = 0
		for a in range(self.dimL1):
			for b in range(self.dimInput):
				tmp += self.W1[b][a] * self.Input[b]
			self.L1[a] = sigmoid(tmp)
		tmp = 0
		for a in range(self.dimL2):
			for b in range(self.dimL1):
				tmp += self.W2[b][a] * self.L1[b]
			self.L2[a] = sigmoid(tmp)
		tmp = 0
		for a in range(self.dimOutput):
			for b in range(self.dimL2):
				tmp += self.W3[b][a] * self.L2[b]
			self.Output[a] = sigmoid(tmp)

	def backProp(self): 
		for a in range(self.dimL2):
			for b in range(self.dimOutput):
				self.L2Change[a] += self.OutputChange[b] * self.W3[a][b] #onsket endring i "celler"

		for a in range(self.dimL1):
			for b in range(self.dimL2):
				self.L1Change[a] += self.L2Change[b] * self.W2[a][b]

		for a in range(self.dimL2):
			for b in range(self.dimOutput):
				self.W3Change[0][0] = self.OutputChange[0] * self.L2Change[0] * abs(self.W3[0][0])


	def calculateCost(self, goal):
		self.goal = goal
		self.cost = 0
		for x in range(self.dimOutput):
			self.OutputChange[x] = self.Output[x] - self.goal[x]
			self.cost += (self.goal[x] - self.Output[x])**2

	def netCost(self, setSize): 
		self.setCost = 0
		for x in range(setSize): 
			thisPicType = loadNextPic()
			self.forwProp()
			self.calculateCost(thisPicType)
			self.setCost += self.cost
		self.setCost /= setSize	


	def printStuff(self):
		#pprint(self.Input)
		#pprint(self.goal)
		#pprint(self.Output)
		pprint(self.L1Change)
		pprint(self.L2Change)

net = NeuralNetwork()

currentPic = 0
def loadPics(fromPic, toPic):
	pics = [0 for x in range(1 + toPic - fromPic)]
	with open('..\egne_bilder\data.json') as f:
		for i, line in enumerate(f):
		    if (i >= fromPic) and (i <= toPic):
		        pics[i - fromPic] = line
		    elif i > toPic:
		        break
	return pics

def loadNextPic():
	thisPicType = [0 for x in range(8)]
	global currentPic; global pics
	if currentPic % PICS_LOAD == 0:
		pics = loadPics(currentPic, currentPic + PICS_LOAD)
	buff = pics[currentPic % PICS_LOAD].split(",")
	for x in range(64):
		net.Input[x] = buff[x + 2]
	currentPic += 1
	for x in range(8):
		if x != int(buff[1]):
			thisPicType[x] = 0
		else:
			thisPicType[x] = 1
	return thisPicType

def sigmoid(inp):
	return  1 / (1 + numpy.exp(-inp))

def sigmoidDer(inp):
	return numpy.exp(-inp)/((1 + numpy.exp(-inp))**2)



net.shuffleWeights()
thisPicType = loadNextPic()
net.forwProp()
net.calculateCost(thisPicType)
net.backProp()
net.printStuff()