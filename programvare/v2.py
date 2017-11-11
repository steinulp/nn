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
		self.set()

	def set(self):
		self.Input 	= [0 for x in range(self.dimInput)]
		self.L1 	= [0 for x in range(self.dimL1)]
		self.L2 	= [0 for x in range(self.dimL2)]
		self.Output = [0 for x in range(self.dimOutput)]

		self.W1	= [[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2	= [[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3	= [[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]

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

	def printWeight(self, n):
		pprint(self.Output)

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
		pass

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
	global currentPic; global pics
	if currentPic % PICS_LOAD == 0:
		pics = loadPics(currentPic, currentPic + PICS_LOAD)
	buff = pics[currentPic % PICS_LOAD].split(",")
	for x in range(64):
		net.Input[x] = buff[x + 2]
	currentPic += 1
	return buff[0]

def sigmoid(inp):
	return  1 / (1 + numpy.exp(-inp))

loadNextPic()
loadNextPic()


# net.mod(68, 10, 10, 1)
net.shuffleWeights()
net.forwProp()
net.printWeight(1)
