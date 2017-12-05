#-----------------NN for gjenkjenning av enkle bilder. utrent nett (tilfeldige vekter) --> kost paa 2.68, noyaktighet paa 12.50
#-----------------er langt ifra optimalisert, men skal gi forstaaelse

#matte shiet
import math
import numpy
import random

#visualisering
import sys
from pprint import pprint
import os
import Tkinter, tkMessageBox #til canvas


PICS_LOAD = 100

class NeuralNetwork(object):
	def __init__(self):
		self.dimInput 	= 64 #fast for disse bildene
		self.dimL1 		= 16 #variabel
		self.dimL2 		= 16 #variabel
		self.dimOutput 	= 8  #fast for disse bildene

		self.batchSize  = 16
		self.stepSize   = 0.1

		self.draw = 0
		self.echoStat = 1

		self.set()

	def set(self):
		#-----------------"lagene" og bias--------------
		self.Input 				= 	[0 for x in range(self.dimInput)]
		self.L1 				= 	[0 for x in range(self.dimL1)]
		self.L2 				= 	[0 for x in range(self.dimL2)]
		self.Output 			= 	[0 for x in range(self.dimOutput)]
		self.bias   			= 	[0 for x in range(3)]

		#-----------------vekter-------------------
		self.W1					= 	[[1 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2					= 	[[1 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3					= 	[[1 for x in range(self.dimOutput)] for x in range(self.dimL2)]
		self.biasW1 			= 	[1 for x in range(self.dimL1)]
		self.biasW2 			= 	[1 for x in range(self.dimL2)]
		self.biasW3 			= 	[1 for x in range(self.dimOutput)]

		#-----------------onkset endring i "lagene"-----------------
		self.L1Change 			= 	[0 for x in range(self.dimL1)]
		self.L2Change 			= 	[0 for x in range(self.dimL2)]
		self.OutputChange 		= 	[0 for x in range(self.dimOutput)]

		#-----------------onsket endring i vektene
		self.W1TotalChange  	= 	[[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2TotalChange  	= 	[[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3TotalChange  	= 	[[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]
		self.biasW1TotalChange 	= 	[0 for x in range(self.dimL1)]
		self.biasW2TotalChange	= 	[0 for x in range(self.dimL2)]
		self.biasW3TotalChange 	= 	[0 for x in range(self.dimOutput)]

	def mod(self, dimL1, dimL2, batchSize, stepSize, echoStat, draw):
		self.dimL1 		= dimL1
		self.dimL2 		= dimL2
		self.batchSize 	= batchSize
		self.stepSize 	= stepSize
		self.echoStat 	= echoStat
		self.draw 		= draw
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
		for a in range(self.dimL1): 
			self.biasW1[a] = random.uniform(-1, 1)
		for a in range(self.dimL2): 
			self.biasW2[a] = random.uniform(-1, 1)
		for a in range(self.dimOutput): 
			self.biasW3[a] = random.uniform(-1, 1)

	def forwProp(self):
		#-----------------sette inputtet paa 0-1 format i stedet for 0-255----------------
		for a in range(self.dimInput):
			self.Input[a] = float(self.Input[a]) / 255


		#-----------------gange inn vekter-------------------
		for a in range(self.dimL1):
			tmp = 0
			for b in range(self.dimInput):
				tmp += self.W1[b][a] * self.Input[b]
			self.L1[a] = sigmoid(tmp + self.biasW1[a])
			if(abs(tmp + self.biasW1[a]) > 100): 
				print("")
				print("Lag 1")
				print("Bias: " + str(self.biasW1[a]))
				print("Tmp: " + str(tmp))
				sys.exit("FOR STORT INPUT TIL SIGMOID")


		for a in range(self.dimL2):
			tmp = 0
			for b in range(self.dimL1):
				tmp += self.W2[b][a] * self.L1[b]
			self.L2[a] = sigmoid(tmp + self.biasW2[a])
			if(abs(tmp + self.biasW2[a]) > 100): 
				print("Lag 2")
				print("Bias: " + str(self.biasW2[a]))
				print("Tmp: " + str(tmp))
				print("Nummer: " + str(self.e))
				sys.exit("FOR STORT INPUT TIL SIGMOID")


		for a in range(self.dimOutput):
			tmp = 0
			for b in range(self.dimL2):
				tmp += self.W3[b][a] * self.L2[b]
			self.Output[a] = sigmoid(tmp + self.biasW3[a])
			if(abs(tmp + self.biasW3[a]) > 100): 
				print("Lag 3")
				print("Bias: " + str(self.biasW3[a]))
				print("Tmp: " + str(tmp))
				print("Nummer: " + str(self.e))
				sys.exit("FOR STORT INPUT TIL SIGMOID")


	def backProp(self): 
		self.L1Change 	= [0 for x in range(self.dimL1)]
		self.L2Change 	= [0 for x in range(self.dimL2)]

		#-----------------endring i "celler"------------------------
		for a in range(self.dimL2):
			for b in range(self.dimOutput):
				self.L2Change[a] += self.OutputChange[b] * self.W3[a][b]

		for a in range(self.dimL1):
			for b in range(self.dimL2):
				self.L1Change[a] += self.L2Change[b] * self.W2[a][b]

		#-----------------endring i vekter-----------------
		for a in range(self.dimL2):
			for b in range(self.dimOutput):
				self.W3TotalChange[a][b] += self.OutputChange[b] * self.L2[a] * abs(self.W3[a][b])
				# print(self.OutputChange[b] * self.L2[a])
				# print(self.OutputChange[b])
				# print(self.L2[a])
				# print("")
				# x = raw_input("")

		for a in range(self.dimL1):
			for b in range(self.dimL2):
				self.W2TotalChange[a][b] += self.L2Change[b] * self.L1[a] * abs(self.W2[a][b])

		for a in range(self.dimInput):
			for b in range(self.dimL1):
				self.W1TotalChange[a][b] += self.L1Change[b] * self.Input[a] * abs(self.W1[a][b])
		

		#-----------------bias-------------------
		for a in range(self.dimL1):
			self.biasW1TotalChange[a] += self.L1Change[a] * abs(self.biasW1[a] * 0.5)

		for a in range(self.dimL2):
			self.biasW2TotalChange[a] += self.L2Change[a] * abs(self.biasW2[a] * 0.5)

		for a in range(self.dimOutput):
			self.biasW3TotalChange[a] += self.OutputChange[a] * abs(self.biasW3[a] * 0.5)

	def train(self, nPics):
		if(self.echoStat): print("Trener nett...")
		if(self.draw): 
			self.nPics = nPics #gjor nPics tilgengelig hvis det trengs
			self.initGraph()

		nBatches = int(nPics / self.batchSize)
		for x in range(nBatches):
			#-----------------fancy lastebar------------------
			if(self.echoStat):
				sys.stdout.write('\r')
				sys.stdout.write("[%-19s] %d%%" % ('=' * int((20 * (0.5+x)) / nBatches), int((100 * (1 + x)) / nBatches)))
			
			net.trainBatch(x, nBatches)
		if(self.draw): self.endGraph()

	def trainBatch(self, batchNumber, nBatches):
		batchCost = 0
		#-----------------gaar igjennom en batch----------------
		for x in range(self.batchSize):
			self.thisPicType = loadNextPic()
			self.forwProp()
			self.calculateCost(self.thisPicType)

			self.backProp()

		#-----------------snitt av endringen til vekter og paavirke vektene--------------------------
		for a in range(self.dimL2):
			for b in range(self.dimOutput):
				self.W3TotalChange[a][b] /= self.batchSize #snitt
				self.W3[a][b] += (1 - abs(self.W3[a][b])) * self.W3TotalChange[a][b] * self.stepSize

		for a in range(self.dimL1):
			for  b in range(self.dimL2):
				self.W2TotalChange[a][b] /= self.batchSize
				self.W2[a][b] += (1 - abs(self.W2[a][b])) * self.W2TotalChange[a][b] * self.stepSize

		for a in range(self.dimInput):
			for b in range(self.dimL1):
				self.W1TotalChange[a][b] /= self.batchSize
				self.W1[a][b] += (1 - abs(self.W1[a][b])) * self.W1TotalChange[a][b] * self.stepSize

		#-----------------paavirke bias-------------------------
		for a in range(self.dimL1):
			self.biasW1TotalChange[a] /= self.batchSize
			self.biasW1[a] += (1 - abs(self.biasW1[a])) * self.biasW1TotalChange[a] * self.stepSize

		for a in range(self.dimL2):
			self.biasW2TotalChange[a] /= self.batchSize
			self.biasW2[a] += (1 - abs(self.biasW2[a])) * self.biasW2TotalChange[a] * self.stepSize

		for a in range(self.dimOutput):
			self.biasW3TotalChange[a] /= self.batchSize
			self.biasW3[a] += (1 - abs(self.biasW3[a])) * self.biasW3TotalChange[a] * self.stepSize
		

		#-----------------resetter vektendringene-----------------
		self.W1TotalChange = [[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2TotalChange = [[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3TotalChange = [[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]


	def evaluateWeights(self): #tester om alle vektene er gyldige
		for a in range(self.dimL1): 
			for b in range(self.dimInput):
				if(abs(self.W1[b][a]) > 100): sys.exit("ERROR: ugyldige vekter")
		for a in range(self.dimL2): 
			for b in range(self.dimL1):
				if(abs(self.W2[b][a]) > 100): sys.exit("ERROR: ugyldige vekter")
		for a in range(self.dimOutput): 
			for b in range(self.dimL2):
				if(abs(self.W3[b][a]) > 100): sys.exit("ERROR: ugyldige vekter")

	def calculateCost(self, goal):
		self.goal = goal
		self.cost = 0
		costHit = 0
		maxO = 0
		maxIndex = 0
		goalIndex = 0
		for x in range(self.dimOutput):
			self.OutputChange[x] = self.goal[x] - self.Output[x]
			self.cost += (self.goal[x] - self.Output[x])**2
			if(self.Output[x] > maxO):
				maxO = self.Output[x]
				maxIndex = x
			if(self.goal[x] == 1):
				goalIndex = x
		if(goalIndex == maxIndex):
			costHit = 1

		if(self.draw):
			self.drawGraph(costHit, self.cost)
		return costHit


	def netCost(self, setSize): 
		self.draw = 0 #skal ikke tegne graf ved testing
		self.setCost = 0
		self.realSetCost = 0
		if(self.echoStat): print("\n\nTester nett...")
		for x in range(setSize):
			self.thisPicType = loadNextPic()
			self.forwProp()
			self.realSetCost += self.calculateCost(self.thisPicType)
			self.setCost += self.cost

			#-----------------fancy lastebar------------------
			if(self.echoStat):
				sys.stdout.write('\r')
				sys.stdout.write("[%-19s] %d%%" % ('=' * int((20 * (0.5+x)) / setSize), int((100 * (1 + x)) / setSize)))
				sys.stdout.flush()
		self.setCost /= setSize
		self.realSetCost = 100 * (float(self.realSetCost) / setSize)
		if(self.echoStat):
			print("\n\n")
			print("Kost: " + str(self.setCost))
			print("Noyaktighet: " + str(self.realSetCost) + "%")
			print("\n")			
		return self.realSetCost

	def initGraph(self):
		self.cnvH = 720
		self.cnvW = 1350
		# if(self.cnvW * 8 > self.nPics):
		# 	self.drawBatchSize = 8
		# else:
		self.drawBatchSize = int(self.nPics / self.cnvW)
		self.thisPicNumberDraw = 0
		self.graphBatchHit = 0
		self.graphBatchCost = 0

		self.elForHit = [0 for x in range(self.cnvW)]
		self.elForCost = [0 for x in range(self.cnvW)]
		self.yCoorsHit = [0 for x in range(self.cnvW)]
		self.yCoorsCost = [(self.cnvH * 0.875) for x in range(self.cnvW)]
		self.xCoor = 0				

		self.master = Tkinter.Tk()
		self.canvas = Tkinter.Canvas(self.master, bg="white", height=self.cnvH, width=self.cnvW)
		
		for x in range(self.cnvW):
			self.elForHit[x] = self.canvas.create_oval(x, self.cnvH + 100, x + 3, self.cnvH + 100 + 3, width = 0, fill="red")
		for x in range(self.cnvW):
			self.elForCost[x] = self.canvas.create_oval(x, self.cnvH + 100, x + 3, self.cnvH + 100 + 3, width = 0, fill="blue")

		# if(self.cnvW * 8 > self.nPics): 
		# 	nPicsInGraph = float(self.cnvW * 8)
		# else:
		nPicsInGraph = self.nPics

		axisDim = 10
		if(min(self.cnvW, self.cnvH) < 200):
			axisDim = 5
		yAxis = int(nPicsInGraph)
		yAxis2 = 0
		base = 10
		yAxisExtra = 1
		while(yAxis > axisDim):
			yAxis /= base
			yAxis2 += 1
		while(yAxis < axisDim):
			yAxis *= 2
			yAxisExtra *= 2



		for x in range(axisDim):
			self.canvas.create_line(0, (x * self.cnvH) / axisDim, self.cnvW, (x * self.cnvH) / axisDim, width = 0, fill="gray")
			self.canvas.create_text(self.cnvW - 23, (x * self.cnvH) / axisDim, text=str((100 * (axisDim - x)) / axisDim) + "%", anchor="nw", fill="red")
			self.canvas.create_text(self.cnvW - 15, (x * self.cnvH) / axisDim + 12, text=str((2 * float(axisDim - x) / axisDim)), anchor="nw", fill="blue")  
		for x in range(yAxis):
			self.canvas.create_line((x * self.cnvW) / yAxis, 0, (x * self.cnvW) / yAxis, self.cnvH, width = 0, fill="gray")
			self.canvas.create_text((x * self.cnvW) / yAxis, self.cnvH - 10, text=str((x * (base**yAxis2) / yAxisExtra)), anchor="nw")


		self.canvas.pack()
		self.canvas.update()

	def endGraph(self):
		self.master.mainloop()

	def drawGraph(self, hit, cost):
		self.graphBatchHit += hit
		self.graphBatchCost += cost
		self.thisPicNumberDraw += 1
		if(self.thisPicNumberDraw % self.drawBatchSize == 0):
			lastTenHit = 0
			lastTenCost = 0
			self.yCoorsHit[min(self.xCoor, self.cnvW - 1)] = self.cnvH * float(self.graphBatchHit) / float(self.drawBatchSize)
			self.yCoorsCost[min(self.xCoor, self.cnvW - 1)] = self.cnvH * float(self.graphBatchCost) / float(self.drawBatchSize * 2)

			for x in range(24):
				lastTenHit += self.yCoorsHit[min(self.cnvW - 1, abs(self.xCoor - x))]
				lastTenCost += self.yCoorsCost[min(self.cnvW - 1, abs(self.xCoor - x))]
		
			lastTenHit = int(self.cnvH - (lastTenHit / 24))
			lastTenCost = int(self.cnvH - (lastTenCost / 24))


			x1, y1, x2, y2 = self.canvas.bbox(self.elForHit[min(self.cnvW - 1, self.xCoor)])
			self.canvas.move(self.elForHit[min(self.cnvW - 1, self.xCoor)], 0, lastTenHit - y1)

			x1, y1, x2, y2 = self.canvas.bbox(self.elForCost[min(self.cnvW - 1, self.xCoor)])
			self.canvas.move(self.elForCost[min(self.cnvW - 1, self.xCoor)], 0, lastTenCost - y1)


			
			self.canvas.update()
			self.xCoor += 1
			self.graphBatchHit = 0
			self.graphBatchCost = 0


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

def derSigmoid(inp):
	return numpy.exp(-inp) / ( (1 + numpy.exp(-inp))**2 )

def inverseSigmoid(inp):
	return numpy.log10(inp / (1 - inp))

def testBest(n, m):
	avCost = 0
	tSize = 12
	global currentPic
	for x in range(tSize):
		printx = float((n * tSize) + x)
		printM = float(tSize * m)
		os.system('cls')
		sys.stdout.write("Jobber...\n\n")
		sys.stdout.write("[%-19s]"% ('=' * int((20 * printx) / printM)))
		sys.stdout.write(str((100 * printx) / printM) + "%")
		sys.stdout.write("\n\n    Totalt: " + str(n + 1) + " av " + str(m))
		sys.stdout.write("\n    Del: " + str(x + 1) + " av " + str(tSize) + "\n\n")
		sys.stdout.flush()
		currentPic = 0
		net.shuffleWeights()
		net.train(40000)
		avCost += net.netCost(2000)
	avCost /= tSize
	with open('dump.data', 'a') as f:
		f.write("L1:" + str(net.dimL1))
		f.write(" L2:" + str(net.dimL2))
		f.write(" BS:" + str(net.batchSize))
		f.write(" SS:" + str(net.stepSize))
		f.write(" R:" + str(avCost) + "\n")
		f.close() 

def testALOT():
	L1Settings 		= [  64,  48,    64,  48]
	L2Settings		= [  64,  48,    64,  48]
	batchSettings 	= [  16,  16,    16,  16]
	stepSettings 	= [ 0.1, 0.1,  0.08, 0.1]

	for x in range(0,len(L1Settings)):
		net.mod(L1Settings[x], L2Settings[x], batchSettings[x], stepSettings[x], 1, 0)
		testBest(x, len(L1Settings))

net.mod(42, 42, 16, 0.1, 1, 1)
net.shuffleWeights()
net.train(47000)
print("")
net.netCost(2000)
pprint(net.Output)
pprint(net.goal)