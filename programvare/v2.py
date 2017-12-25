#matte shiet
import numpy
import random

#visualisering
import sys
import os
from pprint import pprint
import Tkinter, tkMessageBox #til canvas

class NeuralNetwork(object):
	def __init__(self):
		#dimensjoner til de forskjellige lagene
		self.dimInput 	= 64 #fast for disse bildene
		self.dimL1 		= 16 #variabel
		self.dimL2 		= 16 #variabel
		self.dimOutput 	= 8  #fast for disse bildene

		#konstanter for laering
		self.batchSize  = 16
		self.stepSize   = 0.1

		#booleans for plot av data
		self.draw = 0
		self.echoStat = 1

		#canvas konstanter
		self.cnvH = 1050
		self.cnvW = 1900
		self.axisN = 10

		#variabel for midlertidig testing
		self.testvar = 0.5

		#lager arrayer ut ifra dataen over
		self.set()

	#setter arrayer
	def set(self):
		#-----------------nodene/cellene til de forskjellige lagene--------------
		self.Input 				= 	[0 for x in range(self.dimInput)]
		self.L1 				= 	[0 for x in range(self.dimL1)]
		self.L2 				= 	[0 for x in range(self.dimL2)]
		self.Output 			= 	[0 for x in range(self.dimOutput)]

		#-----------------vekter------------------------------------------------------
		self.W1					= 	[[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2					= 	[[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3					= 	[[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]
		self.biasW1 			= 	[0 for x in range(self.dimL1)]
		self.biasW2 			= 	[0 for x in range(self.dimL2)]
		self.biasW3 			= 	[0 for x in range(self.dimOutput)]

		#-----------------onkset endring i lagene------------------------------------
		self.L1Change 			= 	[0 for x in range(self.dimL1)]
		self.L2Change 			= 	[0 for x in range(self.dimL2)]
		self.OutputChange 		= 	[0 for x in range(self.dimOutput)]

		#-----------------onsket endring i vektene----------------------------------
		self.W1TotalChange  	= 	[[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2TotalChange  	= 	[[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3TotalChange  	= 	[[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]
		self.biasW1TotalChange 	= 	[0 for x in range(self.dimL1)]
		self.biasW2TotalChange	= 	[0 for x in range(self.dimL2)]
		self.biasW3TotalChange 	= 	[0 for x in range(self.dimOutput)]

	#endrer konstanter i nettet
	def mod(self, dimL1, dimL2, batchSize, stepSize, echoStat, draw):
		self.dimL1 		= dimL1
		self.dimL2 		= dimL2
		self.batchSize 	= batchSize
		self.stepSize 	= stepSize
		self.echoStat 	= echoStat
		self.draw 		= draw
		self.set()

	#setter vektene til tilfeldige verdier mellom -1 og 1
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

	#forward propagation
	def forwProp(self):
		#-----------------setter inputtet paa 0-1 format i stedet for 0-255----------------
		for a in range(self.dimInput):
			self.Input[a] = float(self.Input[a]) / 255

		#-----------------ganger inn vektene------------------------------
		for a in range(self.dimL1):
			tmp = 0 #til aa lagre summen av vektene gange input
			for b in range(self.dimInput):
				tmp += self.W1[b][a] * self.Input[b]
			self.L1[a] = self.activationFunc(tmp + self.biasW1[a]) #sum av vektene gange inputet pluss biasen

		for a in range(self.dimL2):
			tmp = 0
			for b in range(self.dimL1):
				tmp += self.W2[b][a] * self.L1[b]
			self.L2[a] = self.activationFunc(tmp + self.biasW2[a])

		for a in range(self.dimOutput):
			tmp = 0
			for b in range(self.dimL2):
				tmp += self.W3[b][a] * self.L2[b]
			self.Output[a] = self.activationFunc(tmp + self.biasW3[a])

	#backwards propagation
	def backProp(self): 
		#nullsetter endring i lagene
		self.L1Change 	= [0 for x in range(self.dimL1)]
		self.L2Change 	= [0 for x in range(self.dimL2)]

		#-----------------endring i noder for lag 1 og 2------------------------
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

		for a in range(self.dimL1):
			for b in range(self.dimL2):
				self.W2TotalChange[a][b] += self.L2Change[b] * self.L1[a] * abs(self.W2[a][b])

		for a in range(self.dimInput):
			for b in range(self.dimL1):
				self.W1TotalChange[a][b] += self.L1Change[b] * self.Input[a] * abs(self.W1[a][b])
		
		#-----------------endring i bias-------------------
		for a in range(self.dimL1):
			self.biasW1TotalChange[a] += self.L1Change[a] * abs(self.biasW1[a] * 0.5)

		for a in range(self.dimL2):
			self.biasW2TotalChange[a] += self.L2Change[a] * abs(self.biasW2[a] * 0.5)

		for a in range(self.dimOutput):
			self.biasW3TotalChange[a] += self.OutputChange[a] * abs(self.biasW3[a] * 0.5)

	#trener nettet over "nPics" antall bilder
	def train(self, nPics):
		global currentPic 
		if(self.echoStat): print("Trener nett...")
		if(self.draw): #hvis ting skal tegnes
			self.nPics = nPics #gjor nPics tilgengelig hvis det trengs
			self.initGraph() #lager en canvas som kan plottes paa

		nBatches = int(nPics / self.batchSize)
		for x in range(nBatches):
			currentPic = 0
			#-----------------fancy lastebar------------------
			if(self.echoStat):
				sys.stdout.write('\r')
				sys.stdout.write("[%-19s] %d%%" % ('=' * int((20 * (0.5+x)) / nBatches), int((100 * (1 + x)) / nBatches)))
			
			net.trainOneBatch(x, nBatches)
		print("batchOutputCost: " + str(self.batchOutputCost))
		print("batchL2Cost: " + str(self.batchL2Cost))
		print("batchL1Cost: " + str(self.batchL1Cost))

	def trainOneBatch(self, batchNumber, nBatches):
		self.batchOutputCost = 0
		self.batchL2Cost = 0
		self.batchL1Cost = 0
		batchCost = 0
		#-----------------gaar igjennom en batch----------------
		for x in range(self.batchSize):
			self.thisPicType = loadNextPic()
			self.forwProp()
			self.calculateCost(self.thisPicType)

			#costen i de forksjellige lagene for den siste batchen
			self.batchOutputCost += self.outputCost  
			self.batchL2Cost += self.L2Cost 
			self.batchL1Cost += self.L1Cost 

			self.backProp()

		#snitt av batchen
		self.batchOutputCost /= self.batchSize  
		self.batchL2Cost /= self.batchSize 
		self.batchL1Cost /= self.batchSize 

		#-----------------snitt av endringen til vekter og paavirke vektene--------------------------
		for a in range(self.dimL2):
			for b in range(self.dimOutput):
				self.W3TotalChange[a][b] /= self.batchSize #snitt
				self.W3[a][b] += self.activationFuncDer(self.W3[a][b]) * self.W3TotalChange[a][b] * self.stepSize * self.batchOutputCost

		for a in range(self.dimL1):
			for  b in range(self.dimL2):
				self.W2TotalChange[a][b] /= self.batchSize
				self.W2[a][b] += self.activationFuncDer(self.W2[a][b]) * self.W2TotalChange[a][b] * self.stepSize # * (self.batchL2Cost / 10)

		for a in range(self.dimInput):
			for b in range(self.dimL1):
				self.W1TotalChange[a][b] /= self.batchSize
				self.W1[a][b] += self.activationFuncDer(self.W1[a][b]) * self.W1TotalChange[a][b] * self.stepSize # * (self.batchL1Cost / 1000)

		#-----------------paavirke bias-------------------------
		for a in range(self.dimOutput):
			self.biasW3TotalChange[a] /= self.batchSize
			self.biasW3[a] += self.activationFuncDer(self.biasW3[a]) * self.biasW3TotalChange[a] * self.stepSize * self.batchOutputCost

		for a in range(self.dimL2):
			self.biasW2TotalChange[a] /= self.batchSize
			self.biasW2[a] += self.activationFuncDer(self.biasW2[a]) * self.biasW2TotalChange[a] * self.stepSize # * (self.batchL2Cost / 10)

		for a in range(self.dimL1):
			self.biasW1TotalChange[a] /= self.batchSize
			self.biasW1[a] += self.activationFuncDer(self.biasW1[a]) * self.biasW1TotalChange[a] * self.stepSize # * (self.batchL1Cost / 1000)
		

		#-----------------resetter vektendringene-----------------
		self.W1TotalChange = [[0 for x in range(self.dimL1)] for x in range(self.dimInput)]
		self.W2TotalChange = [[0 for x in range(self.dimL2)] for x in range(self.dimL1)]
		self.W3TotalChange = [[0 for x in range(self.dimOutput)] for x in range(self.dimL2)]

	def calculateCost(self, goal):
		self.goal = goal
		self.outputCost = 0
		self.L1Cost = 0
		self.L2Cost = 0

		costHit = 0
		maxO = 0
		maxIndex = 0
		goalIndex = 0

		#------------------------"cost" i de skjulte lagene-------------------
		for x in range(self.dimL1):
			self.L1Cost += (self.L1Change[x])**2
		for x in range(self.dimL2):
			self.L2Cost += (self.L2Change[x])**2

		for x in range(self.dimOutput):
			self.OutputChange[x] = self.goal[x] - self.Output[x]
			self.outputCost += (self.goal[x] - self.Output[x])**2
			if(self.Output[x] > maxO):
				maxO = self.Output[x]
				maxIndex = x
			if(self.goal[x] == 1):
				goalIndex = x
		if(goalIndex == maxIndex):
			costHit = 1

		if(self.draw):
			self.drawGraph(costHit, self.outputCost)
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
			self.setCost += self.outputCost

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

	def modCanvas(self, width, height, axisN):
		self.cnvH = height
		self.cnvW = width
		self.axisN = axisN

	def initGraph(self):
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

		nPicsInGraph = self.nPics

		if(min(self.cnvW, self.cnvH) < 200):
			self.axisN = 5
		yAxis = int(nPicsInGraph)
		yAxis2 = 0
		base = 10
		yAxisExtra = 1

		while(yAxis > self.axisN):
			yAxis /= base
			yAxis2 += 1
		if(yAxis == 0): yAxis = 1
		while(yAxis < self.axisN):
			yAxis *= 2
			yAxisExtra *= 2

		for x in range(self.axisN):
			self.canvas.create_line(0, (x * self.cnvH) / self.axisN, self.cnvW, (x * self.cnvH) / self.axisN, width = 0, fill="gray")
			self.canvas.create_text(self.cnvW - 23, (x * self.cnvH) / self.axisN, text=str((100 * (self.axisN - x)) / self.axisN) + "%", anchor="nw", fill="red")
			self.canvas.create_text(self.cnvW - 15, (x * self.cnvH) / self.axisN + 12, text=str((2 * float(self.axisN - x) / self.axisN)), anchor="nw", fill="blue")  
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

			for x in range(32):
				lastTenHit += self.yCoorsHit[min(self.cnvW - 1, abs(self.xCoor - x))]
				lastTenCost += self.yCoorsCost[min(self.cnvW - 1, abs(self.xCoor - x))]
		
			lastTenHit = int(self.cnvH - (lastTenHit / 32))
			lastTenCost = int(self.cnvH - (lastTenCost / 32))

			try: #tegn graf i vinduet
				x1, y1, x2, y2 = self.canvas.bbox(self.elForHit[min(self.cnvW - 1, self.xCoor)])
				self.canvas.move(self.elForHit[min(self.cnvW - 1, self.xCoor)], 0, lastTenHit - y1)

				x1, y1, x2, y2 = self.canvas.bbox(self.elForCost[min(self.cnvW - 1, self.xCoor)])
				self.canvas.move(self.elForCost[min(self.cnvW - 1, self.xCoor)], 0, lastTenCost - y1)
				
				self.canvas.update()
				self.xCoor += 1
				self.graphBatchHit = 0
				self.graphBatchCost = 0
			except Tkinter.TclError:
				print("\rERROR: Grafvindu lukket av bruker.\n")
				self.draw = 0
			except:
				print("\rERROR: Noe gikk galt med grafvinduet        " + str(sys.exc_info()[0]) + "\n")
				self.draw = 0

	def activationFuncDer(self, inp):
		return self.ReLUPrime(inp)
		#return self.sigmoidPrime(inp)
	def activationFunc(self, inp):
		#return self.ReLU(inp)
		return self.sigmoid(inp)

	def ReLU(self, inp):
		if(inp < 0):
			return 0
		elif(inp > 10):
			return 10
		else:
			return inp
	def ReLUPrime(self, inp):
		return 2 - abs(inp)

	def sigmoidPrime(self, inp):
		return numpy.exp(inp) / (numpy.exp(inp) + 1)**2
	def sigmoid(self, inp):
		if(inp > 700):
			return 1
		if(inp < -700):
			return 0
		return  1 / (1 + numpy.exp(-inp))

PICS_LOAD = 64
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


def testOneType(n, m):
	avCost = 0
	tSize = 16
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
		net.train(14000)
		avCost += net.netCost(4000)
	avCost /= tSize
	with open('dump.data', 'a') as f:
		f.write("L1:" + str(net.dimL1))
		f.write(" L2:" + str(net.dimL2))
		f.write(" BS:" + str(net.batchSize))
		f.write(" SS:" + str(net.stepSize))
		f.write(" TV:" + str(net.testVar))
		f.write(" R:" + str(avCost) + "\n")
		f.close() 

def testALOT():
	L1Settings 		= [  42,  42,  42,  42,  42,  42,  42,  42,  42]
	L2Settings		= [  42,  42,  42,  42,  42,  42,  42,  42,  42]
	batchSettings 	= [  16,  16,  16,  16,  16,  16,  16,  16,  16]
	stepSettings 	= [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
	testvar 		= [ 0.1, 0.5,   1,0.01, 0.8, 0.3, 0.9, 0.4, 0.7]

	for x in range(0, len(L1Settings)):
		net.testVar = testvar[x]
		net.mod(L1Settings[x], L2Settings[x], batchSettings[x], stepSettings[x], 1, 0)
		testOneType(x, len(L1Settings))

net = NeuralNetwork()
	
#net.modCanvas(1911, 1052, 10) #STORSKJERM
net.modCanvas(1000, 500, 8) #PC-SKJERM
net.mod(42, 42, 64, 0.1, 1, 1)
net.shuffleWeights()
net.train(20000)
net.netCost(1000)

pprint(net.Output)
pprint(net.goal)