from __future__ import print_function
#matte shiet
import numpy
import random

#visualisering
import sys
import Tkinter, tkMessageBox #til canvas
import os
from pprint import pprint

currentPic = 0
def loadPics(fromPic, nPic):
	print("Henter flere bilder..")
	pics 		= [[0 for x in range(784)] for x in range(nPic)]
	picTypes 	= [0 for x in range(nPic)]
	with open("..\database_fra_nettet\\train-images.idx3-ubyte", "rb") as fI, open("..\database_fra_nettet\\train-labels.idx1-ubyte", "rb") as fL:
		fI.seek(16 + (784 * fromPic))
		fL.seek(8 + fromPic)
		for i in range(nPic):
			picTypes[i] = int(fL.read(1).encode('hex'), 16)
			for x in range(784):
				pics[i][x] = int(fI.read(1).encode('hex'), 16)
	return pics, picTypes
	
def loadNextPic(PICS_LOAD):
	global currentPic, picTypes, pics
	if currentPic % PICS_LOAD == 0:
		pics, picTypes = loadPics(currentPic, PICS_LOAD)

	pic = pics[currentPic % PICS_LOAD]
	pType = picTypes[currentPic % PICS_LOAD]
	currentPic += 1
	return 	pic, pType

def printB(bilde):
	CANVS = 500
	top = Tkinter.Tk()
	w = Tkinter.Canvas(top, bg="white", height=CANVS, width=CANVS)

	for y in range (0, 28):
		for x in range(0, 28):
			farge = "#" + '{:02x}'.format(bilde[(y * 28) + x]) + '{:02x}'.format(bilde[(y * 28) + x]) + '{:02x}'.format(bilde[(y * 28) + x])
			w.create_rectangle(x*(CANVS/28), y*(CANVS/28), (x + 1)*(CANVS/28), (y + 1)*(CANVS/28), width = 0, fill=farge)
	w.pack()
	top.mainloop()

for m in range(1000):
	pic, n = loadNextPic(5)

	print("\n\nBilde: " + str(n))
	printB(pic)
	for x in range(28):
		for y in range(28):
			print(pic[(x * 28) + y], end="")
		print("")
	e = raw_input("")