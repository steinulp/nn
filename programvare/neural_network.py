import math
import simplejson as json

import Tkinter, tkMessageBox #til canvas, altsaa testing

PICS_LOAD = 10
currentPic = 0


def printB(arr):
	CANVS = 500
	top = Tkinter.Tk()
	w = Tkinter.Canvas(top, bg="white", height=CANVS, width=CANVS)

	for y in range (0, 8):
		for x in range(0, 8):
			farge = "#" + '{:02x}'.format(int(arr[(y * 8)+ x])) + '{:02x}'.format(int(arr[(y * 8)+ x])) + '{:02x}'.format(int(arr[(y * 8)+ x]))
			w.create_rectangle(x*(CANVS/8), y*(CANVS/8), (x + 1)*(CANVS/8), (y + 1)*(CANVS/8), width = 0, fill=farge)
	w.pack()
	top.mainloop()


#Deklarering av vektorer (for ordens skyld)
#"holdere for verdiene"
inp = [0 for x in range(64)] #bildet
layer1 = [0 for x in range(16)] #"skjult lag" del 1
layer1 = [0 for x in range(16)] #"skjult lag" del 1
output = [0 for x in range(8)] #resultatet

#"vekter" mellom nervene
weight1 = [[0 for x in range(64)] for x in range(16)]
weight2 = [[0 for x in range(16)] for x in range(16)]
weight3 = [[0 for x in range(16)] for x in range(8)]

def sigmoid(inp):
	return float((2 / (1 + math.exp(-inp))) - 1)


def loadPics(fromPic, toPic):
	pics = [0 for x in range(1 + toPic - fromPic)]
	with open('..\egne_bilder\data.json') as f:
		for i, line in enumerate(f):
		    if (i >= fromPic) and (i <= toPic):
		        pics[i - fromPic] = line
		    elif i > toPic:
		        break
	return pics

# def getPic(picNumb, arr):
# 	inp = arr[picNumb].split(",")
# 	for x in range(10):
# 		print(inp[x])

def loadNextPic():
	global currentPic
	global inp
	global pics
	if currentPic % PICS_LOAD == 0:
		pics = loadPics(currentPic, currentPic + PICS_LOAD)
	buff = pics[currentPic % PICS_LOAD].split(",")
	for x in range(64):
		inp[x] = buff[x + 2]
	# if int(buff[0]) - 1 != int(currentPic):
	# 	print("ERROR: feil i rekkefolge: hentet tall [" + str(buff[0]) + "] ulik med [" + str(currentPic) + "]")
	# 	return -1
	currentPic += 1
	return buff[0]


try:
	x = raw_input("Antall: ")
	x = int(x)
except ValueError:
	print("ops")

	
for x in range(x):
	print(loadNextPic())
	#printB(inp)