from __future__ import print_function #print uten linjeskift, kan fjernes
from random import randint
from shutil import copyfile
import math
import struct
import simplejson as json #til jsonfilen
import Tkinter, tkMessageBox #til canvas


CANVS = 320
testarray = [[0 for x in range(8)] for x in range(8)]

def lagBilde(bType):
	if bType == 0: #full
		for y in range (0, 8):
			for x in range(0, 8):
				testarray[y][x] = randint(0, 32);
	if bType == 1: #tom 
		for y in range (0, 8):
			for x in range(0, 8):
				testarray[y][x] = randint(224, 255)
	if bType == 2: #kryss
		for y in range (0, 8):
			for x in range(0, 8):
				testarray[y][x] = 0
	if bType == 3: #pluss
		rh_ra = randint(-1,1)
		rh_rb = randint(-3, 3)
		rv_ra = randint(-1,1)
		rv_rb = randint(-3, 3)
		rc = randint(0, 20) / 10
		rd = randint(0, 20) / 10
		for y in range (0, 8):
			rv_yfaktor = ((y * rv_ra) + rv_rb) / 10	
			for x in range(8):

				rh_xfaktor = (((x - rd) * rh_ra) + rh_rb) / 10
				rh_yfaktor = ((y - (4 - rc)) + rh_xfaktor)
				rv_xfaktor = ((x - (4 - rd)) + rv_yfaktor)
				delb = 255 - int(max(0, min(255, rv_xfaktor * rv_xfaktor * randint(80, 100))) + randint(0,20))
				if (x - rc > y - rd - 1) and (x - rc < y - rd + 1):
					delb = 0
				dela = 255 - int(max(0, min(255, rh_yfaktor * rh_yfaktor * randint(80, 100))) + randint(0,20))
				testarray[y][x] = 255 - int(max(0, min(255, dela + delb)))

	if bType == 4: #rett, horisontal
		ra = randint(-1,1)
		rb = randint(-3, 3)
		rc = randint(-10, 0) / 10
		for y in range (0, 8):	
			for x in range(0, 8):
				ny = y - rc #for aa unngaa yfaktor = 0
				xfaktor = ((x*ra) + rb) / 10
				yfaktor = ((ny - 4) + xfaktor)
				testarray[y][x] = int(max(0, min(255, yfaktor * yfaktor * randint(20, 255)) - randint(0,10)))
	if bType == 5: #rett, vertikal
		ra = randint(-1,1)
		rb = randint(-3, 3)
		for y in range (0, 8):
			yfaktor = ((y*ra) + rb) / 5	
			for x in range(0, 8):
				nx = x + 0.5
				xfaktor = ((nx - 4) + yfaktor)
				testarray[y][x] = int(max(0, min(255, xfaktor * xfaktor * randint(20, 255)) - randint(0,10)))
	if bType == 6: #skraa, topp venstre
		ra = randint(-100,100)
		for y in range (0, 8):
			for x in range(0, 8):
				skr = (abs(y - x) * ra) / 200
				vektor = (y - 4) - (x - 4) + 0.2 + skr
				testarray[y][x] = int(max(0, min(255, vektor * vektor * randint(10, 255) - randint(0,10))))
	if bType == 7: #skraa, topp hooyre
		for y in range (0, 8):
			for x in range(0, 8):
				testarray[y][x] = 0;

def printB():
	lagBilde(3)
	top = Tkinter.Tk()
	w = Tkinter.Canvas(top, bg="white", height=CANVS, width=CANVS)

	for y in range (0, 8):
		for x in range(0, 8):
			farge = "#" + '{:02x}'.format(testarray[y][x]) + '{:02x}'.format(testarray[y][x]) + '{:02x}'.format(testarray[y][x])
			w.create_rectangle(x*(CANVS/8), y*(CANVS/8), (x + 1)*(CANVS/8), (y + 1)*(CANVS/8), width = 0, fill=farge)
	w.pack()
	top.mainloop()

def skrivTilJSON(n):
	with open('data.json', 'wb') as f:
		for i in range(0, n):
			lagBilde(i % 8)
			v = i % 8

			f.write(str(i) + "," + str(v) + ",")
			for y in range (0, 8):
				for x in range(0, 8):
					f.write(str(testarray[y][x]) + ",")
			f.write("\n")



def lagBildeFil():
	lagBilde(4)
	copyfile("head.txt", "bilde.bmp")
	file = open("bilde.bmp","a")
	for y in range (0, 8):
		for x in range(0, 8):
			print("{0:<4}".format(testarray[y][x]), end=' ')
			file.write(struct.pack('>B', 	testarray[y][x]))
		print('')

	file.write(struct.pack('>h', 0))
	file.close()

n = 0
bool = 1
while bool:
	inp = raw_input("Skrive som bildefil (b), eller til JSON (j): ")
	if inp == "j":
		npics = int(raw_input("Hvor mange bilder?: "))
		skrivTilJSON(npics)
	elif inp == "b":
		lagBildeFil()
	elif inp == "d": #debug, viser bildet i canvas
		printB()
		printB()
		printB()
	else:
		bool = 0
