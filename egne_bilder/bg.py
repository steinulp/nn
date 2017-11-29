from __future__ import print_function #print uten linjeskift, kan fjernes
from random import randint
from shutil import copyfile
import math
import struct
import Tkinter, tkMessageBox #til canvas
import sys


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
		ra = randint(-100,100)
		rb = randint(-100,100)
		for y in range (0, 8):
			for x in range(0, 8):
				skraaEn = (y - 4) - (x - 4) + 0.2 + (abs(y - x) * ra) / 200
				skraaTo = (y - 4) + (x - 4) + 0.2 + (abs(y - x) * ra) / 200
				delb = 255 - int(max(0, min(255, skraaEn * skraaEn * randint(80, 100))) + randint(0,20))
				dela = 255 - int(max(0, min(255, skraaTo * skraaTo * randint(80, 100))) + randint(0,20))
				testarray[y][x] = 255 - int(max(0, min(255, dela + delb)))
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
				testarray[y][x] = int(max(0, min(255, (xfaktor ** 2) * randint(20, 255)) - randint(0,10)))
	if bType == 6: #skraa, topp venstre
		ra = randint(-100,100)
		for y in range (0, 8):
			for x in range(0, 8):
				skraa = (abs(y - x) * ra) / 200
				vektor = (y - 4) - (x - 4) + 0.2 + skraa
				testarray[y][x] = int(max(0, min(255, (vektor ** 2) * randint(10, 255) - randint(0,10))))
	if bType == 7: #skraaaa, topp hooyre
		ra = randint(-100,100)
		for y in range (0, 8):
			for x in range(0, 8):
				skraa = (abs(y - x) * ra) / 200
				vektor = (y - 4) + (x - 3.5) - 0.1 + skraa
				testarray[y][x] = int(max(0, min(255, (vektor ** 2) * randint(10, 255) - randint(0,10))))

def printB():
	lagBilde(randint(0, 7))
	top = Tkinter.Tk()
	w = Tkinter.Canvas(top, bg="white", height=CANVS, width=CANVS)

	for y in range (0, 8):
		for x in range(0, 8):
			farge = "#" + '{:02x}'.format(testarray[y][x]) + '{:02x}'.format(testarray[y][x]) + '{:02x}'.format(testarray[y][x])
			w.create_rectangle(x*(CANVS/8), y*(CANVS/8), (x + 1)*(CANVS/8), (y + 1)*(CANVS/8), width = 0, fill=farge)
	w.pack()
	top.mainloop()

def skraaivTilFil(n):
	with open('data.json', 'wb') as f:
		for i in range(0, n):
			if(n > 10000 and i % 100 == 0):
				sys.stdout.write('\r')
				sys.stdout.write("[%-19s] %d%%" % ('=' * int((20 * (1+i)) / n), int((100 * (1 + i)) / n)))
				sys.stdout.flush()
			print("\n\n")
			lagBilde(i % 8)
			v = i % 8

			f.write(str(i) + "," + str(v) + ",")
			for y in range (0, 8):
				for x in range(0, 8):
					f.write(str(testarray[y][x]) + ",")
			f.write("\n")



def lagBildeFil():
	lagBilde(7)
	copyfile("head.txt", "bilde.bmp")
	file = open("bilde.bmp","a")
	for y in range (0, 8):
		for x in range(0, 8):
			print("{0:<4}".format(testarray[y][x]), end=' ')
			file.write(struct.pack('>B', testarray[y][x]))
		print('')

	file.write(struct.pack('>h', 0))
	file.close()

n = 0
bool = 1
while bool:
	inp = raw_input("Skrive som bilde (b), eller til fil (f): ")
	if inp == "f":
		npics = int(raw_input("Hvor mange bilder?: "))
		skraaivTilFil(npics)
	elif inp == "b":
		lagBildeFil()
	elif inp == "d": #debug, viser bildet i canvas
		while 1:
			printB()
	else:
		bool = 0
