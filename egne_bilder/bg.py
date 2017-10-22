from __future__ import print_function
from random import randint
from shutil import copyfile
import math


testarray = [[0 for x in range(8)] for x in range(8)]

n = 0
bool = 1
while bool:
	inp = raw_input("Skrive som bildefil (b), eller til JSON (j): ")
	if inp == "j":
		n = int(raw_input("Hvor mange bilder?: "))

	if ((inp != "j") and (inp != "b")) or (type(n) != int):
		print("Noe gikk galt...")
	else:
		bool = 0


		#file.write(testarray[y][x]);
lagBilde()

if inp == "b":
	copyfile(head.bmp, bilde.bmp)
	#file = open("eget.bmp","w")

def lagBilde():
	for y in range (0, 8):
		for x in range(0, 8):
			testarray[y][x]=format(randint(0, 255), '#01');
			print(testarray[y][x])
