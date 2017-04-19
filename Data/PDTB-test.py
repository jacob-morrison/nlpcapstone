import os
import sys
from collections import Counter

def main(path):
	#training = open('trainingPDTB.txt', 'w')
	#explicitTraining = open('explicitTrainPDTB.txt', 'w')
	implicitTraining = open('train.txt', 'w')

	#development = open('developmentPDTB.txt', 'w')
	#explicitDev = open('devExplicitPDTB.txt', 'w')
	implicitDev = open('dev.txt', 'w')

	#test = open('testPDTB.txt', 'w')
	#explicitTest = open('testExplicitPDTB.txt', 'w')
	implicitTest = open('test.txt', 'w')
	types = Counter()
	for folder in os.listdir(path):
		#training
		if folder[0] != '.' and folder != '00' and folder != '01' and folder != '22' and folder != '23' and folder != '24':
			for filename in os.listdir(path + "/" + folder):
				with open(path + "/" + folder + "/" + filename) as f:
					for line in f:
						tokens = line.split('|')
						types[tokens[11]] += 1
						#print tokens
						#if tokens[0] == 'Explicit':
						#
						if tokens[0] == 'Implicit':
							#print(tokens[11])
							implicitTraining.write(line)
						#for line in f:
						#	training.write(line)
		#dev
		elif folder == '22':
			print("done with training")
			for filename in os.listdir(path + "/" + folder):
				with open(path + "/" + folder + "/" + filename) as f:
					for line in f:
						tokens = line.split('|')
						if tokens[0] == 'Implicit':
							implicitDev.write(line)
		#test
		elif folder == '23':
			print("done with development")
			for filename in os.listdir(path + "/" + folder):
				with open(path + "/" + folder + "/" + filename) as f:
					for line in f:
						tokens = line.split('|')
						if tokens[0] == 'Implicit':
							implicitTest.write(line)
	print(len(types))
	for t in types:
		print(t + " " + str(types[t]))
	print("done with everything")


if __name__ == '__main__':
	main(sys.argv[1])