#!/usr/bin/env python

import os
import gdal
import osr
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from bisect import bisect
from random import random 

import read_input
import write_output


class Boreal(object):

	def __init__(self):

		self.config_file = sys.argv[1]
		self.run_atm()

	def run_atm(self):

		read_input.read_config(self)
#		print "The period simulation is ", self.period

		if self.period == "projection":
			self.start = self.start_sc
			self.end = self.end_sc
		elif self.period == "historical":
			self.start = self.start_hist
			self.end = self.end_hist

#		print "The simulation is conducted from ", self.start , ' to ' , self.end
#		print "Outputs will be stored in ", self.outdir
#		print "Input data are read from ", self.indir

		read_input.read_init_landcover(self)
		read_input.read_parms(self)



 		#### REPLICATION LOOP
 		for h in range (1 , self.replicates + 1):
 			self.rep = h
 			print "Replicate #", self.rep

 			#### TIME LOOP
 			for i in range (self.start , self.end + 1):
  				i1 = i-1
  				self.year = format(i, '04d')
 # 				self.year1 = format(i1, '04d')
 # 				print "The year is ", self.year
 # 				# read land cover data
 # 				if i == self.start:
 # 					read_input.read_init_landcover(self)
 # 				else:
 # 					read_input.read_landcover(self)
 # 				self.driv = np.zeros(shape=(self.lc.shape[0],self.lc.shape[1]))

 # 				#### SPATIAL LOOP
 # 				for self.j in range(0, self.lc.shape[0]) :
 # 					for self.k in range(0, self.lc.shape[1]) :
 # #				for self.j in range(1000, 1010) :
 # #					for self.k in range(1000, 1010) :
 # 						#print "For raw ",self.j," and column ",self.k," the mask is ",self.mask[self.j,self.k]
 # #						print "j= ",self.j, " k= ",self.k," lc = ",int(self.lc[self.j,self.k])
 # 						if int(self.lc[self.j,self.k]) > 0:
 # 							P = []
 # 							tmp = 1
 # 							for row in self.trans:
 # 								if int(row[0]) == self.lc[self.j,self.k] :
 # 									P = row[1:len(fields)]
 # 									P = [float(i) for i in P]
 # 								#	P.append(max(0,1-sum(P)))
 # #									print(P)
 # 									tmp = 0

 # 									cdf = [P[0]]
 # 									for i in xrange(1, len(P)):
 # 										cdf.append(cdf[-1] + P[i])
 # 									prob = bisect(cdf,random())+1
 # 									lc_new = int(fields[prob].split("_")[1])
 # 									driv_new = int(fields[prob].split("_")[2])
 # #									print "prob = ",prob," lc_new = ",lc_new," driv_new = ",driv_new

 # 							if tmp == 1:
 # 								lc_new = self.lc[self.j,self.k]
 # 								driv_new = 22

 # #							print "tmp = ",tmp
 # #							print "lc_new = ",lc_new
 # #							print "driv_new = ",driv_new

 # 							self.lc[self.j,self.k]=lc_new
 # 							self.driv[self.j,self.k]=int(driv_new)

 # #							print "self.lc[self.j,self.k] = ", int(self.lc[self.j,self.k])
 # #							print "self.driv[self.j,self.k] = ", int(self.driv[self.j,self.k])



 # 				#### OUTPUT FILES
 # #				print np.amax(self.driv)

 # 				# write the annual land cover and driver maps
 # 				directory = self.outdir + "/AnnualMaps"
 # 				if not os.path.exists(directory):
 # 					os.makedirs(directory)
 # 				write_output.write_landcover(self)
 # 				write_output.write_driver(self)


		# write the annual land cover and driver maps
		directory = self.outdir + "/TimeSeries"
		if not os.path.exists(directory):
			os.makedirs(directory)
		write_output.write_lc_time_series(self)
#		write_output.write_driv_time_series(self)

		# directory = self.outdir + "/VulnMap"
		# if not os.path.exists(directory):
		# 	os.makedirs(directory)
		# write_output.write_driv_synth_map(self)

Boreal()




