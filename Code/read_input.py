#!/usr/bin/env python

import os
import gdal
import osr
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


def read_config(self):

#	print 'Reading control file: ', self.config_file
	control = {}

	with open(self.config_file, 'r') as f:
		for line in f:
	 		if line.startswith('#'):
	 			continue
	 		else:
	 			(key, val) = line.split()
				control[(key)] = val

	self.site 			= control['Site']
	self.period 		= control['Period']
	self.replicates		= int(control['Replicates'])
	self.parms_file 	= control['Parameter_file']
	self.trans_table 	= control['Transition_table']
	self.driver_table 	= control['Driver_table']
	self.lc_table 		= control['Landcover_table']
	self.indir 			= control['Input_Directory']
	self.outdir 		= control['Output_Directory']
	self.start_sc 		= int(control['Start_sc_year'])
	self.end_sc  		= int(control['End_sc_year'])
	self.start_hist   	= int(control['Start_hist_year'])
	self.end_hist   	= int(control['End_hist_year'])

def read_parms(self):

	self.trans = [] 
	with open(self.trans_table) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		fields = readCSV.next()
		for row in readCSV: 
			self.trans.append(row) 

	self.driv_code = pd.read_csv(self.driver_table)
	self.driv_code.columns=['type','abbrev','descr','DRIVER_num']
	self.driv_code['DRIVER_num'] = self.driv_code.DRIVER_num.astype(int)
	self.driv_code['type'] = self.driv_code.type.astype(str)
	self.driv_code['abbrev'] = self.driv_code.abbrev.astype(str)
	self.driv_code['descr'] = self.driv_code.descr.astype(str)

	self.lc_code = pd.read_csv(self.lc_table)
	self.lc_code.columns=['type','abbrev','descr','LC_num']
	self.lc_code['LC_num'] = self.lc_code.LC_num.astype(int)
	self.lc_code['type'] = self.lc_code.type.astype(str)
	self.lc_code['abbrev'] = self.lc_code.abbrev.astype(str)
	self.lc_code['descr'] = self.lc_code.descr.astype(str)


def read_init_landcover(self):

#	print "Reading initial land cover data ..."
	ds = gdal.Open(os.path.join(self.indir, 'LCinit_' + self.site + '.tif'))
	self.lc = ds.ReadAsArray()
	self.gt = ds.GetGeoTransform() 
	self.proj = ds.GetProjection()
	self.lc.shape


def read_landcover(self):

	ds = gdal.Open(os.path.join(self.outdir + "/AnnualMaps" ,'LC_' + self.site + "_" + self.period + "_" + str(self.rep) + "_" + str(self.year1) + '.tif'))
	self.lc = ds.ReadAsArray()

