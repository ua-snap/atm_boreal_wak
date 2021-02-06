#!/usr/bin/env python

import os
import gdal
import osr
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter



def write_landcover(self):

	# write barren ground percent cover
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(os.path.join(self.outdir + "/AnnualMaps" , 'LC_' + self.site + "_" + self.period + "_" + str(self.rep) + "_" + str(self.year) + '.tif'),self.lc.shape[1],self.lc.shape[0],1,gdal.GDT_Float32, )
	dataset.SetGeoTransform(self.gt)  
	dataset.SetProjection(self.proj)
	dataset.GetRasterBand(1).WriteArray(self.lc)
#	dataset.FlushCache()  
	dataset = None
	del dataset

	gtif = gdal.Open(os.path.join(self.outdir + "/AnnualMaps" , 'LC_' + self.site + "_" + self.period + "_" + str(self.rep) + "_" + str(self.year) + '.tif'))
	srcband = gtif.GetRasterBand(1)
	stats = srcband.GetStatistics(True, True)
#	print "[ STATS _ lc ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % (stats[0], stats[1], stats[2], stats[3])



def write_driver(self):

	# write barren ground percent cover
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(os.path.join(self.outdir + "/AnnualMaps" , 'DRIVER_' + self.site + "_" + self.period + "_" + str(self.rep) + "_" + str(self.year) + '.tif'),self.lc.shape[1],self.lc.shape[0],1,gdal.GDT_Float32, )
	dataset.SetGeoTransform(self.gt)  
	dataset.SetProjection(self.proj)
	dataset.GetRasterBand(1).WriteArray(self.driv)
#	dataset.FlushCache()  
	dataset = None
	del dataset
	gtif = gdal.Open(os.path.join(self.outdir + "/AnnualMaps" , 'DRIVER_' + self.site + "_" + self.period + "_" + str(self.rep) + "_" + str(self.year) + '.tif'))
	srcband = gtif.GetRasterBand(1)
	stats = srcband.GetStatistics(True, True)
#	print "[ STATS _ driver] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % (stats[0], stats[1], stats[2], stats[3])


def write_lc_time_series(self):

	print "Write time series csv for land cover"
	total_rep = pd.DataFrame()
	for h in range (1 , self.replicates + 1):
		rep = h
		print "Replicate = ", rep
		total_year = pd.DataFrame()
		for i in range (self.start , self.end + 1):
			year = format(i, '04d')
			print "Year = ", year
#			ds = gdal.Open(os.path.join(self.outdir + "/AnnualMaps" ,'LC_' + self.site + "_" + self.period + "_" + str(rep) + "_" + str(year) + '.tif'))
			if i == self.start:
				ds = gdal.Open(os.path.join(self.indir, 'LCinit_' + self.site + '.tif'))
			else:
				ds = gdal.Open(os.path.join(self.outdir + "/AnnualMaps" ,'LC_' + self.site + "_" + self.period + "_" + str(rep) + "_" + str(year) + '.tif'))
			st = ds.ReadAsArray()
			unique, counts = np.unique(st, return_counts=True)
			tmp = np.asarray((unique, counts))
			tmp1 = tmp[:,tmp[0]!=0]
			total = np.vstack((np.asarray([rep] * len(tmp1[1,]),dtype=np.int32), np.asarray([year] * len(tmp1[1,]),dtype=np.int32), tmp1, np.asarray([sum(tmp1[1])] * len(tmp1[1,]),dtype=np.int32))).T
			totalpd = pd.DataFrame(data=total[0:,0:])
			totalpd.columns = ['rep','year','LC_num','count','total_pxl']
			totalpd['LC_num'] = pd.to_numeric(totalpd['LC_num'])
			totalpd['LC_num'] = totalpd.LC_num.astype(int)
#			totalpd['count'] = totalpd.count.astype(int)
#			totalpd['total_pxl'] = totalpd.total_pxl.astype(int)
			totalpd["PCTCOV"] = 100 * totalpd["count"] / totalpd["total_pxl"]

			if totalpd['PCTCOV'].sum() < 99.99 :
				print "warning! the total percent cover across land cover is not 100%, it is ", totalpd['PCTCOV'].sum(), " %"
				print "A land cover type is probably missing in the input land cover table."

			totalpd['LC_num'] = totalpd.LC_num.astype(int)

			totalpd = pd.merge(totalpd, self.lc_code, on=['LC_num'])
			total_year = pd.concat([total_year, totalpd])
		total_rep = pd.concat([total_rep,total_year])

	total_rep.to_csv(self.outdir + "/TimeSeries/" + 'TS_LC_' + self.site + "_" + self.period + '.csv', sep=',',index=False)


 	print "Plot time series for land cover"
 	df = pd.read_csv(self.outdir + "/TimeSeries/" + 'TS_LC_' + self.site + "_" + self.period + '.csv')
	landform = np.unique(df['type'])
 	rep_avg = pd.DataFrame({'PC_mean' : df.groupby( ['type','year', 'LC_num','descr'])['PCTCOV'].mean()}).reset_index()
 	rep_min = pd.DataFrame({'PC_min' : df.groupby( ['type','year', 'LC_num','descr'])['PCTCOV'].min()}).reset_index()
 	rep_max = pd.DataFrame({'PC_max' : df.groupby( ['type','year', 'LC_num','descr'])['PCTCOV'].max()}).reset_index()
 	result = pd.merge(rep_avg, rep_min, on=['type','year', 'LC_num','descr'])
 	result = pd.merge(result, rep_max, on=['type','year', 'LC_num','descr'])
 	result = pd.merge(result, result[(result['year'] == 1999)], on=['type','LC_num','descr'])
 	result["Rel_pct_cover_mean"] = result["PC_mean_x"] - result["PC_mean_y"]
 	result["Rel_pct_cover_min"] = result["PC_min_x"] - result["PC_min_y"]
 	result["Rel_pct_cover_max"] = result["PC_max_x"] - result["PC_max_y"]
	result = result.drop(columns=["year_y", "PC_mean_x","PC_min_x","PC_max_x","PC_mean_y","PC_min_y","PC_max_y"])
 	result = result[(result['LC_num'] != 22)]

 	fig, ax = plt.subplots(figsize=(8,6))
 	for ldform, data in result.groupby('type'):
	 	fig, ax = plt.subplots(figsize=(8,6))
	 	ax.set_title('Change in percent area (%)')
	 	ax.set_xlabel('Time (year)')
	 	ax.set_ylabel('Relative change in percent cover')
	 	ax.grid()
	 	for label, df in data.groupby('descr'):
	 		ax.plot(df['year_x'], df['Rel_pct_cover_mean'], lw=2, label=label)
	 		ax.fill_between(df['year_x'], df['Rel_pct_cover_min'], df['Rel_pct_cover_max'], alpha=0.5)

	 	plt.legend(loc='upper left',ncol=2)
		#plt.show()
		fig.savefig(self.outdir + "/TimeSeries/" + 'TS_LC_' + self.site + "_" + self.period + '_plot.pdf', bbox_inches='tight')




def write_driv_time_series(self):

	print "Write time series csv for drivers"
	total_rep = pd.DataFrame()
	for h in range (1 , self.replicates + 1):
		rep = h
		print "Replicate = ", rep
		total_year = pd.DataFrame()
		for i in range (self.start , self.end + 1):
			year = format(i, '04d')
			print "Year = ", year
			ds = gdal.Open(os.path.join(self.outdir + "/AnnualMaps" ,'DRIVER_' + self.site + "_" + self.period + "_" + str(rep) + "_" + str(year) + '.tif'))
			st = ds.ReadAsArray()
			unique, counts = np.unique(st, return_counts=True)
			tmp = np.asarray((unique, counts))
			tmp1 = tmp[:,tmp[0]!=0]
			total = np.vstack((np.asarray([rep] * len(tmp1[1,]),dtype=np.int32), np.asarray([year] * len(tmp1[1,]),dtype=np.int32), tmp1, np.asarray([sum(tmp1[1])] * len(tmp1[1,]),dtype=np.int32))).T
			totalpd = pd.DataFrame(data=total[0:,0:])
			totalpd.columns = ['rep','year','DRIVER_num','count','total_pxl']
			totalpd['DRIVER_num'] = pd.to_numeric(totalpd['DRIVER_num'])
			totalpd['DRIVER_num'] = totalpd.DRIVER_num.astype(int)
			totalpd["PCTCOV"] = 100 * totalpd["count"] / totalpd["total_pxl"]

			if totalpd['PCTCOV'].sum() < 99.99 :
				print "warning! the total percent cover across land cover is not 100%, it is ", totalpd['PCTCOV'].sum(), " %"
				print "A driver type is probably missing in the input land cover table."

			totalpd['DRIVER_num'] = totalpd.DRIVER_num.astype(int)
			totalpd = pd.merge(totalpd, self.driv_code, on=['DRIVER_num'])
			total_year = pd.concat([total_year, totalpd])
		total_rep = pd.concat([total_rep,total_year])
	total_rep.to_csv(self.outdir + "/TimeSeries/" + 'TS_DRIVER_' + self.site + "_" + self.period + '.csv', sep=',',index=False)

	# For some very frustrating reason, cumulative sums wouldn't compute unless I write - read - compute - write...
	total_rep = pd.read_csv(self.outdir + "/TimeSeries/" + 'TS_DRIVER_' + self.site + "_" + self.period + '.csv')
	total_rep = (total_rep.assign(PCTCOVcum=total_rep.sort_values(['year']).groupby(['rep','DRIVER_num'])['PCTCOV'].cumsum())).reset_index()
	total_rep.to_csv(self.outdir + "/TimeSeries/" + 'TS_DRIVER_' + self.site + "_" + self.period + '.csv', sep=',',index=False)

 	print "Plot time series for driver"
 	df = pd.read_csv(self.outdir + "/TimeSeries/" + 'TS_DRIVER_' + self.site + "_" + self.period + '.csv')
 	rep_avg = pd.DataFrame({'PCTCOVcum_mean' : df.groupby( ['year', 'DRIVER_num','descr'])['PCTCOVcum'].mean()}).reset_index()
 	rep_min = pd.DataFrame({'PCTCOVcum_min' : df.groupby( ['year', 'DRIVER_num','descr'])['PCTCOVcum'].min()}).reset_index()
 	rep_max = pd.DataFrame({'PCTCOVcum_max' : df.groupby( ['year', 'DRIVER_num','descr'])['PCTCOVcum'].max()}).reset_index()
 	result = pd.merge(rep_avg, rep_min, on=['year', 'DRIVER_num','descr'])
 	result = pd.merge(result, rep_max, on=['year', 'DRIVER_num','descr'])
 	result = result[(result['DRIVER_num'] != 22)]

 	fig, ax = plt.subplots(figsize=(8,6))
 	ax.set_title('Driver of land cover change')
 	ax.set_xlabel('Time (year)')
 	ax.set_ylabel('Cumulative percent cover (%)')
 	ax.grid()
 	for label, df in result.groupby('descr'):
 		ax.plot(df['year'], df['PCTCOVcum_mean'], lw=2, label=label)
 		ax.fill_between(df['year'], df['PCTCOVcum_min'], df['PCTCOVcum_max'], alpha=0.5)

 	plt.legend(loc='upper left',ncol=2)
# #	plt.show()
 	fig.savefig(self.outdir + "/TimeSeries/" + 'TS_DRIVER_' + self.site + "_" + self.period + '_plot.pdf', bbox_inches='tight')



def write_driv_synth_map(self):

	for drv in self.drivers:
		print "driver is ", drv
		mean_rep = 0
		for h in range (1 , self.replicates + 1):
			rep = h
			mean_year = 0
			for i in range (self.start , self.end + 1):
				year = format(i, '04d')
				ds = gdal.Open(os.path.join(self.outdir + "/AnnualMaps" , 'DRIVER_' + self.site + "_" + self.period + "_" + str(rep) + "_" + str(year) + '.tif'))
				st = ds.ReadAsArray()
				st[st != drv] = 0
				st[st == drv] = 1
				mean_year += st
			mean_year = mean_year / (self.end-self.start)
			mean_rep += mean_year
		mean_rep = mean_rep / (self.replicates)
		driver = gdal.GetDriverByName('GTiff')
		dataset = driver.Create(os.path.join(self.outdir + "/VulnMap" , 'DRIVER_CHANGE_' + self.site + "_" + self.period + "_" + str(drv) + '.tif'),self.lc.shape[1],self.lc.shape[0],1,gdal.GDT_Float32, )
		dataset.SetGeoTransform(self.gt)  
		dataset.SetProjection(self.proj)
		dataset.GetRasterBand(1).WriteArray(mean_rep)
		dataset = None
		del dataset








