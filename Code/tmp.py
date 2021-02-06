

import os
import gdal
import osr
import numpy as np
import matplotlib.pyplot as plt
import sys
from bisect import bisect
from random import random 
import csv
import pandas as pd


ecotype = "70633"
ecotype = "300"
import csv
rows = [] 

with open('/Helene/SERDP_Douglas/TorreData/Torre_prob_mn.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	fields = readCSV.next()
	for row in readCSV: 
		rows.append(row) 

P = []
tmp = 1
for row in rows:
	if row[0] == ecotype :
		P = row[1:len(fields)]
		P = [float(i) for i in P]
#		P.append(max(0,1-sum(P)))
		print(P)
		tmp = 0

cdf = [P[0]]
for i in xrange(1, len(P)):
	cdf.append(cdf[-1] + P[i])

prob = bisect(cdf,random())+1
prob

new_LC = int(fields[prob].split("_")[1])
new_change = int(fields[prob].split("_")[2])

if tmp == 1:
	new_LC = int(ecotype)
	new_change = 22

tmp
new_LC
new_change






lc_code = pd.read_csv("/Helene/SERDP_Douglas/Model/input/Ecotype_code.csv")
lc_code.columns=['type','abbrev','descr','LC_num']
lc_code['LC_num'] = lc_code.LC_num.astype(int)

ds = gdal.Open(os.path.join("/Helene/SERDP_Douglas/Model/output/AnnualMaps/LC_TF_projection_1_2000.tif"))
st = ds.ReadAsArray()
unique, counts = np.unique(st, return_counts=True)
tmp = np.asarray((unique, counts))
tmp1 = tmp[:,tmp[0]!=0]
total = np.vstack((np.asarray([1] * len(tmp1[1,]),dtype=np.int32), np.asarray([1999] * len(tmp1[1,]),dtype=np.int32), tmp1, np.asarray([sum(tmp1[1])] * len(tmp1[1,]),dtype=np.int32))).T
totalpd = pd.DataFrame(data=total[0:,0:])
totalpd.columns=['rep','year','LC_num','count','total_pxl']
totalpd['rep'] = totalpd.rep.astype(int)
totalpd['year'] = totalpd.year.astype(int)
totalpd['LC_num'] = totalpd.LC_num.astype(int)
totalpd["PCTCOV"] = 100 * totalpd["count"] / totalpd["total_pxl"]

if totalpd['PCTCOV'].sum() < 99.99 :
	print "warning! the total percent cover across land cover is not 100%, it is ", totalpd['PCTCOV'].sum(), " %"

totalpd['LC_num'] = pd.to_numeric(totalpd['LC_num'])
totalpd['LC_num'] = totalpd.LC_num.astype(int)

totalpd = pd.merge(totalpd, lc_code, on=['LC_num'],how='left')
totalpd.to_csv('/Helene/SERDP_Douglas/Model/TS.csv', sep=',',index=False)





for drv in self.drivers:
 	mean_rep = 0
	for h in range (1 , self.replicates + 1):
 		rep = h
 		mean_year = 0
		for i in range (self.start , self.end + 1):
 			year = format(i, '04d')
			ds = gdal.Open(os.path.join("/Helene/SERDP_Douglas/Model/output/AnnualMaps/DRIVER_TF_projection_1_" + year + ".tif"))
			st = ds.ReadAsArray()
			st[st != drv] = 0
			st[st == drv] = 1
			mean_year += st
		mean_year = mean_year / (self.end-self.start)
		mean_rep += mean_year
	mean_rep = mean_rep / (self.replicates)
	driver = gdal.GetDriverByName('GTiff')
	dataset = driver.Create(os.path.join("/Helene/SERDP_Douglas/Model/output/DRV_CHANGE_" + drv + ".tif"),self.lc.shape[1],self.lc.shape[0],1,gdal.GDT_Float32, )
	dataset.SetGeoTransform(self.gt)  
	dataset.SetProjection(self.proj)
	dataset.GetRasterBand(1).WriteArray(st)
	dataset = None
	del dataset







import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

plotly.tools.set_credentials_file(username='hgenet', api_key='MrsCnQB2iM1qzvOOxIEb')
totalpd = pd.read_csv('/Helene/SERDP_Douglas/Model/output/TimeSeries/TS_DRIVER_TF_projection.csv')
totalpd = (totalpd.assign(PCTCOVcum=totalpd.sort_values(['year']).groupby(['rep','DRIVER_num'])['PCTCOV'].cumsum()))









rows = [] 
with open() as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	fields = readCSV.next()
	for row in readCSV: 
		rows.append(row) 




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF

plotly.tools.set_credentials_file(username='hgenet', api_key='MrsCnQB2iM1qzvOOxIEb')

df = pd.read_csv('/Helene/SERDP_Douglas/Model/output/TimeSeries/TS_DRIVER_TF_projection.csv')
rep_avg = pd.DataFrame({'PC_mean' : df.groupby( ['year', 'DRIVER_num','descr'])['PCTCOV'].mean()}).reset_index()
rep_min = pd.DataFrame({'PC_min' : df.groupby( ['year', 'DRIVER_num','descr'])['PCTCOV'].min()}).reset_index()
rep_max = pd.DataFrame({'PC_max' : df.groupby( ['year', 'DRIVER_num','descr'])['PCTCOV'].max()}).reset_index()
result = pd.merge(rep_avg, rep_min, on=['year', 'DRIVER_num','descr'])
result = pd.merge(result, rep_max, on=['year', 'DRIVER_num','descr'])
result = pd.merge(result, result[(result['year'] == 1999)], on=['DRIVER_num','descr'])
result["Rel_pct_cover_mean"] = result["PC_mean_x"] - result["PC_mean_y"]
result["Rel_pct_cover_min"] = result["PC_min_x"] - result["PC_min_y"]
result["Rel_pct_cover_max"] = result["PC_max_x"] - result["PC_max_y"]
tmp = pd.DataFrame({'Cum_avg' : result.groupby( ['DRIVER_num','year_x'])['PC_mean_x'].sum().groupby(level=[0]).cumsum()}).reset_index()
result = pd.merge(result, tmp, on=['DRIVER_num','year_x'])
result = result.drop(columns=["year_y"])
result = result[(result['DRIVER_num'] != 22)]

fig, ax = plt.subplots(figsize=(8,6))
#ax.set_title('Change in percent area (%)')
ax.set_xlabel('Time (year)')
ax.set_ylabel('Cumulated percent cover (%)')
ax.grid()
for label, df in result.groupby('descr'):
	ax.plot(df['year_x'], df['Cum_avg'], lw=2, label=label)
#	ax.fill_between(df['year_x'], df['Rel_pct_cover_min'], df['Rel_pct_cover_max'], alpha=0.5)
#	ax.plot(df['year_x'], df['PC_mean_x'], lw=2, label=label)
#	ax.fill_between(df['year_x'], df['PC_min_x'], df['PC_max_x'], alpha=0.5)

plt.legend(loc='upper left',ncol=2)
plt.show()
fig.savefig("/Helene/SERDP_Douglas/Model/output/TimeSeries/foo.pdf", bbox_inches='tight')






df = pd.read_csv('/Helene/SERDP_Douglas/Model/output/TimeSeries/TS_LC_TF_projection.csv')
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
ax.set_title('Change in percent area (%)')
ax.set_xlabel('Time (year)')
ax.set_ylabel('Relative change in percent cover')
ax.grid()
for label, df in result.groupby('LC_num'):
	ax.plot(df['year_x'], df['Rel_pct_cover_mean'], lw=2, label=label)
	ax.fill_between(df['year_x'], df['Rel_pct_cover_min'], df['Rel_pct_cover_max'], alpha=0.5)

plt.legend(loc='upper left',ncol=2)
plt.show()
fig.savefig("/Helene/SERDP_Douglas/Model/output/TimeSeries/foo.pdf", bbox_inches='tight')









import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('/Helene/SERDP_Douglas/Model/output/TimeSeries/TS_LC_TF_projection.csv')
rep_avg = pd.DataFrame({'PC_mean' : df.groupby( ['type','year', 'LC_num','descr'])['PCTCOV'].mean()}).reset_index()
data = rep_avg[(rep_avg['type'] == 'Gravelly lowland')]
data = data[(data['year'] == 2000)]

start=1999
end=2100

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))
kw = dict(arrowprops=dict(arrowstyle="-"),zorder=0, va="center")
wedges, texts = ax.pie(data['PC_mean'], startangle=-40)

data['PC_mean'] = pd.to_numeric(data['PC_mean'])

for i, p in enumerate(wedges):
	data['PC_mean'] = pd.to_numeric(data['PC_mean'])
	ang = (p.theta2 - p.theta1)/2. + p.theta1
	y = np.sin(np.deg2rad(ang))
	x = np.cos(np.deg2rad(ang))
	horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
	connectionstyle = "angle,angleA=0,angleB={}".format(ang)
	kw["arrowprops"].update({"connectionstyle": connectionstyle})
	ax.annotate(("%s(%.2f %%)" % (data.iloc[i]['descr'], data.iloc[i]['PC_mean'])), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)

ax.set_title("Year %s" % (i))

plt.show()

from matplotlib.gridspec import GridSpec

df = pd.read_csv('/Helene/SERDP_Douglas/Model/output/TimeSeries/TS_LC_TF_projection.csv')
rep_avg = pd.DataFrame({'PC_mean' : df.groupby( ['type','year', 'LC_num','descr'])['PCTCOV'].mean()}).reset_index()
data = rep_avg[(rep_avg['type'] == 'Gravelly lowland')]
data = data[(data['year'] == 2000)]

the_grid = GridSpec(2, 3)
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(aspect="equal"))
fig.subplots_adjust(hspace=.4, wspace=.4)

for i in range(0,2):
	for j in range(0,3):
		ax = plt.subplot(the_grid[i,j], aspect=1)
		ax.pie(data['PC_mean'], startangle=90)
		ax.set_title("Year %s" % (2000),fontsize=16,y=1.08,style='italic')
		kw = dict(arrowprops=dict(arrowstyle="-"),zorder=0, va="center")
		wedges, texts = ax.pie(data['PC_mean'], startangle=90)
		for k, p in enumerate(wedges):
			data['PC_mean'] = pd.to_numeric(data['PC_mean'])
			ang = (p.theta2 - p.theta1)/2. + p.theta1
			y = np.sin(np.deg2rad(ang))
			x = np.cos(np.deg2rad(ang))
			horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
			connectionstyle = "angle,angleA=0,angleB={}".format(ang)
			kw["arrowprops"].update({"connectionstyle": connectionstyle})
			ax.annotate(("%.2f %%" % (data.iloc[k]['PC_mean'])), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)
		patches, texts = ax.pie(data['PC_mean'], startangle=90)

fig.suptitle('Gravelly lowland',fontsize=20,fontweight='bold')
fig.legend(patches, data['descr'], loc="lower center", ncol=2,fontsize=16)
plt.show()



import os
import gdal
import osr
import numpy as np
import matplotlib.pyplot as plt
import sys
from bisect import bisect
from random import random 
import csv
import pandas as pd
=from matplotlib.gridspec import GridSpec

df = pd.read_csv('/Helene/SERDP_Douglas/Model/output/TimeSeries/TS_LC_TF_projection.csv')
rep_avg = pd.DataFrame({'PC_mean' : df.groupby( ['type','year', 'LC_num','descr'])['PCTCOV'].mean()}).reset_index()

start = 1999
end = 2030

for i in np.unique(rep_avg['type']):
	the_grid = GridSpec(2, int(round((((end-start)/10)/2.0),0))+1)
	fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(aspect="equal"))
	fig.subplots_adjust(hspace=.4, wspace=.4)
	for j in range(0,2):
		for k in range(0,int(round((((end-start)/10)/2.0),0))+1):
			decade = 10*(j+k)+int(round(start+4,-1))
			data = rep_avg[(rep_avg['type'] == i) & (rep_avg['year'] == decade)]
			if data['PC_mean'].sum() < 1:
				values = [v*100 for v in data['PC_mean']]
			else: 
				values = data['PC_mean']
			ax = plt.subplot(the_grid[j,k], aspect=1)
			plt.pie(values, startangle=90)
			ax.set_title("Year %s" % (decade),fontsize=16,y=1.08,style='italic')
			kw = dict(arrowprops=dict(arrowstyle="-"),zorder=0, va="center")
			wedges, texts = ax.pie(values, startangle=90)
			for l, p in enumerate(wedges):
				values = pd.to_numeric(values)
				ang = (p.theta2 - p.theta1)/2. + p.theta1
				y = np.sin(np.deg2rad(ang))
				x = np.cos(np.deg2rad(ang))
				horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
				connectionstyle = "angle,angleA=0,angleB={}".format(ang)
				kw["arrowprops"].update({"connectionstyle": connectionstyle})
				ax.annotate(("%.4f %%" % (data.iloc[l]['PC_mean'])), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)
			patches, texts = ax.pie(values, startangle=90)
	fig.suptitle(i,fontsize=20,fontweight='bold')
	fig.legend(patches, data['descr'], loc="lower center", ncol=2,fontsize=16)
	fig.savefig("/Helene/SERDP_Douglas/Model/output/TimeSeries/" + i + ".pdf", bbox_inches='tight')
	plt.show()







data = data[(data['year'] == 2000)]
start = 1999
end = 2030

the_grid = GridSpec(2, 3)
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw=dict(aspect="equal"))
fig.subplots_adjust(hspace=.4, wspace=.4)

for i in range(0,2):
	for j in range(0,3):
		ax = plt.subplot(the_grid[i,j], aspect=1)
		ax.pie(data['PC_mean'], startangle=90)
		ax.set_title("Year %s" % (2000),fontsize=16,y=1.08,style='italic')
		kw = dict(arrowprops=dict(arrowstyle="-"),zorder=0, va="center")
		wedges, texts = ax.pie(data['PC_mean'], startangle=90)
		for k, p in enumerate(wedges):
			data['PC_mean'] = pd.to_numeric(data['PC_mean'])
			ang = (p.theta2 - p.theta1)/2. + p.theta1
			y = np.sin(np.deg2rad(ang))
			x = np.cos(np.deg2rad(ang))
			horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
			connectionstyle = "angle,angleA=0,angleB={}".format(ang)
			kw["arrowprops"].update({"connectionstyle": connectionstyle})
			ax.annotate(("%.2f %%" % (data.iloc[k]['PC_mean'])), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)
		patches, texts = ax.pie(data['PC_mean'], startangle=90)

fig.suptitle('Gravelly lowland',fontsize=20,fontweight='bold')
fig.legend(patches, data['descr'], loc="lower center", ncol=2,fontsize=16)
plt.show()







df = pd.read_csv('/Helene/SERDP_Douglas/Model/output/TimeSeries/TS_LC_TF_projection.csv')
rep_avg = pd.DataFrame({'PC_mean' : df.groupby( ['type','year', 'LC_num','descr'])['PCTCOV'].mean()}).reset_index()
data = rep_avg[(rep_avg['type'] == 'Gravelly lowland')]

start = 1999
end = 2030

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(1,((end-start)/10)+2):
	decade = int(round(start+4,-1)) + 10 * (i -1)
	print decade
	data['year'] = pd.to_numeric(data['year'])
	tmp = data[(data['year'] == decade)]
	print tmp

	ax = fig.add_subplot(int(round((((end-start)/10)+1)/2,0)+1), 2, i)

	wedges, texts = ax.pie(tmp['PC_mean'], startangle=-40)
	kw = dict(arrowprops=dict(arrowstyle="-"),zorder=0, va="center")

	for j, p in enumerate(wedges):
		tmp['PC_mean'] = pd.to_numeric(tmp['PC_mean'])
		ang = (p.theta2 - p.theta1)/2. + p.theta1
		y = np.sin(np.deg2rad(ang))
		x = np.cos(np.deg2rad(ang))
		horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
		connectionstyle = "angle,angleA=0,angleB={}".format(ang)
		kw["arrowprops"].update({"connectionstyle": connectionstyle})
		ax.annotate(("%s(%.2f %%)" % (tmp.iloc[j]['descr'], tmp.iloc[j]['PC_mean'])), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)
	ax.set_title("Year %s" % (decade))

plt.show()



fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
	j=i+10
	ax = fig.add_subplot(2, 3, i)
	ax.text(0.5, 0.5, str((3, 2, j)),fontsize=18, ha='center')
	ax.set_title("Year %s" % (j))

plt.show()







df = pd.read_csv('/Helene/SERDP_Douglas/Model/output/TimeSeries/TS_LC_TF_projection.csv')
rep_avg = pd.DataFrame({'PC_mean' : df.groupby( ['type','year', 'LC_num','descr'])['PCTCOV'].mean()}).reset_index()
data = rep_avg[(rep_avg['type'] == 'Gravelly lowland')]
data = data[(data['year'] == 2001)]

start=1999
end=2100

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))
kw = dict(arrowprops=dict(arrowstyle="-"),zorder=0, va="center")
wedges, texts = ax.pie(data['PC_mean'], startangle=-40)

data['PC_mean'] = pd.to_numeric(data['PC_mean'])

for i, p in enumerate(wedges):
	data['PC_mean'] = pd.to_numeric(data['PC_mean'])
	ang = (p.theta2 - p.theta1)/2. + p.theta1
	y = np.sin(np.deg2rad(ang))
	x = np.cos(np.deg2rad(ang))
	horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
	connectionstyle = "angle,angleA=0,angleB={}".format(ang)
	kw["arrowprops"].update({"connectionstyle": connectionstyle})
	ax.annotate(("%s(%.2f %%)" % (data.iloc[i]['descr'], data.iloc[i]['PC_mean'])), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)
ax.set_title("Year %s" % (i))

plt.show()










rows = [] 
with open('/Helene/SERDP_Douglas/TorreData/Torre_prob_mn.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	fields = readCSV.next()
	for row in readCSV: 
		rows.append(row) 

for row in rows:
 	if int(row[0]) == self.lc[self.j,self.k] :
 		P = row[1:len(fields)]

