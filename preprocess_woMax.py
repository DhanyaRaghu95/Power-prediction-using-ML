import pandas

df = pandas.read_csv('data.csv')
"""
It might be worth trying to take ratios of some of these things - i.e., instead of instructions and cycles (separate), I provided you with IPC. Could do the same thing for branch misprediction (mispredictions/total branches), LLC miss rate (miss/refs), etc. This will reduce the dimensionality by taking two absolute values and representing them as one ratio.
"""

df["LLC Misses"] = df[["LLC Misses (0)","LLC Misses (1)"]]
df["LLC References"] = df[["LLC References (0)","LLC References (1)"]]

bm=[]
for i in range(24):
	bm.append("Branch Mispredictions ("+str(i)+")")
#df["Branch Misprediction"] = df[bm].mean(axis=24)

bi=[]
for i in range(24):
	bi.append("Branch Instructions ("+str(i)+")")
#df["Branch Instructions"] = df[bi].max(axis=1)

ipc=[]
for i in range(24):
	ipc.append("IPC ("+str(i)+")")
df["IPC"] = df[ipc].max(axis=1)


bm.insert(0,"Active Power (W)")
bm.extend(["LLC Misses","LLC References","IPC"])
#headers = ["Active Power (W)","LLC Misses","LLC References","IPC",i for i in bm]#"Branch Misprediction"
headers = bm
df.to_csv('output_woMax.csv', columns = headers)

