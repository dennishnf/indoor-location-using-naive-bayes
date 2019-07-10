
# Indoor Loacation using Bayes
# code for offline processing
# Spyder, Python 3.6 (Ancaconda2 version)
# By: Dennis Nunez Fernandez


#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.pyplot import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import os
import glob



#%%

plt.close('all')


DATA = "DATA_001_House"

path_data  = "data1_filtered_wifis/"+DATA+"/"

path_part01 = "data2_divided_parts/"+DATA+"/part01/"
path_part02 = "data2_divided_parts/"+DATA+"/part02/"
path_part03 = "data2_divided_parts/"+DATA+"/part03/"
path_part04 = "data2_divided_parts/"+DATA+"/part04/"
path_part05 = "data2_divided_parts/"+DATA+"/part05/"
path_part06 = "data2_divided_parts/"+DATA+"/part06/"
path_part07 = "data2_divided_parts/"+DATA+"/part07/"
path_part08 = "data2_divided_parts/"+DATA+"/part08/"
path_part09 = "data2_divided_parts/"+DATA+"/part09/"
path_part10 = "data2_divided_parts/"+DATA+"/part10/"


numRegions = 0
for fullname in glob.glob(path_data+"*.csv"):
    dfInRegion = pd.read_csv(fullname, sep=" ")
    numAPs = len(dfInRegion.columns)
    numRegions = numRegions+1


Regions = ["R%.3d" % i for i in range(1,numRegions+1)]
APs = ["W%.3d" % i for i in range(1,numAPs+1)]


#%%


# DIVISION OF THE RSSI DATA INTO TRAINING AND TESTING SETS


# Read the .csv files, should be saved as R001.csv, R002.csv, R003.csv, R004.csv, R005.csv, ...


D = {}

for r in Regions:
    name = r+".csv"
    D[r+"_"] = pd.read_csv(path_data+name, sep=" ")
    D[r] = D[r+"_"][APs]
    print("reading: "+path_data+name)
  

# Shuffle

for r in Regions:
    D[r] = D[r].sample(frac=1)
    print("Shuffling "+r)

    
# Divide in *_test and *_train datasets

Dtt = {}

for r in Regions:
    print("Dividing "+r)
    num_interval = int(0.1*len(D[r]))
    Dtt[r+"_part01"] = D[r].iloc[0*num_interval:1*num_interval].reset_index(drop=True)
    Dtt[r+"_part02"] = D[r].iloc[1*num_interval:2*num_interval].reset_index(drop=True)
    Dtt[r+"_part03"] = D[r].iloc[2*num_interval:3*num_interval].reset_index(drop=True)
    Dtt[r+"_part04"] = D[r].iloc[3*num_interval:4*num_interval].reset_index(drop=True)
    Dtt[r+"_part05"] = D[r].iloc[4*num_interval:5*num_interval].reset_index(drop=True)
    Dtt[r+"_part06"] = D[r].iloc[0*num_interval:1*num_interval].reset_index(drop=True)
    Dtt[r+"_part07"] = D[r].iloc[1*num_interval:2*num_interval].reset_index(drop=True)
    Dtt[r+"_part08"] = D[r].iloc[2*num_interval:3*num_interval].reset_index(drop=True)
    Dtt[r+"_part09"] = D[r].iloc[3*num_interval:4*num_interval].reset_index(drop=True)
    Dtt[r+"_part10"] = D[r].iloc[4*num_interval:5*num_interval].reset_index(drop=True)


# Save in train/test folders

if (not os.path.exists(path_part01)):
    os.makedirs(path_part01)

if (not os.path.exists(path_part02)):
    os.makedirs(path_part02)

if (not os.path.exists(path_part03)):
    os.makedirs(path_part03)

if (not os.path.exists(path_part04)):
    os.makedirs(path_part04)

if (not os.path.exists(path_part05)):
    os.makedirs(path_part05)
    
if (not os.path.exists(path_part06)):
    os.makedirs(path_part06)

if (not os.path.exists(path_part07)):
    os.makedirs(path_part07)

if (not os.path.exists(path_part08)):
    os.makedirs(path_part08)

if (not os.path.exists(path_part09)):
    os.makedirs(path_part09)

if (not os.path.exists(path_part10)):
    os.makedirs(path_part10)

    
for r in Regions:
    name = r+".csv"
    Dtt[r+"_part01"].to_csv(path_part01+name, sep=' ', index=False)
    print("Saving " + path_part01 + name)
    Dtt[r+"_part02"].to_csv(path_part02+name, sep=' ', index=False)
    print("Saving " + path_part02 + name)
    Dtt[r+"_part03"].to_csv(path_part03+name, sep=' ', index=False)
    print("Saving " + path_part03 + name)
    Dtt[r+"_part04"].to_csv(path_part04+name, sep=' ', index=False)
    print("Saving " + path_part04 + name)
    Dtt[r+"_part05"].to_csv(path_part05+name, sep=' ', index=False)
    print("Saving " + path_part05 + name)
    Dtt[r+"_part06"].to_csv(path_part06+name, sep=' ', index=False)
    print("Saving " + path_part06 + name)
    Dtt[r+"_part07"].to_csv(path_part07+name, sep=' ', index=False)
    print("Saving " + path_part07 + name)
    Dtt[r+"_part08"].to_csv(path_part08+name, sep=' ', index=False)
    print("Saving " + path_part08 + name)
    Dtt[r+"_part09"].to_csv(path_part09+name, sep=' ', index=False)
    print("Saving " + path_part09 + name)
    Dtt[r+"_part10"].to_csv(path_part10+name, sep=' ', index=False)
    print("Saving " + path_part10 + name)


#%%
