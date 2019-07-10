
# Indoor Loacation using Bayes
# code for offline processing
# Spyder, Python 3.6 (Ancaconda2 version)
# By: Dennis Nunez Fernandez


#%%

import os
import csv
import numpy as np
import pandas as pd
import glob
import collections


#%%

# Read pre csv files and conver to dataframe 

DATA = "DATA_001_House"

path_Pre = "data0_original/"+DATA+"/"
path_Post = "data1_filtered_wifis/"+DATA+"/"

path_info = "data_info/"+DATA+"/"


#%%

dup_general = []

#twice: first time fill the duplicates in dup_general
for ntimes in [1,2]:
    #read recursively
    for fullname in glob.glob(path_Pre+"*.csv"):
        
        #obtain only the name of the csv
        name_csv = fullname.replace(path_Pre, "")
        
        path_DataPre = path_Pre + name_csv
        path_DataPost = path_Post + name_csv[0:4]+".csv"
        
        print("Processing:")
        print(path_DataPre)
        print(path_DataPost)
        
        #create a dictionary, composed by a dataframe for each tick
        dicRegion = {}
        
        with open(path_DataPre, "r") as f:
            #read csv
            reader = csv.reader(f, delimiter="\t")
            
            #read tick i
            for i, line in enumerate(reader):
                
                #show tick and number of detected APs
                #print "tick", i , "detects", len(line), "APs"
                
                #parameters
                ssid = []
                bssid = []
                rssi = []
                
                #read AP j and their parameters
                for j in range(0,len(line)-1):              
                    #not consider if AP doesn't have name
                    if line[j][0:len(line[j])-22] !="~xx~":
                        #length of string 'ssid bssid rssi'
                        w = len(line[j])
                        #append all parameters
                        ssid.append(line[j][0:w-22])
                        bssid.append(line[j][w-21:w-4])
                        rssi.append(int(line[j][w-3:w-0]))
                
                #convert to dataframe and assign a place in the dictionary
                df_tmp = pd.DataFrame(np.array(rssi).reshape(1,len(rssi)), columns = ssid)  #columns = ssid or bssid
        
        
                #find duplicates in region
                df_dup = [x for x, y in collections.Counter(df_tmp.columns.tolist()).items() if y > 1]
                
                #save duplicates in general
                for dup in df_dup:
                    dup_general.append(dup)
                
                #remove duplicates in region
                df_uniq = df_tmp.columns.tolist()[:]
                for item in df_dup:
                    while df_uniq.count(item) > 0:
                        df_uniq.remove(item)
                #remove duplicates in general
                df_uniq1 = df_uniq[:]
                for item in set(dup_general):
                    while df_uniq1.count(item) > 0:
                        df_uniq1.remove(item)
            
                #convert to dataframe and assign a place in the dictionary
                dicRegion[i] = df_tmp[df_uniq1]
                            
                
        #concatenate dataframes of ticks
        dfRegion = pd.concat(dicRegion, sort=True)
        #reset index
        dfRegion = dfRegion.reset_index(drop=True)
        #fill NaNs with the value -100.0 (no detectable signal)
        dfRegion = dfRegion.fillna(-100.0)
        
        #create folder
        if (not os.path.exists(path_Post)):
            os.makedirs(path_Post)
        
        #save as csv
        dfRegion.to_csv(path_DataPost, sep=' ', index=False)



#%%


# Obtain name of all APs

listAPs = []

#read recursively
for fullname in glob.glob(path_Post+"*.csv"):
    
    #read dataframe of a region 
    dfRegion = pd.read_csv(fullname, sep=" ")
    
    #obtain column names in a list 
    dfRegion = dfRegion.columns.tolist()
    
    #concatenate woth new AP name
    listAPs = dfRegion + list(set(listAPs) - set(dfRegion))
  


# Fill all tables with APs not detected at a given region

#read recursively
for fullname in glob.glob(path_Post+"*.csv"):
    
    #make a copy of listAPs
    listPartial = listAPs[:]
    
    #read dataframe of a region
    dfRegion = pd.read_csv(fullname, sep=" ")
    
    #obtain column names in a list 
    listRegion = dfRegion.columns.tolist()
    
    # remove listRNo from listAPs
    for item in listRegion:
        while listPartial.count(item) > 0:
            listPartial.remove(item)
    
    #create a dataframe with a size
    dfPar = pd.DataFrame(index=range(len(dfRegion)),columns=range(len(listPartial)))
    
    #name columns 
    dfPar.columns = listPartial
    
    #fill with constant value
    dfPar = dfPar.fillna(-100.0)
    
    #conactenate
    dfRegion = pd.concat([dfRegion, dfPar], axis=1)
    
    #order columns
    dfRegion = dfRegion[listAPs]
    
    #save csv
    dfRegion.to_csv(fullname, sep=' ', index=False)





#%%


# Discart signals that appears in less than 25% of # ticks and less than 25% of # regions


#read recursively
number = 0
for fullname in glob.glob(path_Post+"*.csv"):
    number = number +1

df_counter = pd.DataFrame(columns = listAPs, index=["R%.3d" % i for i in range(1,number+1)])

#read recursively
for fullname in glob.glob(path_Post+"*.csv"):
    
    print(fullname)
    
    #read dataframe of a region
    dfInRegion = pd.read_csv(fullname, sep=" ")
    
    for x in range(8,101):
        #print -x
        if -x ==-100:
            dfInRegion = dfInRegion.replace(-x, 0)
        dfInRegion = dfInRegion.replace(-x, 1)
    
    #print len(dfInRegion)
    
    df_count = dfInRegion.sum()/len(dfInRegion)
    df_count1 = df_count.tolist()
    df_count0 = dfInRegion.columns.tolist()
    df_counter.loc[fullname[len(fullname)-8:len(fullname)-4]] = df_count1


df_counter[df_counter<=0.25]=0
df_counter[df_counter>0.25]=1

df_countt = df_counter.sum()/len(df_counter)
df_countt1 = df_countt.tolist()
df_countt0 = dfInRegion.columns.tolist()

df_counters = pd.DataFrame(columns = listAPs, index=['0'])
df_counters.loc['0'] = df_countt1


df_counters[df_counters<=0.25]=0
df_counters[df_counters>0.25]=1

df_counterss = df_counters[(df_counters ==1)].dropna(axis=1)


#read recursively
for fullname in glob.glob(path_Post+"*.csv"):
    
    #read dataframe of a region
    dfRegion = pd.read_csv(fullname, sep=" ")
    
    #order columns
    dfRegion = dfRegion[df_counterss.columns]
    
    #save csv
    dfRegion.to_csv(fullname, sep=' ', index=False)



#%%
#%%


# Read as a test 
dfTest = pd.read_csv(path_Post+"R002.csv", sep=" ")


#%%
#%%


# Visualiza csvs into dicRegions, a dictionary composed by a dataframes for each csv
# then evaluate APs manually


dicRegions = {}

i=0

#read recursively
for fullname in glob.glob(path_Post+"*.csv"):
    #read csv of a region
    dicRegions[fullname[len(fullname)-8:len(fullname)-4]] = pd.read_csv(fullname, sep=" ")
    #increase counter
    i = i+1

#list of all APs
listAPs = dicRegions["R001"].columns.tolist()

print("All Regions APs:", dicRegions)
print("All APs:", listAPs)


#%%


#create folder
if (not os.path.exists(path_info)):
    os.makedirs(path_info)


#%%


# Save all APs originals

info_APs_all = pd.DataFrame(list(set(dup_general))+listAPs)
info_APs_all.columns = ["Alias"]

info_APs_all.to_csv(path_info+"01_info_APs_all_originals"+".csv", sep=' ', index=False)


# Save repeted APs

info_APs_repeted = pd.DataFrame(list(set(dup_general)))
info_APs_repeted.columns = ["Alias"]

info_APs_repeted.to_csv(path_info+"02_info_APs_repeted"+".csv", sep=' ', index=False)


# Save low deleted APs

info_APs_low_deleted = pd.DataFrame(df_counters[(df_counters ==1)].dropna(axis=1).columns)
info_APs_low_deleted.columns = ["Alias"]

info_APs_low_deleted.to_csv(path_info+"03_info_APs_low_deleted"+".csv", sep=' ', index=False)


# Save all APs filtered

info_APs_all_filtered = pd.DataFrame(listAPs)
info_APs_all_filtered.columns = ["Alias"]

info_APs_all_filtered.to_csv(path_info+"04_info_APs_all_filtered"+".csv", sep=' ', index=False)

#%%

print(listAPs)

#%%


# Select APs manually from "listAPs"
# Select APs manually from "listAPs"
# Select here the discarted APs!!!
# Select here the discarted APs!!!

mode = "discart"  #select discart

the_names = [] #put the names to discart or select 



if mode == "select":
    listAPs_sel = the_names
    #remove discartedAPs from listAPs, and result is listAPs_sel
    discartedAPs = listAPs[:]
    for item in listAPs_sel:
            while discartedAPs.count(item) > 0:
                discartedAPs.remove(item)


if mode == "discart":
    discartedAPs = the_names    
    #remove discartedAPs from listAPs, and result is listAPs_sel
    listAPs_sel = listAPs[:]
    for item in discartedAPs:
            while listAPs_sel.count(item) > 0:
                listAPs_sel.remove(item)



print("All APs:", listAPs)
print("Discarted APs:", discartedAPs)
print("Selected APs:", listAPs_sel)



#%%

# Save discarted APs

try:
    info_APs_discarted = pd.DataFrame(discartedAPs)
    info_APs_discarted.columns = ["Alias"]
    
    info_APs_discarted.to_csv(path_info+"05_info_APs_discarted"+".csv", sep=' ', index=False)

except:
    pass


#%%

# Save selected APs info

info_APs_selected = []

for idx in range(0,len(listAPs_sel)):
    
    ids = "W"+"%.3d" % (idx+1)
    alias = listAPs_sel[idx]#[2:len(listAPs_sel[idx])-2]
    
    info_APs_selected.append([ids, alias])
    
    print(ids, alias)

df_info_APs = pd.DataFrame(info_APs_selected)
df_info_APs.columns = ["ID","Alias"]

df_info_APs.to_csv(path_info+"06_info_APs_selected"+".csv", sep=' ', index=False)



#%%


# Only select the columns of selcted APs

dicRegions_sel = {}

for idx in dicRegions:
    #select the columns of selected APs
    dicRegions_sel[idx] = dicRegions[idx][listAPs_sel]

print("Selected Regions APs:", dicRegions_sel)
print("Selected APs:", listAPs_sel)



#%%


# Rename APs to W01 W02 W03 ...
# Save CSVs

dicRegions_sel_renam = {}


for idx in dicRegions_sel:
    #read csv of a region
    df_tmp = dicRegions_sel[idx]
    #generate names of columns
    w = ["W%.3d" % i for i in range(1,len(df_tmp.columns)+1)]
    #assign name of column as list w
    df_tmp.columns = w
    #save datagrame as element of dictionary
    dicRegions_sel_renam[i] = df_tmp
    #save csv
    dicRegions_sel_renam[i].to_csv(path_Post+idx+".csv", sep=' ', index=False)
    #show progress
    print("Saved: "+path_Post+idx+".csv")




#%%


# Save REGIONS info

info_Regions = []

#read recursively
for fullname in sorted(glob.glob(path_Pre+"*.csv")):
    
    #obtain only the name of the csv
    name_csv = fullname.replace(path_Pre, "")
    
    ids = name_csv[0:4]
    alias = name_csv[5:len(name_csv)-4]
    
    info_Regions.append([ids, alias])
    
    print(ids, alias)

df_info_Regions = pd.DataFrame(info_Regions)
df_info_Regions.columns = ["ID","Alias"]

df_info_Regions.to_csv(path_info+"info_Regions"+".csv", sep=' ', index=False)



#%%
