
# Indoor Loacation using Bayes
# code for offline processing
# Spyder, Python 3.6 (Ancaconda2 version)
# By: Dennis Nunez Fernandez


#%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import itertools
import operator
import glob



#%%

plt.close('all')

DATA = "DATA_001_House"

path_data  = "data1_filtered_wifis/"+DATA+"/"

path_parts = "data2_divided_parts/"+DATA+"/"


numRegions = 0
for fullname in glob.glob(path_data+"*.csv"):
    dfInRegion = pd.read_csv(fullname, sep=" ")
    numAPs = len(dfInRegion.columns)
    numRegions = numRegions+1


Regions = ["R%.3d" % i for i in range(1,numRegions+1)]
APs = ["W%.3d" % i for i in range(1,numAPs+1)]


#%%

# Generate tables to see important Regions and APs: df_counter, df_counters_w, df_counters_r

df_counter = pd.DataFrame(columns = APs, index =Regions)

#read recursively
for fullname in glob.glob(path_data+"*.csv"):
    
    print(fullname)
    
    #read dataframe of a region
    dfInRegion = pd.read_csv(fullname, sep=" ")
    
    for x in range(8,101):
        if -x ==-100:
            dfInRegion = dfInRegion.replace(-x, 0)
        dfInRegion = dfInRegion.replace(-x, 1)
        
    df_count = dfInRegion.sum()/len(dfInRegion)
    df_count1 = df_count.tolist()
    df_count0 = dfInRegion.columns.tolist()
    df_counter.loc[fullname[len(fullname)-8:len(fullname)-4]] = df_count1


df_counter_ = df_counter.copy()
df_counter_[df_counter_<=0.6]=0
df_counter_[df_counter_>0.6]=1


df_countt = df_counter_.sum()/len(df_counter_)
df_countt1 = df_countt.tolist()
df_countt0 = dfInRegion.columns.tolist()

df_counters_w = pd.DataFrame(columns = APs, index=['0'])
df_counters_w.loc['0'] = df_countt1


df_countt = df_counter_.sum(axis=1)/len(APs)
df_countt1 = df_countt.tolist()
df_countt0 = dfInRegion.columns.tolist()

df_counters_r = pd.DataFrame(columns = Regions, index=['0'])
df_counters_r.loc['0'] = df_countt1



#%%

# SELECT Regions and APs from df_counter, df_counters_w, df_counters_r
# SELECT Regions and APs from df_counter, df_counters_w, df_counters_r
# SELECT Regions and APs from df_counter, df_counters_w, df_counters_r
# SELECT Regions and APs from df_counter, df_counters_w, df_counters_r

Regions_ = [1,2,3,4]
APs_ = [2,8]

Regions = ["R%.3d" % i for i in Regions_]
APs = ["W%.3d" % i for i in APs_]


numRegions = len(Regions)
numAPs = len(APs)



#%%


# Import training datasets

# Folds

parts = ['part01','part02','part03','part04','part05','part06','part07','part08','part09','part10']

test_parts = ['part01','part02']

train_parts = [x for x in parts if x not in test_parts]


#%%


# SELECT TRAINING

Dtr = {}

for r in Regions:
    Dtr[r] = pd.DataFrame()
    for p in train_parts:
        name = r+".csv"
        Dtr[r+p] = pd.read_csv(path_parts+p+"/"+name, sep=" ")
        Dtr[r+p] = Dtr[r+p][APs]
        Dtr[r] = Dtr[r].append(Dtr[r+p], ignore_index=True)
        print("Reading "+ path_parts+p+"/"+name)



# SELECT TESTING

Dte = {}

for r in Regions:
    Dte[r] = pd.DataFrame()
    for p in test_parts:
        name = r+".csv"
        Dte[r+p] = pd.read_csv(path_parts+p+"/"+name, sep=" ")
        Dte[r+p] = Dte[r+p][APs]
        Dte[r] = Dte[r].append(Dte[r+p], ignore_index=True)
        print("Reading "+ path_parts+p+"/"+name)


    

#%%
#%%

# CLEAN THE TRAINING RSSI DATASETS BY USING GAUSSIAN CURVES


# Import training datasets


# Define function to calcule Gaussian histogram

def hist_gauss(values):
    val = sorted(values[values!=-100.0])
    x = range(-100, -10)
    mu = np.mean(val)
    sigma = np.std(val)
    if np.isnan(mu) and np.isnan(sigma):
        y = [0]*len(x)
    elif mu==0 or sigma==0:
        y = [0]*len(x)
    else:
        y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * \
        (np.power(np.e, -(np.power((x - mu), 2) / (2 * np.power(sigma, 2)))))
    return y


# Calculate Gaussian Histograms

Dtr_g = {}

for r in Regions:
    for w in APs:
        print("Calculating Gaussian Histogram for "+r+"_"+w)
        Dtr_g[r+"_"+w] = hist_gauss(Dtr[r][w])
       

# As instance, draw histograms at region R1
        
R_draw = Regions[0]

print("Ploting for " + R_draw+ "\n ...")

color = cm.rainbow(np.linspace(0,1,numAPs+1))

# Plot Histogram
for w in APs:
    plt.hist(Dtr[R_draw][w][Dtr[R_draw][w]!=-100],bins=range(-100, -10),alpha=0.5,color=color[APs.index(w)],density=True,label=w+" histogram")
    
# Plot Gaussians
for w in APs:
    plt.plot(range(-100, -10), Dtr_g[R_draw+"_"+w],color=color[APs.index(w)],label=w+" Gaussian histogram")

# Show
plt.title("APs histograms at "+"region " + R_draw)
plt.legend()
plt.show()


#%%


# PERFORM ANALYSIS OF BAYESIAN FILTERS OFFLINE


dict_params = {}
dict_w_post = {}

# Create tables for Wn
for w in APs:
    dict_params[w] = [0]*90

    for r in Regions:
        dict_params[w] = np.vstack((dict_params[w],Dtr_g[r+"_"+w]))
    
    dict_params[w] = dict_params[w][1:numRegions+1][:]
    
# W1 is composed by:
# R1 x W1_gauss
# R2 x W1_gauss
# R3 x W1_gauss
# R4 x W1_gauss
    

# Function for prediction using Bayes
    
def prediction_region(row):
    
    bins = range(-100, -9)
    
    for w in APs:
        dict_params["value"+w] = row[w]
        dict_params["indx"+w] = bins.index(dict_params["value"+w]) 
    
    
    # we take uniform distribution as prior
    for w in APs:
        dict_params["prior"+w] = [0]*1
        for r in Regions:
            dict_params["prior"+w] = np.vstack((dict_params["prior"+w],[1.0/numRegions]))
        dict_params["prior"+w] = dict_params["prior"+w][1:numRegions+1][:]
        print("prior"+w+"\n", np.array(dict_params["prior"+w]).round(decimals=3))
        
    
    accuracy = 0
    
    while accuracy < 0.95:
        
        print("\n")
        
        
        for w in APs:
            
            print ("Starting Bayes for "+w)
            
            # Apply Bayes
            dict_params["posterior"+w] = dict_params["prior"+w] * np.vstack(dict_params[w][:,dict_params["indx"+w]])
            
            if np.sum(dict_params["posterior"+w])!= 0:
                # Normalization
                dict_params["posterior"+w+"_norm"] = np.asarray([float(i)/sum(dict_params["posterior"+w]) for i in dict_params["posterior"+w]])
                
                # Max normalized value
                dict_w_post["max_"+"posterior"+w+"_norm"] = np.amax(dict_params["posterior"+w+"_norm"])
                
                # Predicted region
                dict_params["posterior"+w+"_pred"] = Regions[np.where(dict_params["posterior"+w+"_norm"]==dict_w_post["max_"+"posterior"+w+"_norm"])[0][0]]
            
            else:
                # Normalization
                dict_params["posterior"+w+"_norm"] = [0]*len(dict_params["posterior"+w])
                
                # Max normalized value
                dict_w_post["max_"+"posterior"+w+"_norm"] = 0
                
                # Predicted region
                dict_params["posterior"+w+"_pred"] = 0
            
            print("posterior"+w+"_norm",": \n", np.array(dict_params["posterior"+w+"_norm"]).round(decimals=3))
        
        
        # Select the highest accuracy after the first iteration
        # be careful when two regions have the same accuracy
        max_W = max(dict_w_post.items(), key=operator.itemgetter(1))[0][13:17]
        
        # assign values
        for w in APs:
            dict_params["prior"+w] = dict_params["posterior"+max_W+"_norm"]
            
        accuracy = dict_w_post["max_"+"posterior"+max_W+"_norm"]
        prediction = int(dict_params["posterior"+max_W+"_pred"][1:4])
        print("\nContinue with ",max_W)
        
        print("Pred:", "", dict_params["posterior"+max_W+"_pred"], ", Acc:", np.array(accuracy).round(decimals=3))
        
    return prediction, accuracy


#%%
#%%
    

# Evaluating for a single measurement
    
values = {}

for w in APs:
    values[w] = -60

for w in APs:
    print(w+" meassure:"+str(values[w])+"dBm")
print("\n \n")

pred, acc = prediction_region(values)

print("\n \n")
print("Predicted region:", pred)
print("Accuracy:", "%.2f" % (100*acc), "%")



#%%
#%%


## CONSTRUCTION OF CONFUSION MATRIX 


# Read testing dataset


for r in Regions:
    # assign true values
    d = {'true': int(r[1:4])*np.ones(len(Dte[r]), dtype=int)}
    df = pd.DataFrame(data=d)
    Dte[r] = Dte[r].join(df)


acc = Dte[Regions[0]] 
Regionss = list(Regions)
Regionss.remove(Regionss[0])
for r in Regionss:
    acc = acc.append(Dte[r])
test_R = acc.reset_index(drop=True)

# Shuffle all testing dataset
test_R = test_R.sample(frac=1).reset_index(drop=True)

# True values of testing dataset
R_true = test_R["true"].tolist()

# Predicted values of testing dataset
R_predicted = test_R.apply(prediction_region, axis=1)
R_predicted = [row[0] for row in R_predicted]

# Set labels
labels = Regions

# Calculation of Confusion Matrix
cm = confusion_matrix(R_true, R_predicted)

# Normalize Confusion Matrix
cm = cm / cm.astype(np.float).sum(axis=1) 


# Plot Confusion Matrix
fig = plt.figure(figsize = (numRegions,numRegions))
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, float("{0:.2f}".format(round(cm[i, j],2))),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
i = range(len(labels))
ax.set(xticks=i, xticklabels=labels, yticks=i, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
#plt.savefig('confusion_matrix.png', format='png')
plt.show()

# Show accuracy
acc = 100*accuracy_score(R_true, R_predicted)

print("\n\n")
print("========================")
print("Overall accuracy: "+ "%.2f" % acc+"%")
print("========================")


#%%

#%%
