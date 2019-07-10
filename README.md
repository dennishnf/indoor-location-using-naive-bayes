
# Indoor Location using Naive Bayes #

The scripts were intended to describe a Naive-Bayes-based approach to indoor location via WiFi signals. The data was captured with a Samsung J2 smartphone, however, another device with a WiFi Wireless module could be used. The scripts allow you to select the best access points, order the data into testing/training, visualize the data, perform Naive Bayes algorithm and obtain some metrics as indoor location accuracy and confusion matrix.

Such a scripts were developed from scratch with Python 3.6, Anaconda and Spyder.

## - data0_original/ ##

Principal and initial folder. This data is captured in the next format:

```
[sample1] <AP1-NAME> <AP1-ADDRESS> <AP1-LEVEL> <AP2-NAME> <AP2-ADDRESS> <AP2-LEVEL> ...
[sample2] <AP1-NAME> <AP1-ADDRESS> <AP1-LEVEL> <AP2-NAME> <AP2-ADDRESS> <AP2-LEVEL> ...
[sample3] <AP1-NAME> <AP1-ADDRESS> <AP1-LEVEL> <AP2-NAME> <AP2-ADDRESS> <AP2-LEVEL> ...
... 
[sampleN] <AP1-NAME> <AP1-ADDRESS> <AP1-LEVEL> <AP2-NAME> <AP2-ADDRESS> <AP2-LEVEL> ...
```

## - 1_filter_wifis.py ## 

Selects the best access points in order to be used for indoor localization. The filtered data is stored in the folders: "data1_filtered_wifis" and "data_info". The folder "data1_filtered_wifis" contains the sorted data only with the selected access points. The folder "data_info" contains the information about the nomenclature of regions and access points which are selected of discarded. 

## - 2_division_data.py ##

Divide the data ("data1_filtered_wifis") into 10 balanced parts in order to be used for creation of data/testing datasets. The division is stored in the folder "data2_divided_parts". We divided in 10 parts because the next script ("3_indoor_localization.py") we can select the parts for training and testing (10% or 20% for testing for example) or perform cross validation analysis.

## - 3_indoor_localization.py ##

Perform indoor location prediction based on Naive Bayes. Here you select the regions and access points to be used in the analysis. Also, you can select the percent for testing/training. The results are based on the confusion matrix and average location accuracy. This script allows you to visualize the original histogram and the cleaned Gaussian histograms. The pseudo algorithm is shown as follows:

```
N = number of regions; R = number of routers
W r = AP table f or router r
procedure BayesEstimator(w 1 , w 2 , ..., w R )
  # Start with uniform distribution
  priorW 1,2,..,R = [1/N ; 1/N ; ...; 1/N ] N x1
  probability = (100/N )%
  while probability < 95% do
    # Perform Bayes
    for r from 1 to R do
      posteriorW r = norm(priorW r × W r [:, w r ])
      prob r = max(posteriorW r )
      pred r = where(posteriorW r == prob r )
    end for
    # Find the highest probability
    probability = max(prob 1,2,..,R )
    r best = where(prob 1,2,..,R == probability)
    prediction = pred r best
    # Update the new prior
    for r from 1 to R do
      priorW r = posteriorW r best
    end for
  end while
  Return prediction, probability
end procedure
```


## License ##

GNU General Public License, version 3 (GPLv3).

You can visit my personal website: [http://dennishnf.bitbucket.io](http://dennishnf.bitbucket.io)!.


