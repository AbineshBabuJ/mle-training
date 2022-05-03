# mle-training

# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techniques are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Scripts
 - ingest_data.py - To download and create training and validation datasets.
 - train.py       - To train the model.
 - score.py       - To score the model.
 - testing.py     - To check for proper installation of packages. Checked after performing the import functions.

## To excute the script
python < scriptname.py >
