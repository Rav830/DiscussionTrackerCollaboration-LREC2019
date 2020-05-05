# DiscussionTrackerCollaboration-LREC2020
This is code for the Discussion Tracker Project set to produce the results found in our LREC submission



## Folder Description

The folders in this project are:

1. Data - This folder will contain the data to be used by the code to generate the models 
2. Python - This folder contains the code for the models and prints the results each python file (except for the header files and some other configuration files) corresponds to one model.
3. Results - This folder contains a text file of the same results seen on the console (mainly used for convenience)


## Data
The data folder is currently empty and needs to be filled in. Go to the [data site](discussiontracker.cs.pitt.edu) and fill out the google form to get access to the data.

Once you get the data, place the excel files inside the folder called 'EAGER' in the 'Data' folder.

## Packages and Python Version

The packages used are in the Python/requirements.txt file and can be installed using 
```
pip install -r requirements.txt
```

Also this code was tested on python3.7 



## Usage

```
cd Python
python3.7 model_Dummy.py eager
OR
cd Python
./cmdList.sh #to run all experiments
```

This code uses arguments to pull off most of the results that are seen. To get the results seen in the report run following commands. 
```
python3.7 model_Naive_Bayes.py eager --remove-non --tf-idf --use-cv ../Data/Eager_10_fold_crossvalidation.json

python3.7 model_Gaussian_Naive_Bayes_regroup.py eager --remove-non --use-cv ../Data/Eager_10_fold_crossvalidation.json

python3.7 model_DetectNon_Gaussian_Naive_Bayes.py eager --use-cv ../Data/Eager_10_fold_crossvalidation.json
```

