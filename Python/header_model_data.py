from config import *
import pandas as pd
import numpy as np
import os
from collections import Counter

from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_validate
from sklearn.utils import resample

import prep_data as pr
from borrowed_code import *


#extracted_file = "../Data/extracted_data_20.pkl"
#extracted_file = "../Data/extracted_data.pkl"

start = 0;
end = 0;
#extracted_file = "../Data/%s/extracted_data_%s.pkl"%(args.dataset, args.tf_idf)
if(args.num_files == -1):
    start = None
    end = None
    extracted_file = "../Data/%s/extracted_data%s%s.pkl"%(args.dataset, args.tf_idf, args.remove_non)
else:
    start = 0
    end = args.num_files
    extracted_file = "../Data/%s/extracted_data_%d%s%s.pkl" % (args.dataset, end, args.tf_idf, args.remove_non)

if(args.rebuild_data):
    try:
        os.remove(extracted_file)
    except FileNotFoundError:
        print("Already not there")

#read a file instead of doing this process
try:
    df = pd.read_pickle(extracted_file)
    print("pre-made Data Found")
except FileNotFoundError:
    print("Creating the Data")
    dataPath = "../Data/%s/" %(args.dataset)
    di = [dataPath+x for x in os.listdir(dataPath) if x.endswith(".xlsx") or x.endswith(".xls")][start:end]
    x,y,df = pr.gen_data_and_df(di,removeNon = args.remove_non ,isTrain = True, cvSplits = args.use_cv)
    if(args.tf_idf):
    	print("Converting to TF-IDF")
    	df = pr.to_tf_idf(df)
    df.to_pickle(extracted_file)
 

"""
for x in df.columns:
    #print(x)
    val = df[x].value_counts()
    val = val/sum(val)
    if(0 not in val):
        val[0] = 0
    if(val[0] < .7):
        print(x)
print(len(df.columns))
"""

print("The Class Distribution is:")
classDist = Counter(df['|-Collab'])
for k in classDist.keys():
    print("\t"+str(pr.conversion_y(k))+":"+str(classDist[k]))

print("The number of features is:")
print("\t" + str(len(df.columns)-2))

#raise SystemExit
### From https://stackoverflow.com/a/42567557
# Variables for average classification report
originalclass = []
predictedclass = []
accuracyscores = []
macro_f1 = []
cohen_kappa = []
#reset the statistics if it will run multiple times 
def reset_stats():
    global originalclass, predictedclass, accuracyscores, macro_f1
    originalclass = []
    predictedclass = []
    accuracyscores = []
    macro_f1 = []
    cohen_kappa = []

#one idea to handle this is to write each score to disk and then read it in when I want to compute the stats.
def classification_report_with_metric(y_true, y_pred):
    global originalclass, predictedclass
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    accuracyscores.append(accuracy_score(y_true, y_pred))
    macro_f1.append( f1_score(y_true, y_pred, average = 'macro') )
    cohen_kappa.append( cohen_kappa_score(y_true, y_pred) )
    return cohen_kappa[-1] # return kappa score 

###
def compute_stats(actual, predicted, doPrint = False):
    npActual = np.array(actual)
    npPredict = np.array(predicted)
    npAcc = np.array(accuracyscores)
    npF1 = np.array(macro_f1)
    npK = np.array(cohen_kappa)
    #labels = [0,1,2,3,4]
    labels = list(Counter(actual).keys())
    #rt(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False)[source]¶
    if(doPrint):
        print("Accuracy: %0.3f (+/- %0.3f)" % (npAcc.mean(), npAcc.std() * 2))
        print("Macro-F1: %0.3f (+/- %0.3f)" % (npF1.mean(), npF1.std() * 2))
        print("Cohen-Kappa: %0.3f (+/- %0.3f)" % (npK.mean(), npK.std() * 2))
        print(classification_report(npActual, npPredict, 
                                   labels = labels, 
                                   target_names = [pr.conversion_y(x) for x in labels],
                                   digits = 3))
    retval = classification_report(npActual, npPredict, 
                                   labels=labels, 
                                   target_names = [pr.conversion_y(x) for x in labels],
                                   digits = 3, output_dict=True)
    
    return retval

print("Defining vars for ease of use")
xDF = df[df.columns.difference(['|-Collab','|-group', '|-transcript'])]
yDF = df['|-Collab']
gDF = df['|-group']
tDF = df['|-transcript']
logo = [x for x in LeaveOneGroupOut().split(xDF, yDF, gDF)]

scorer = make_scorer(classification_report_with_metric)
#print(list(logo))

if __name__ == "__main__":
	print("This is just a test of the header data file.")
	print("The preprocessing handled the dataset %s" %(args.dataset))
	print("The location of the data is %s" %(extracted_file))
	print("The flag for tfidf is %s" %(args.tf_idf))
	#print(args)
	#print(xDF)
	#print(set(gDF))
	
	print(Counter(gDF))
	print(Counter(tDF))

