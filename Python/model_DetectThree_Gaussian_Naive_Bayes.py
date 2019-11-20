import config
tee = config.Tee('../Results/%s/Three_Label_GNB%s%s_model.txt'% (config.args.dataset, config.args.tf_idf, config.args.remove_non), 'w')
from header_model_data import *
from sklearn.naive_bayes import GaussianNB

print("Regrouping the labels")
labelCheck = [pr.y_conversion(x) for x in ['agree', 'challenge', 'extension']]
labelList = set([])
for i in range(len(yDF)):
    if(yDF[i] in labelCheck):
        yDF[i] = labelCheck[0]
print("Post Regroup Stats")
print("The Class Distribution is:")
classDist = Counter(yDF)
for k in classDist.keys():
    labelList.add(pr.conversion_y(k))
    print("\t"+str(pr.conversion_y(k))+":"+str(classDist[k]))

alpha = 1
print("Defining and doing a Gaussian Naive Bayes classifier Trying to three class labels")
print("Non is non, new-idea is initiation, agree is the other labels.")
print("this is smoothed to an alpha of %s"%(alpha))
NB = GaussianNB()
scores = cross_validate(NB, xDF, yDF, cv=logo, scoring = scorer)

#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
compute_stats(originalclass, predictedclass, True)
print("Confusion Matrix")
print_cm(confusion_matrix(originalclass, predictedclass), list(labelList))

tee.close()
