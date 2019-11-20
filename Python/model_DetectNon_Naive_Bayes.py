import config
tee = config.Tee('../Results/%s/Non_MNB%s%s_model.txt'% (config.args.dataset, config.args.tf_idf, config.args.remove_non), 'w')
from header_model_data import *
from sklearn.naive_bayes import MultinomialNB

print("Regrouping the labels")
for i in range(len(yDF)):
    if(yDF[i] != pr.y_conversion('Non')):
        yDF[i] = pr.y_conversion('new-idea')
print("Post Regroup Stats")
print("The Class Distribution is:")
classDist = Counter(yDF)
for k in classDist.keys():
    print("\t"+str(pr.conversion_y(k))+":"+str(classDist[k]))

alpha = 1
print("Defining and doing a Multinomial Naive Bayes classifier Trying to detect Non")
print("this is smoothed to an alpha of %s"%(alpha))
NB = MultinomialNB(alpha =alpha, fit_prior = True, )
scores = cross_validate(NB, xDF, yDF, cv=logo, scoring = scorer)

#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
compute_stats(originalclass, predictedclass, True)
print("Confusion Matrix")
print_cm(confusion_matrix(originalclass, predictedclass), [pr.conversion_y(x) for x in range(2)])

tee.close()
