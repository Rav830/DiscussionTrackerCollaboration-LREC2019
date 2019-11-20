import config
tee = config.Tee('../Results/%s/MNB%s%s_model.txt' % (config.args.dataset, config.args.tf_idf, config.args.remove_non), 'w')
from header_model_data import *
from sklearn.naive_bayes import MultinomialNB



print("Defining and doing a Multinomial Naive Bayes classifier")

NB = MultinomialNB(alpha = 1, fit_prior = True, )
scores = cross_validate(NB, xDF, yDF, cv=logo, scoring = scorer)

#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
compute_stats(originalclass, predictedclass, True)
print("Confusion Matrix")

labelList = list(range(4))
if(config.args.remove_non):
	labelList = [x+1 for x in labelList]
else:
	labelList.append(4)

print_cm(confusion_matrix(originalclass, predictedclass, labels = labelList), [pr.conversion_y(x) for x in labelList])

tee.close()
