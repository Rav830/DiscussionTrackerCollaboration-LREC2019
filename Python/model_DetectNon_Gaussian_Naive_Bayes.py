import config
tee = config.Tee('../Results/%s/Non_GNB%s%s_model.txt' % (config.args.dataset, config.args.tf_idf, config.args.remove_non), 'w')
from header_model_data import *
from sklearn.naive_bayes import GaussianNB

print("Regrouping the labels")
for i in range(len(yDF)):
    if(yDF[i] != pr.y_conversion('Non')):
        yDF[i] = pr.y_conversion('new-idea')


print("Defining and doing a Gaussian Naive Bayes classifier Trying to detect Non")

NB = GaussianNB()
scores = cross_validate(NB, xDF, yDF, cv=logo, scoring = scorer)

#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
compute_stats(originalclass, predictedclass, True)
print("Confusion Matrix")
print_cm(confusion_matrix(originalclass, predictedclass), [pr.conversion_y(x) for x in range(2)])

#labels = list(range(5)))
tee.close()
