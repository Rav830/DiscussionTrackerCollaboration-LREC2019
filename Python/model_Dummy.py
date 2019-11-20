import config
tee = config.Tee('../Results/%s/Dummy%s%s_model.txt'  % (config.args.dataset, config.args.tf_idf, config.args.remove_non), 'w')
from header_model_data import *
from sklearn.dummy import DummyClassifier




print("Defining and doing a dummy classifier set to predict based on the class distribution")

dumDum = DummyClassifier(strategy='stratified', random_state=None, constant = None)
scores = cross_validate(dumDum, xDF, yDF, cv=logo, scoring = scorer)

#print(scores)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
compute_stats(originalclass, predictedclass, True)
print("Confusion Matrix")
print_cm(confusion_matrix(originalclass, predictedclass, labels = list(range(5))), [pr.conversion_y(x) for x in range(5)])

tee.close()
