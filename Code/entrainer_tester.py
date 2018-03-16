import numpy as np
import sys
import load_datasets
import BayesNaif # importer la classe du classifieur bayesien
import Knn # importer la classe du Knn
#importer dautres fichiers et classes si vous en avez developpes

#-*- coding: utf-8 -*-

"""
Cest le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire cest quoi les parametres que vous avez utilises
En gros, vous allez :
1- Initialiser votre classifieur avec ses parametres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester

"""

# Initializer vos parametres

# Initializer/instanciez vos classifieurs avec leurs parametres
knn_iris = Knn.Knn()
knn_congress = Knn.Knn()
knn_monks1 = Knn.Knn()
knn_monks2 = Knn.Knn()
knn_monks3 = Knn.Knn()

bayesNaif_iris = BayesNaif.BayesNaif()
bayesNaif_congress = BayesNaif.BayesNaif()
bayesNaif_monks1 = BayesNaif.BayesNaif()
bayesNaif_monks2 = BayesNaif.BayesNaif()
bayesNaif_monks3 = BayesNaif.BayesNaif()

# Charger/lire les datasets
(train_iris, train_labels_iris, test_iris, test_labels_iris) = load_datasets.load_iris_dataset(0.5)
(train_congress, train_labels_congress, test_congress, test_labels_congress) = load_datasets.load_congressional_dataset(0.5)
(train_monks1, train_labels_monks1, test_monks1, test_labels_monks1) = load_datasets.load_monks_dataset(1)
(train_monks2, train_labels_monks2, test_monks2, test_labels_monks2) = load_datasets.load_monks_dataset(2)
(train_monks3, train_labels_monks3, test_monks3, test_labels_monks3) = load_datasets.load_monks_dataset(3)

# # Entrainez votre classifieur
knn_iris.train(train_iris, train_labels_iris)
knn_congress.train(train_congress, train_labels_congress)
knn_monks1.train(train_monks1, train_labels_monks1)
knn_monks2.train(train_monks2, train_labels_monks2)
knn_monks3.train(train_monks3, train_labels_monks3)

bayesNaif_iris.train(train_iris, train_labels_iris)
bayesNaif_congress.train(train_congress, train_labels_congress)
bayesNaif_monks1.train(train_monks1, train_labels_monks1)
bayesNaif_monks2.train(train_monks2, train_labels_monks2)
bayesNaif_monks3.train(train_monks3, train_labels_monks3)

# # Tester votre classifieur
print "Knn - iris"
knn_iris.test(test_iris, test_labels_iris)
print "****************"
print "Knn - congress"
knn_congress.test(test_congress, test_labels_congress)
print "****************"
print "Knn - monks 1"
knn_monks1.test(test_monks1, test_labels_monks1)
print "****************"
print "knn - monks 2"
knn_monks2.test(test_monks2, test_labels_monks2)
print "****************"
print "Knn - monks 3"
knn_monks3.test(test_monks3, test_labels_monks3)
print "****************"

print "Bayes - iris"
bayesNaif_iris.test(test_iris, test_labels_iris)
print "****************"
print "Bayes - congress"
bayesNaif_congress.test(test_congress, test_labels_congress)
print "****************"
print "Bayes - monks 1"
bayesNaif_monks1.test(test_monks1, test_labels_monks1)
print "****************"
print "Bayes - monks 2"
bayesNaif_monks2.test(test_monks2, test_labels_monks2)
print "****************"
print "Bayes - monks 3"
bayesNaif_monks3.test(test_monks3, test_labels_monks3)
print "****************"
