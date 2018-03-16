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

# train_list = [[6, 180, 12], [5.92, 190, 11], [5.58, 170, 12], [5.92, 165, 10], [5, 100, 6], [5.5, 150, 8], [5.42, 130, 7], [5.75, 150, 9]]
# train_labels_list = [0, 0, 0, 0, 1, 1, 1, 1]
# test = [[6, 180, 12]]
# test_labels = [0]

# train_list = [[182, 81.6, 30], [180, 86.2, 28], [170, 77.1, 30], [180, 74.8, 25], [152, 45.4, 15], [168, 68.0, 20], [165, 59.0, 18], [175, 68.0, 23]]
# train_labels_list = [0, 0, 0, 0, 1, 1, 1, 1]

# train = np.array(train_list).astype(np.float)
# train_labels = np.array(train_labels_list)

# Initializer vos parametres

# Initializer/instanciez vos classifieurs avec leurs parametres
knn_classifier = Knn.Knn()
bayesNaif_classifier = BayesNaif.BayesNaif()

# Charger/lire les datasets
(train, train_labels, test, test_labels) = load_datasets.load_monks_dataset(3)
# (train, train_labels, test, test_labels) = load_datasets.load_iris_dataset(0.5)
# (train, train_labels, test, test_labels) = load_datasets.load_congressional_dataset(0.5)

# # Entrainez votre classifieur
knn_classifier.train(train, train_labels)
bayesNaif_classifier.train(train, train_labels)

# # Tester votre classifieur
# knn_classifier.test(test, test_labels)
bayesNaif_classifier.test(test, test_labels)
