"""
Vous allez definir une classe pour chaque algorithme que vous allez developper,
votre classe doit contenit au moins les 3 methodes definies ici bas,
    * train     : pour entrainer le modele sur l'ensemble d'entrainement
    * predict     : pour predire la classe d'un exemple donne
    * test         : pour tester sur l'ensemble de test
vous pouvez rajouter d'autres methodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les methodes test, predict et test de votre code.
"""

import numpy as np

class Knn:
    def __init__(self, **kwargs):
        """
        c'est un Initializer.
        Vous pouvez passer d'autre parametres au besoin,
        c'est a vous d'utiliser vos propres notations
        """
        self.k = 5
        #Rennome train_list a cause conflit avec fonction train
        self.train_list= []
        self.train_labels = []

    def train(self, train, train_labels): #vous pouvez rajouter d'autres attribus au besoin

        self.train_list= train
        self.train_labels = train_labels


    def predict(self, exemple, label):


        """
        Predire la classe d'un exemple donne en entree
        exemple est de taille 1xm

        si la valeur retournee est la meme que la valeur dans label
        alors l'exemple est bien classifie, si non c'est une missclassification

        """
        prediction = self.k_nearest_neighbor(exemple)


        value = 1 if (prediction == label) else 0
        return value, prediction


    def test(self, test, test_labels):
        """
        c'est la methode qui va tester votre modele sur les donnees de test
        l'argument test est une matrice de type Numpy et de taille nxm, avec
        n : le nombre d'exemple de test dans le dataset
        m : le mobre d'attribus (le nombre de caracteristiques)

        test_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter d'autres arguments, il suffit juste de
        les expliquer en commentaire

        Faites le test sur les donnees de test, et afficher :
        - la matrice de confision (confusion matrix)
        - l'accuracy
        - la precision (precision)
        - le rappel (recall)

        Bien entendu ces tests doivent etre faits sur les donnees de test seulement
        """
        predictions = []
        for i in range(0, len(test)):
            value, prediction = self.predict(test[i], test_labels[i])
            predictions.append(prediction)
        self.confusion_matrix(predictions, test_labels)

    def k_nearest_neighbor(self, data):
        dist_index = []
        labels = []
        for i in range(0, len(self.train_list)):
            dist = np.linalg.norm(np.array(data) - self.train_list[i])
            dist_index.append([dist, i])
        dist_index = sorted(dist_index)
        # print "dist_index : ", dist_index
        # print "train : ", self.train_labels
        for i in range(0, self.k):
            # print "test : ", self.train_labels[dist_index[i][1]]
            index = dist_index[i][1]
            # print "index : ", index
            labels.append(self.train_labels[index])
        # print "labels: ",labels
        most_common_item = max(set(labels), key=labels.count)
        # print "most_common_item : ", most_common_item
        return most_common_item

    def confusion_matrix(self, predictions, test_labels):
        labels = set(test_labels)
        test = np.array(test_labels)
        predictions = np.array(predictions)

        # calculate the confusion matrix; labels is numpy array of classification labels
        matrix = np.zeros((len(labels), len(labels)))
        for a, p in zip(test, predictions):
            matrix[a][p] += 1

        false_positive = matrix.sum(axis=0) - np.diag(matrix)
        false_negative = matrix.sum(axis=1) - np.diag(matrix)
        true_positive = np.diag(matrix)
        true_negative = matrix.sum() - (false_negative + false_positive + true_positive)

        accuracy = (true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive)
        print "accuracy: ", accuracy

        print "confusion matrix : "
        print matrix

        precision = true_positive / (true_positive + false_positive)
        print "Precision : ", precision

        recall = true_positive / (true_positive + false_negative)
        print "Recall : ", recall
