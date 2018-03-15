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
        self.train = []
        self.train_labels = []

    def train(self, train, train_labels): #vous pouvez rajouter d'autres attribus au besoin

        self.train = train
        self.train_labels = train_labels


    def predict(self, exemple, label):


        """
        Predire la classe d'un exemple donne en entree
        exemple est de taille 1xm

        si la valeur retournee est la meme que la valeur dans label
        alors l'exemple est bien classifie, si non c'est une missclassification

        """
        prediction = self.k_nearest_neighbor(exemple)
        return 1 if prediction == label else 0


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
        accuracy = 0
        for i in range(1, len(test)):
            value = self.predict(test[i], test_labels[i])
            accuracy += value

        print "accuracy : ", accuracy / len(test)



    # def knearest(self, data, k, compare_matrix, compare_matrix_labels) :
    #     index = []
    #     labels = []
    #     #On check les k nearest
    #     for i in range(0, k):
    #
    #         # best_dist = np.linalg.norm(row - compare_matrix[0]) #On veut tu vrm comparer avec la premier value
    #         best_dist = 999999
    #         best_index = 0
    #         for i in range(1, len(compare_matrix)):
    #             #Peut etre compare_matrix simplement
    #             dist = np.linalg.norm(np.array(data) - compare_matrix[i])
    #             # print "dist: ",dist
    #             if dist < best_dist and i not in index:
    #                 best_dist = dist
    #                 best_index = i
#         index.append(best_index)
    #         labels.append(compare_matrix_labels[best_index])
    #     most_common_item = max(set(labels), key=labels.count)
    #     print "most_common_item : ", most_common_item
    #     return most_common_item

    def k_nearest_neighbor(self, data):
        dist_index = []
        labels = []

        for i in range(1, len(data)):
            #Peut etre compare_matrix simplement
            dist = np.linalg.norm(np.array(data) - self.train[i])
            dist_index.append([dist, i])
        dist_index = sorted(dist_index)
        for i in range(1, self.k):
            labels.append(self.train_label[dist_index[i][1]])
        most_common_item = max(set(labels), key=labels.count)
        print "most_common_item : ", most_common_item
        return most_common_item
