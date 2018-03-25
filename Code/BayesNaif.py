"""
Vous allez definir une classe pour chaque algorithme que vous allez developper,
votre classe doit contenit au moins les 3 methodes definies ici bas,
    * train     : pour entrainer le modele sur lensemble dentrainement
    * predict     : pour predire la classe dun exemple donne
    * test         : pour tester sur lensemble de test
vous pouvez rajouter dautres methodes qui peuvent vous etre utiles, mais moi
je vais avoir besoin de tester les methodes test, predict et test de votre code.
"""
import numpy as np
#Import pour calculer le temps d execution de test
import timeit

class BayesNaif:

    def __init__(self, **kwargs):
        """
        cest un Initializer.
        Vous pouvez passer dautre parametres au besoin,
        cest a vous dutiliser vos propres notations
        """
        self.group = {}
        #Rennome train_list a cause conflit avec fonction train
        self.train_list = []
        self.train_labels = []
        self.means = {}
        self.variances = {}

    def train(self, train, train_labels): #vous pouvez rajouter dautres attribus au besoin
        """
        cest la methode qui va entrainer votre modele,
        train est une matrice de type Numpy et de taille nxm, avec
        n : le nombre dexemple dentrainement dans le dataset
        m : le mobre dattribus (le nombre de caracteristiques)

        train_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter dautres arguments, il suffit juste de
        les expliquer en commentaire

        ------------
        Apres avoir fait lentrainement, faites maintenant le test sur
        les donnees dentrainement
        IMPORTANT :
        Vous devez afficher ici avec la commande print() de python,
        - la matrice de confision (confusion matrix)
        - laccuracy
        - la precision (precision)
        - le rappel (recall)

        Bien entendu ces tests doivent etre faits sur les donnees dentrainement
        nous allons faire dautres tests sur les donnees de test dans la methode test()
        """
        self.group = set(train_labels)
        self.train_list = train
        self.train_labels = train_labels
        self.means, self.variances = self.bayes(train, train_labels)


    def predict(self, exemple, label):
        """
        Predire la classe dun exemple donne en entree
        exemple est de taille 1xm

        si la valeur retournee est la meme que la veleur dans label
        alors lexemple est bien classifie, si non cest une missclassification

        """

        probability = self.probability(self.means, self.variances, self.group, exemple)
        best_label = -1
        best_prob = -1
        for l in probability:
            if probability[l] > best_prob:
                best_prob = probability[l]
                best_label = label

        value = 1 if (best_label == label) else 0
        return value, best_label

    def test(self, test, test_labels):
        """
        cest la methode qui va tester votre modele sur les donnees de test
        largument test est une matrice de type Numpy et de taille nxm, avec
        n : le nombre dexemple de test dans le dataset
        m : le mobre dattribus (le nombre de caracteristiques)

        test_labels : est une matrice numpy de taille nx1

        vous pouvez rajouter dautres arguments, il suffit juste de
        les expliquer en commentaire

        Faites le test sur les donnees de test, et afficher :
        - la matrice de confision (confusion matrix)
        - laccuracy
        - la precision (precision)
        - le rappel (recall)

        Bien entendu ces tests doivent etre faits sur les donnees de test seulement

        """
        predictions = []
        start = timeit.default_timer()
        for i in range(0, len(test)):
            prediction = self.predict(test[i], test_labels[i])
            predictions.append(prediction)
        self.confusion_matrix(predictions, test_labels)
        stop = timeit.default_timer()
        print "execution time : ",stop - start

    def bayes(self, train, train_labels):
        means = {}
        variances = {}
        #Nous permet d avoir un les type de labels
        group = set(train_labels)
        for label in group:
            data_by_label = train[train_labels == label]
            means[label] = data_by_label.mean(axis=0)
            variances[label] = data_by_label.var(axis=0)
        return means, variances

    #On y passe un vector exemple
    def probability(self, means, variances, group, exemple):
        probabilities = {}
        for label in group:
            probabilities[label] = 1
            for i in range(0, len(means[label])):
                part_1 = 1/ (np.sqrt(2 * np.pi) * variances[label][i])
                part_2 = (np.power((exemple[i] - means[label][i]), 2) * -1)/ (2 *np.power(variances[label][i],2))
                result =  part_1 * np.exp(part_2)

                probabilities[label] *= result

        return probabilities

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


    # Vous pouvez rajouter dautres methodes et fonctions,
    # il suffit juste de les commenter.
