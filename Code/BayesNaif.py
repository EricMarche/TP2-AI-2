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
import math

class BayesNaif:

    def __init__(self, **kwargs):
        """
        cest un Initializer.
        Vous pouvez passer dautre parametres au besoin,
        cest a vous dutiliser vos propres notations
        """
        self.group = {}
        self.train = []
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
        self.train = train
        self.train_labels = train_labels
        self.means, self.variances, count = self.b(train, train_labels)


    def predict(self, exemple, label):
        """
        Predire la classe dun exemple donne en entree
        exemple est de taille 1xm

        si la valeur retournee est la meme que la veleur dans label
        alors lexemple est bien classifie, si non cest une missclassification

        """

        probabilites = self.probability(means, variances, group, exemple)
        best_label = -1
        best_prob = -1
        for label in probabilites:
            if probabilites[label] > best_prob:
                best_prob = probabilites[label]
                best_label = label
        return best_label

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

    # def bayes(self, train, train_labels):
    #     means = {}
    #     variances = {}
    #     count = {}
    #
    #     for x in range(0, len(train)):
    #         label = train_labels[x]
    #         for y in range(0, len(train[x])):
    #             if train_labels[x] not in means.keys():
    #                 #On creer nos liste en consequence
    #                 means[label] = [0] * len(train[x])
    #                 variances[label] = [0] * len(train[x])
    #                 count[label] = 0
    #             means[label][y] =  round(means[label][y] + train[x][y], 3)
    #             print "means[label][y]: ", means[label][y]
    #             variances[label][y] = round(variances[label][y] + (train[x][y] ** 2), 3)
    #             print "variances[label][y] : ", variances[label][y]
    #         count[label] += 1
    #
    #     # print "Means: ", means
    #     # print "Variances: ", variances
    #
    #     for x in range(0, len(means)):
    #         for y in range(0, len(means[x]) -2 ):
    #             #on arrondi a 2
    #             means[x][y] = round(means[x][y] / (count[x]), 3)
    #             # On calcule la variance sur un echantillon vu quon a pas toutes les donnees
    #             # print "count[x] : ", count[x]
    #             variances[x][y] = (variances[x][y] / (count[x]) - 0.0) - (means[x][y] ** 2)
    #     print "Means: ", means
    #     print "Variances: ", variances

    def b(self, train, train_labels):
        means = {}
        variances = {}
        count = {}
        #Nous permet d avoir un les type de labels
        print "*********************"
        group = set(train_labels)
        for l in group:
            data_by_label = train[train_labels == l]
            means[l] = data_by_label.mean(axis=0)
            variances[l] = data_by_label.var(axis=0)
            count[l] = len(train_labels[train_labels == l]) / len(train_labels)
        print "means: ", means
        print "var: ", variances
        print "count", count
        return means, variances, count
        # variances = {0:[0.5], 1:[5.0]}
        # means = {0:[1], 1:[20]}

        test2 = [5.855, 176.25, 11.25]
        self.probability(means, variances, group, test2)
        summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
        inputVector = [1.1, '?']
        probabilities = self.calculateClassProbabilities(summaries, inputVector)
        print('Probabilities for each class: {0}').format(probabilities)

    #On y passe un vector
    def probability(self, means, variances, group, exemple):
        probabilities = {}
        for label in group:
            print "len(means): ", len(means[label])
            probabilities[label] = 1
            for i in range(0, len(means[label])):
                print "means[label][i]: ", means[label][i]
                print "variances[label][i]: ", variances[label][i]
                print "exemple[i]: ", exemple[i]
                part_1 = 1/ (np.sqrt(2 * np.pi) * variances[label][i])
                part_2 = (np.power((exemple[i] - means[label][i]), 2) * -1)/ (2 *np.power(variances[label][i],2))
                result =  part_1 * np.exp(part_2)

                probabilities[label] *= result

        print "result : ", probabilities
        evidence = 0
        posterior = []
        # for i in range(0, len(self.labels)):

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    def calculateClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.iteritems():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                print "mean: ", mean
                print "stdev: ", stdev
                x = inputVector[i]
                print "x: ", x
                probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
        return probabilities



    # Vous pouvez rajouter dautres methodes et fonctions,
    # il suffit juste de les commenter.
