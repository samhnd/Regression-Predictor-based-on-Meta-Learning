import numpy as np
import pandas as pd
from sklearn import decomposition, preprocessing
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier #utilisé pour problèmes des catégories dans ROC curve, donne Vrai Positifs, FP pour chaque catégorie
from scipy import interp
from itertools import cycle
from automl.gs_params import params #notre librairie de paramètres
import warnings #scikit-learn peut produire des warnings de màj
warnings.filterwarnings('ignore') #ignore les warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

class prediction(object): #init = constructeur et commencer par self pour dire qu'elle appartient à cette classe
    def __init__(self, dataset, target, type_of_estimator=None): #constructeur qui va determiner si notre dataset est de type regressor ou classifier
        if (type_of_estimator == None): #si il n'a pas de type déjà donné en paramètre
            if(dataset[target].nunique() > 10): #et si le nombre de valeur du predicteur est superieur à 10
                self.type = "continuous" #alors c'est un regressor
            else:
                self.type = "classifier" #sinon c'est un classifier
        else:
            if (type_of_estimator == "continuous" or type_of_estimator =="classifier"):
                self.type = type_of_estimator #s'il a deja un type alors il prend son type
            else:
                print('Invalid value for "type_of_estimator". Please pass in either "continuous" or "classifier". You passed in: ' + type_of_estimator) 
                #sinon un warning est renvoyé car il ne reconnait pas le type
        self.dataset = dataset #determine le dataset
        self.result = {} #résultat
        self.reducedDataset = None
        self.withoutOutliers = None
        self.clean() #va nettoyer le dataset et changer le dataset
        self.target = target #determine la valeur a predire
        self.grid_S = params() #importé de gs_params et initialise les paramètres
        self.Outliers() #va enlever les outliers et mettre le dataset modifié dans withoutOutliers
        self.Y = dataset[target].values #valeur a predire
        self.X = dataset.loc[:, dataset.columns != target].values #autre valeurs
        self.params = params() #identique
        self.reduction() #va reduire la dimension et mettre le dataset modifié dans reducedDataset
        self.train(self.X,self.Y) #train le dataset
        self.train(self.reducedDataset,self.Y,reduction=True) #train le dataset réduit

    def reduction(self): #fonction gérant la reduction de dimension
        numberOfComponent = len(
            self.dataset.loc[:, self.dataset.columns != self.target].columns) #nombre de colonne dans x
        total_variance_explained = 0 #total de variance a expliquer pour realiser une reduction de dimension (initialisation)
        X_temp = None #création d'une variable temporaire
        dimension = 0 #on commence à une dimension (nb de features) de 1, tant qu'on a pas une variance totale exprimée de 90%, on rajoute une dimension jusqu'à l'atteindre 
        std_scale = preprocessing.StandardScaler().fit(self.X) #permet de standardiser : va calculer la moyenne et l'ecart type afin de connaitre l'operation.
        X_scaled = std_scale.transform(self.X) #va calculer l'opération qu'on fait et l'applique en changeant ainsi l'échelle.
        V = np.sum(np.var(X_scaled, axis=0)) #on calcule la somme de la variance de notre x standardisé
        while(total_variance_explained < 90 and dimension < numberOfComponent): 
            #tant que le total de variance a expliquer est sinferieur à 90 et que la dimension est inférieur au nombre de composant
            dimension = dimension + 1  #on incrémente la dimension
            pca = decomposition.PCA(n_components=dimension) #recalcule la variance ainsi de suite
            pca.fit(X_scaled)
            X_projected = pca.transform(X_scaled)
            explained_variance = np.var(X_projected, axis=0)
            total_variance_explained = np.sum(explained_variance)/V
            X_temp = pca.transform(X_scaled)
        self.reducedDataset = X_temp #alors ca stocke la valeur

    def clean(self): #fonction permettant de nettoyer le dataset
        number_of_Nan = self.dataset.isnull().sum().sum() #on récupère le nombre de NaN
        pourcentage_of_Nan = (number_of_Nan/self.dataset.count().sum())*100 #on le tranforme en pourcentage
        print('NaNs represent ' + str(pourcentage_of_Nan) + ' pourcentage of dataset') #on l'affiche
        for column in self.dataset.columns.values: #pour chaque colonne dans le dataset
            # Replace NaNs with the median or mode of the column depending on the column type
            try:
                self.dataset[column].fillna(
                    self.dataset[column].median(), inplace=True)
            except TypeError:
                most_frequent = self.dataset[column].mode()
                # If the mode can't be computed, use the nearest valid value
                if len(most_frequent) > 0:
                    self.dataset[column].fillna(
                        self.dataset[column].mode()[0], inplace=True)
                else:
                    self.dataset[column].fillna(method='bfill', inplace=True) #prend la valeur précedente
                    self.dataset[column].fillna(method='ffill', inplace=True) #prend la valeur suivante

    def train(self, X, Y, reduction=False): #fonction entrainant notre modele
        models = {} #initialise la liste des modèles (dictionnaire)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42) #split de 20% test et 80% train
        if (self.type == "continuous"): #si notre valeur a predire est de type regressor alors utiliser un des modeles suivants
            perf = self.modelLasso(X_train, y_train, X_test, y_test) 
            #va chercher les meilleurs paramètres puis faire le K-fold cross validation et retourner un dictionnaire avec: modèle, accuracy et nom
            models.update({'Lasso': perf}) #on run tous les modèle et on les met dans models
            perf = self.modelRandomForestRegressor(X_train, y_train, X_test, y_test)
            models.update({'RandomForestRegressor': perf})
            perf = self.modelElasticNet(X_train, y_train, X_test, y_test)
            models.update({'ElasticNet': perf})
            perf = self.modelLinearSVR(X_train, y_train, X_test, y_test)
            models.update({'LinearSVR': perf})
            perf = self.modelLinearRegression(X_train, y_train, X_test, y_test)
            models.update({'LinearRegression': perf})
            perf = self.modelAdaBoostRegressor(X_train, y_train, X_test, y_test)
            models.update({'AdaBoostRegressor': perf})
        elif (self.type == "classifier"): #si notre valeur a predire est de type classifier alors utiliser un des modeles suivants
            perf = self.modelLinearSVC(X_train, y_train, X_test, y_test) 
            #va chercher les meilleurs paramètres puis faire le K-fold cross validation et retourner un dictionnaire avec: modèle, accuracy, RMSE et nom
            models.update({'SVC': perf})
            perf = self.modelRandomForestClassifier(X_train, y_train, X_test, y_test)
            models.update({'RandomForestClassifier': perf})
            perf = self.modelLogisticRegressor(
                X_train, y_train, X_test, y_test)
            models.update({'LogisticRegressor': perf})
        #à la fin on aura tous les modèles avec leur accuracy dans models
        if (self.type == "classifier"): #si c'est un classifier
            #on trie les modèles en comparant leur accuracy générée par le K-fold
            temp = 0
            for key in models:
                if models[key]['accurracy'] > temp:
                    temp = models[key]['accurracy']
                    final_model1 = models[key]
            temp = 0
            for key in models:
                if models[key]['accurracy'] > temp and models[key]!=final_model1:
                    temp = models[key]['accurracy']
                    final_model2 = models[key]
            temp = 0
            for key in models:
                if models[key]['accurracy'] > temp and models[key]!=final_model1 and models[key]!=final_model2:
                    temp = models[key]['accurracy']
                    final_model3 = models[key]
        if (self.type == "continuous"): #si c'est un regressor
            #on trie les modèles en comparant leur accuracy générée par le K-fold sans prendre en compte le RMSE
            temp = 0
            for key in models:
                if models[key]['accurracy']['accurracy'] > temp:
                    temp = models[key]['accurracy']['accurracy']
                    final_model1 = models[key]
            temp = 0
            for key in models:
                if models[key]['accurracy']['accurracy'] > temp and models[key]!=final_model1:
                    temp = models[key]['accurracy']['accurracy']
                    final_model2 = models[key]
            temp = 0
            for key in models:
                if models[key]['accurracy']['accurracy'] > temp and models[key]!=final_model1 and models[key]!=final_model2:
                    temp = models[key]['accurracy']['accurracy']
                    final_model3 = models[key]
        if(reduction): #si on a effectué une réduction alors ça va chercher les trois meilleurs modèles avec la réduction et ça les compare avec ceux sans réduction
            final_model1.update({'Dimension Reduction' : True}) #on rajoute dans notre dictionnaire TRUE
            final_model2.update({'Dimension Reduction' : True})
            final_model3.update({'Dimension Reduction': True})
            self.result.update({'Fourth' : final_model1,'Fifth' : final_model2, 'Sixth':final_model3}) #on ajoute les trois meilleurs modèles de la réduction
            if(self.type == "continuous"): #si continuous : va comparer les modèles avec réduction et les modèles sans réduction
                temp = 0
                for key in self.result:
                    if self.result[key]['accurracy']['accurracy'] > temp:
                        temp = self.result[key]['accurracy']['accurracy']
                        f1 = self.result[key]
                temp = 0
                for key in self.result:
                    if self.result[key]['accurracy']['accurracy'] > temp and self.result[key]!=f1:
                        temp = self.result[key]['accurracy']['accurracy']
                        f2 = self.result[key]
                temp = 0
                for key in self.result:
                    if self.result[key]['accurracy']['accurracy'] > temp and self.result[key]!=f1 and self.result[key]!=f2:
                        temp = self.result[key]['accurracy']['accurracy']
                        f3 = self.result[key]
                print('first model:') #et print les trois meilleurs modèles et les enregistre dans self.result
                print(f1)
                print('second model:')
                print(f2)
                print('third model:')
                print(f3)
                print('use .result to access models')

            else: #si classifier
                temp = 0
                for key in self.result:
                    if self.result[key]['accurracy'] > temp:
                        temp = self.result[key]['accurracy']
                        f1 = self.result[key]
                temp = 0
                for key in self.result:
                    if self.result[key]['accurracy'] > temp and self.result[key]!=f1:
                        temp = self.result[key]['accurracy']
                        f2 = self.result[key]
                temp = 0
                for key in self.result:
                    if self.result[key]['accurracy'] > temp and self.result[key]!=f1 and self.result[key]!=f2:
                        temp = self.result[key]['accurracy']
                        f3 = self.result[key]
                print('first model:')
                print(f1)
                print('second model:')
                print(f2)
                print('third model:')
                print(f3)
                print('use .result to access models')
                if (f1['name'] != 'RandomForestClassifier'): 
                    #ROC curve buguait (la structure de RFC n'est pas la même que les autre classifier) donc on a implémenté à la main
                    self.rocCurve(f1['model'],final_model1['name'], #implémente rocCurve (fonction plus bas)
                            X_train, y_train, X_test, y_test)
                if (f2['name'] != 'RandomForestClassifier'):
                    self.rocCurve(f2['model'],f2['name'],
                            X_train, y_train, X_test, y_test)
                if (f3['name'] != 'RandomForestClassifier'):
                    self.rocCurve(f3['model'],f3['name'],
                            X_train, y_train, X_test, y_test)
            self.result = {'First' : f1 , 'Second': f2, 'Third' : f3}
        else:
            self.result = {'First' : final_model1 , 'Second': final_model2, 'Third' : final_model3}
        
        
    def evaluate(self, model, X_test, y_test): #fonction evaluant notre modèle
        #va mélanger le train en K-parties de manière aléatoire, puis prend une partie aléatoire et va test dessus : va le faire 10 fois et prendre la moyenne des 10 tests
        results = cross_val_score(
            model, X_test, y_test, cv=KFold(n_splits=10), n_jobs=1) #K=10 (c'est bon, plus on rajoute des KFold plus ça prend du temps)
        result = np.mean(list(filter(lambda x: x > 0, results)))
        if (self.type=="continuous"):
            mse_test = mean_squared_error(y_test, model.predict(X_test))
            result = {'accurracy': result, 'rmse': np.sqrt(mse_test)}
        return result

    #Pour Lasso et LogisticRegression:
    #on a rentré les paramètres à la main

    def modelLasso(self, X_train, y_train, X_test, y_test): #fonction de la regression de Lasso
        lasso = Lasso(random_state=0, max_iter=10000)
        alphas = np.logspace(-4, -0.5, 30) #permet de générer un intervalle (liste pour le GridSearch)
        tuned_parameters = [{'alpha': alphas}]
        n_folds = 5
        clf = GridSearchCV(lasso, tuned_parameters,
                           cv=n_folds, refit=False, return_train_score=True) #on applique un GridSearch
        grid_result = clf.fit(X_train, y_train)
        best_params = grid_result.best_params_
        bestmodel = Lasso(random_state=0, max_iter=10000,
                          alpha=best_params['alpha'])
        bestmodel.fit(X_train, y_train)
        result = self.evaluate(bestmodel, X_test, y_test)
        performance = {'model': bestmodel, 'accurracy': result , 'name': 'lasso'}
        return performance

    def modelLogisticRegressor(self, X_train, y_train, X_test, y_test): #fonction de la regression Logistique
        dual = [True, False]
        max_iter = [100, 110, 120, 130, 140]
        param_grid = dict(dual=dual, max_iter=max_iter)
        lr = LogisticRegression(penalty='l2',solver='liblinear')
        grid = GridSearchCV(
            estimator=lr, param_grid=param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name': 'LogisticRegressor'}

    def rocCurve(self, model,name, X_train, y_train, X_test, y_test): 
        #fonction renvoyant la ROC curve (FP, TP, ... donc valeur de sortie 1 ou 0) en cherchant les catégories et s'il y en a plusieurs, va imprimer ROC Curve pour chaque catégorie
        y_train1 = label_binarize(
            y_train, list(range(0, self.dataset[self.target].nunique())))
        y_test1 = label_binarize(
            y_test, list(range(0, self.dataset[self.target].nunique())))
        n_classes = y_train1.shape[1]
        classifier = OneVsRestClassifier(model)
        y_score = classifier.fit(
            X_train, y_train1).decision_function(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if (n_classes == 1):
            fpr[0], tpr[0], _ = roc_curve(y_test1[:, 0], y_score)
            roc_auc[0] = auc(fpr[0], tpr[0])
        else:
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test1.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(name)
        plt.legend(loc="lower right")
        plt.show()

    def Outliers(self): #fonction gérant les outliers (moyenne = 50%)
        Q1 = self.dataset.quantile(0.25)  #premier quartile = 25%
        Q3 = self.dataset.quantile(0.75) #deuxième quartile = 75%
        IQR = Q3 - Q1 #IQR score = différence entre les deux quartiles
        #on prend 1,5 * IQR et s'ils sont en dehors de Q1 - 1.5 * IQR ou Q3 + 1.5 * IQR => considéré comme outliers
        self.withoutOutliers = self.dataset[~((self.dataset < (
            Q1 - 1.5 * IQR)) | (self.dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
        pourcentage_of_outliers = (self.withoutOutliers.count()[
                                   self.withoutOutliers.columns[0]]/self.dataset.count()[self.dataset.columns[0]])*100 #renvoie le pourcentage
        print('there is ' + str(pourcentage_of_outliers) +
              ' pourcetage of rows with outliers') #print le résultat

    #Pour chaque modèle ici:
    #on met le modèle dans lr
    #on applique GridSearch => param_grid : grille des paramètres à tester dans gs_params
    #run le GridSearc
    #puis renvoie le meilleur estimateur trouvé (best_grid)
    #puis evalue le modèle (fonction evaluate plus haut)
    #retourne le modèle, l'accuracy et le nom du modèle
    
    def modelRandomForestClassifier(self, X_train, y_train, X_test, y_test):  #fonction du Random Forest Classifier
        lr = RandomForestClassifier()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['RandomForestClassifier'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy, 'name' : 'RandomForestClassifier'}

    def modelLinearRegression(self, X_train, y_train, X_test, y_test): #fonction de la regression Linéaire
        lr = LinearRegression()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearRegression'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy, 'name':LinearRegression}

    def modelAdaBoostRegressor(self, X_train, y_train, X_test, y_test): #fonction de Adaboost Regressor
        lr = AdaBoostRegressor()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['AdaBoostRegressor'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'AdaBoostRegressor'}

    def modelElasticNet(self, X_train, y_train, X_test, y_test): #fonction de l'Elastic Net
        lr = ElasticNet()
        grid = GridSearchCV(
            estimator=lr, param_grid= self.params['ElasticNet'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'ElasticNet'}

    def modelLinearSVR(self, X_train, y_train, X_test, y_test): #fonction du SVR Linéaire
        lr = LinearSVR()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearSVR'], cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'LinearSVR'}

    def modelLinearSVC(self, X_train, y_train, X_test, y_test): #fonction du SVC Linéaire
        lr = LinearSVC()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearSVC'], cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'LinearSVC'}
    
    def modelRandomForestRegressor(self, X_train, y_train, X_test, y_test):  #fonction du Random Forest Regressor
        lr = RandomForestRegressor()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['RandomForestRegressor'], cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'RandomForestRegressor'}
