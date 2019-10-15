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
from sklearn.multiclass import OneVsRestClassifier # used for categories issus in the ROC curve (give TP, FP for each category)
from scipy import interp
from itertools import cycle
from automl.gs_params import params # our parameters library
import warnings # scikit-learn can produce updates warnings
warnings.filterwarnings('ignore') # ignore warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

class prediction(object): # preidiction class
    def __init__(self, dataset, target, type_of_estimator=None): # constructor that will determine if our dataset is a regressor or a classifier
        if (type_of_estimator == None):
            if(dataset[target].nunique() > 10):
                self.type = "continuous" # regressor
            else:
                self.type = "classifier" # classifier
        else:
            if (type_of_estimator == "continuous" or type_of_estimator =="classifier"):
                self.type = type_of_estimator
            else:
                print('Invalid value for "type_of_estimator". Please pass in either "continuous" or "classifier". You passed in: ' + type_of_estimator) 
                # return a warning if it doesn't recognize the type
        self.dataset = dataset # determine the dataset
        self.result = {} # result
        self.reducedDataset = None
        self.withoutOutliers = None
        self.clean() # clean and change the dataset
        self.target = target # determine the value to predict
        self.grid_S = params() #import gs_params and parameters initialization
        self.Outliers() #remove outliers and put the modified dataset in withoutOutliers
        self.Y = dataset[target].values # value to predict
        self.X = dataset.loc[:, dataset.columns != target].values # others values
        self.params = params()
        self.reduction() # reduct the dataset dimension and put the modified dataset in reducedDataset
        self.train(self.X,self.Y) # dataset training
        self.train(self.reducedDataset,self.Y,reduction=True) # reduced dataset training

    def reduction(self): # function handling the dimension reduction
        numberOfComponent = len(
            self.dataset.loc[:, self.dataset.columns != self.target].columns)
        total_variance_explained = 0 # variance total to explain to realize the dimension reduction (initialization)
        X_temp = None
        dimension = 0 # we start with a dimension of zero, while we don't have a total expressed of 90%, we add a dimension until reaching it
        std_scale = preprocessing.StandardScaler().fit(self.X) # to standardize: calculate the mean and the standard deviation to know the operation
        X_scaled = std_scale.transform(self.X)
        V = np.sum(np.var(X_scaled, axis=0)) # we calculate the sum of our standardized x variance
        while(total_variance_explained < 90 and dimension < numberOfComponent): 
            dimension = dimension + 1
            pca = decomposition.PCA(n_components=dimension)
            pca.fit(X_scaled)
            X_projected = pca.transform(X_scaled)
            explained_variance = np.var(X_projected, axis=0)
            total_variance_explained = np.sum(explained_variance)/V
            X_temp = pca.transform(X_scaled)
        self.reducedDataset = X_temp # then store the value

    def clean(self): # function handling the dataset cleaning
        number_of_Nan = self.dataset.isnull().sum().sum() # recover NaN number
        pourcentage_of_Nan = (number_of_Nan/self.dataset.count().sum())*100 # change it to a percentage rate
        print('NaNs represent ' + str(pourcentage_of_Nan) + ' pourcentage of dataset') 
        for column in self.dataset.columns.values:
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
                    self.dataset[column].fillna(method='bfill', inplace=True) # take the previous value
                    self.dataset[column].fillna(method='ffill', inplace=True) # take the following value
                    
    def train(self, X, Y, reduction=False): # function handling the dataset training
        models = {} # intitialization of models list
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42) #split of 20% testing and 80% training
        # for regressor
        if (self.type == "continuous"): 
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
        # for classifier
        elif (self.type == "classifier"):
            perf = self.modelLinearSVC(X_train, y_train, X_test, y_test) 
            # look for bests parameters then use a K-fold Cross Validation and return a dictionnary with: the model, the accuracy, the RMSE and the name
            models.update({'SVC': perf})
            perf = self.modelRandomForestClassifier(X_train, y_train, X_test, y_test)
            models.update({'RandomForestClassifier': perf})
            perf = self.modelLogisticRegressor(
                X_train, y_train, X_test, y_test)
            models.update({'LogisticRegressor': perf})
        if (self.type == "classifier"):
            # we sorts models based on their accuracy generated by the cross validation
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
        # for a regressor
        if (self.type == "continuous"):
            # we sorts models based on their accuracy generated by the cross validation regardless of the RMSE
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
        # if a reduction was made, it returns the three bests models with reduction and compare them to the three bests without reduction
        if(reduction):
            final_model1.update({'Dimension Reduction' : True})
            final_model2.update({'Dimension Reduction' : True})
            final_model3.update({'Dimension Reduction': True})
            self.result.update({'Fourth' : final_model1,'Fifth' : final_model2, 'Sixth':final_model3})
            if(self.type == "continuous"):
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
                print('first model:') # print three bests models and save them in self.result
                print(f1)
                print('second model:')
                print(f2)
                print('third model:')
                print(f3)
                print('use .result to access models')
            # for a classifier
            else:
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
                    # we implemented the ROC curve by hand for this one since the RFC structure was different than the other classifiers
                    self.rocCurve(f1['model'],final_model1['name'],
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
        
        
    def evaluate(self, model, X_test, y_test): # function handling our selected model evaluation
        results = cross_val_score(
            model, X_test, y_test, cv=KFold(n_splits=10), n_jobs=1) # we use K=10 (time related)
        result = np.mean(list(filter(lambda x: x > 0, results)))
        if (self.type=="continuous"):
            mse_test = mean_squared_error(y_test, model.predict(X_test))
            result = {'accurracy': result, 'rmse': np.sqrt(mse_test)}
        return result

    # For the Lasso and the Logistic Regression we entered parameters by hand

    def modelLasso(self, X_train, y_train, X_test, y_test): #Lasso Regression function
        lasso = Lasso(random_state=0, max_iter=10000)
        alphas = np.logspace(-4, -0.5, 30)
        tuned_parameters = [{'alpha': alphas}]
        n_folds = 5
        clf = GridSearchCV(lasso, tuned_parameters,
                           cv=n_folds, refit=False, return_train_score=True)
        grid_result = clf.fit(X_train, y_train)
        best_params = grid_result.best_params_
        bestmodel = Lasso(random_state=0, max_iter=10000,
                          alpha=best_params['alpha'])
        bestmodel.fit(X_train, y_train)
        result = self.evaluate(bestmodel, X_test, y_test)
        performance = {'model': bestmodel, 'accurracy': result , 'name': 'lasso'}
        return performance

    def modelLogisticRegressor(self, X_train, y_train, X_test, y_test): # Logistic Regression function
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

    def rocCurve(self, model,name, X_train, y_train, X_test, y_test): # function returning the ROC curve
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

    def Outliers(self): # function handling outliers (mean = 50%)
        Q1 = self.dataset.quantile(0.25)  # first quartile = 25%
        Q3 = self.dataset.quantile(0.75) # second quartile = 75%
        IQR = Q3 - Q1 # IQR score: difference between both quartiles
                      # we take 1.5 * IQR and if it's outside of Q1 - 1.5 * IQR or Q3 + 1.5 * IQR, it's considered as an outlier
        self.withoutOutliers = self.dataset[~((self.dataset < (
            Q1 - 1.5 * IQR)) | (self.dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
        pourcentage_of_outliers = (self.withoutOutliers.count()[
                                   self.withoutOutliers.columns[0]]/self.dataset.count()[self.dataset.columns[0]])*100
        print('there is ' + str(pourcentage_of_outliers) +
              ' pourcetage of rows with outliers')

    # For every model here:
        # we put the model in lr
        # we apply GridSearch => param_grid: parameters grid to test in gs_params
        # we run it
        # we return the best estimator found (best_grid)
        # we evaluate the model
        # we return it as well as the accuracy and the model name
    
    def modelRandomForestClassifier(self, X_train, y_train, X_test, y_test):  # Random Forest Classifier function
        lr = RandomForestClassifier()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['RandomForestClassifier'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy, 'name' : 'RandomForestClassifier'}

    def modelLinearRegression(self, X_train, y_train, X_test, y_test): # Linear Regression function
        lr = LinearRegression()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearRegression'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy, 'name':LinearRegression}

    def modelAdaBoostRegressor(self, X_train, y_train, X_test, y_test): # Adaboost Regressor function
        lr = AdaBoostRegressor()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['AdaBoostRegressor'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'AdaBoostRegressor'}

    def modelElasticNet(self, X_train, y_train, X_test, y_test): # Elastic Net function
        lr = ElasticNet()
        grid = GridSearchCV(
            estimator=lr, param_grid= self.params['ElasticNet'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'ElasticNet'}

    def modelLinearSVR(self, X_train, y_train, X_test, y_test): # Linear SVR function
        lr = LinearSVR()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearSVR'], cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'LinearSVR'}

    def modelLinearSVC(self, X_train, y_train, X_test, y_test): # Linear SVC function
        lr = LinearSVC()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearSVC'], cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'LinearSVC'}
    
    def modelRandomForestRegressor(self, X_train, y_train, X_test, y_test):  # Random Forest Regressor function
        lr = RandomForestRegressor()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['RandomForestRegressor'], cv=3, n_jobs=1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'RandomForestRegressor'}
