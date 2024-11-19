import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
import time
import numpy as np
import matplotlib.pyplot as plt
import logging
"""
Random Forest Hyperparameter Optimization

Description:
    This file implements a technique called the 'Plumber Optimizer', which provides automated tuning for RandomForest models.
    The core idea divides the tuning process into two phases: a general and exploratory phase, followed by a second, more focused phase.
    It also implements an early stopping technique if a significant number of trials pass with no improvement.
    The optimizer allows the user to choose between F1 and accuracy as the metric score for optimization in classification tasks,
    while using MSE for regression tasks.

Usage:
    By providing the training data, specifying whether it's a classification or a regression task, and choosing if F1 should be used
    instead of accuracy, the Plumber Optimizer will automatically suggest a number of trials that balance performance and efficiency.
    The tuning process begins using the Optuna library. In addition to fully automated tuning, a visualization function is available
    that provides diagrams to help users visualize what happened behind the scenes.

Authors:
    Mohammed Aldahmani
    Ali Al-Ali
    Abdalla Alzaabi

Date:
    November 14, 2024

Dependencies:
    - optuna
    - numpy
    - pandas
    - sklearn
    - matplotlib
    - psutil
"""

class PlumberOptimizer:

    def __init__(self, X, y, classification=True, f1=False, verbose=0):
        """
        Constructor for PlumberOptimizer class.
        Args:
            X: Training data.
            y: Target vector.
            classification (bool): Set to True if the task is a classification task, False for regression.
            f1 (bool): Set to True to use F1 score as the evaluation metric for classification tasks.
            verbose (int): If set to 0, suppresses logging; otherwise, Optuna will print the results of each trial.
        
        Initializes the class by training a single RandomForest model to measure the time it takes, then automatically calculates
        a suitable number of trials based on this single model's training time and prints the result to the user.
        """
        self.X = X
        self.y = y
        self.best_param = None
        self.importance = None
        self.study = None
        self.improved_study = None
        self.best_value = float('-inf') 
        self.trials_since_last_improvement = 0
        self.classification = classification
        self.f1 = f1
        self.num_trials = self.compute_num_trials()
        print(f"Suggested number of trails is {self.num_trials} ")
        if verbose==0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)




    def compute_single_model_time(self):
        """
        Helper function to measure the duration to train a single RandomForest model.
        Returns:
            float: Time in seconds it takes to train the model.
        """
        if self.classification:
            model = RandomForestClassifier(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
        start_time = time.time()
        model.fit(self.X, self.y)
        return time.time() - start_time
    


    
    def compute_num_trials(self):
        """
        Helper function that uses a logistic function to compute the number of trials.
        Models that take less than a second to train will have a large number of trials (150),
        while models that take more than 30 seconds will have fewer trials (about 30).
        Returns:
            int: Computed number of trials.
        """
        single_model_time = self.compute_single_model_time()
        print("A single model takes ",single_model_time,"seconds to run")


        return int(30 + (220) / (1 + np.exp(0.2 * (single_model_time - 1))))
    


    def objective(self, trial):
        """
        An Optuna Objective function, this is the phase 1 function that tunes all 5 parameters each having a wide
        range of values. 
        """

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 30, step = 5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        if self.classification:
            model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
            if not self.f1:
                score = cross_val_score(model, self.X, self.y, cv=3, n_jobs=-1, scoring='accuracy').mean()
            else:
                scorer = make_scorer(f1_score, average='weighted')
                score = cross_val_score(model, self.X, self.y, cv=3, n_jobs=-1, scoring=scorer).mean()
        else:
            model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
            score = cross_val_score(model, self.X, self.y, cv=3, n_jobs=-1, scoring='neg_mean_squared_error').mean()

        return score




    def focused_objective(self, trial):
        """
        A second Optuna Objective function, this is the phase 2 function that tunes only the top 3 parm according to their 
        importance value determained by Optuna. It then assign a narower set of values for each parameter
        """

        # Sort parameter importances and select the top three
        top_params = dict(sorted(self.importance.items(), key=lambda item: item[1], reverse=True)[:3])

        params = {}
        for param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']:
            if param in top_params:
                base_value = self.best_param[param]
                # Adjust the range slightly based on the base_value
                if param == 'n_estimators':  # Specific handling for n_estimators
                    params[param] = trial.suggest_int(param, max(50, base_value - 50), base_value + 50, step=25)
                elif param == 'min_samples_split':  # Specific handling for min_samples_split
                    params[param] = trial.suggest_int(param, max(2, base_value - 10), base_value + 10)
                elif isinstance(base_value, int):
                    params[param] = trial.suggest_int(param, max(1, base_value - 5), base_value + 5)
                else:  # for categorical parameters like max_features
                    params[param] = trial.suggest_categorical(param, ['sqrt', 'log2'])
            else:
                params[param] = self.best_param[param]

        if self.classification:
            model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
            if not self.f1:
                score = cross_val_score(model, self.X, self.y, cv=3, n_jobs=-1, scoring='accuracy').mean()
            else:
                scorer = make_scorer(f1_score, average='weighted')
                score = cross_val_score(model, self.X, self.y, cv=3, n_jobs=-1, scoring=scorer).mean()
        else:
            model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
            score = cross_val_score(model, self.X, self.y, cv=3, n_jobs=-1, scoring='neg_mean_squared_error').mean()

        return score
    


    def early_stopping_callback(self, study, trial):
        """
        An optuna callback function, in our case it is used to stop early if no improvamant is made after 0.4*num_trials steps since the
        last best score is found or 30 for cases where 0.4*num_trials < 30
        """
        threshold = max(self.num_trials * 0.40, 30)
        if study.best_value == self.best_value:
            self.trials_since_last_improvement += 1
        else:
            self.best_value = study.best_value
            self.trials_since_last_improvement = 0

        if self.trials_since_last_improvement >= threshold:
            study.stop()  





    def optimize(self):
        """
        The main function that is to be called by the user, it begins phase1, obtain the importance, then start the improved stude
        phase 2. finaly it prints the best paramaters alongside the best score, 
        returns the best params dictionary
        Returns:
            dic: Best value for each parameter
        """

        ### Phase 1: self.study tunse all 5 paramaters, in a broad set of values

        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, self.num_trials, callbacks=[self.early_stopping_callback])
        self.best_param = self.study.best_params                                       # Obtain the best Params
        self.importance = optuna.importance.get_param_importances(self.study)          # Obtain the Importance of each param
        
        
        ### Phase 2: using only the top 3 parameters, begin a focused study
        self.best_value = float('-inf')                                             
        self.trials_since_last_improvement = 0
        self.improved_study = optuna.create_study(direction='maximize')
        self.improved_study.optimize(self.focused_objective, int(self.num_trials), callbacks=[self.early_stopping_callback])

        # Update best parameters with improvements from phase 2 if the best score from phase 2 is higher than phase 1 best score
        if self.improved_study.best_value > self.study.best_value:
            self.best_param.update(self.improved_study.best_params)


        print("Best Parameters:", self.best_param)
        print("Best Score:", self.improved_study.best_value)
        return  self.best_param
    



    def visualize(self):
        """
        A function that is to be called if the user wish to visualize phase 1 params importance (figure 1)
        as well as pararral coordinate of the optimization process for phase 2.
        Returns:
            tuple: Figures representing parameter importances and parallel coordinates of the optimization process.
        """
        print("fig1: After the first phase, This is how important each variable is")
        fig1 = optuna.visualization.plot_param_importances(self.study)
        print("")
        print("fig2: Visualizing the focused study plot parallel coordinate")
        fig2 = optuna.visualization.plot_parallel_coordinate(self.improved_study)
        return fig1, fig2
