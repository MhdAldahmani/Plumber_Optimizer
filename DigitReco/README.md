Digit Recongizer Dataset - Performance Comparison of Hyperparameter
Optimization Techniques

The following notebook considers various hyperparameter optimization techniques 
including our custom Plumber Optimizer on the Digit Recognizer dataset for the 
task of balanced multi-class classification using the metric of accuracy. The 
dataset comprises images of handwritten digits from 0 to 9 and are represented by 
pixel intensity values whose target is the label of the associated digit. Next is 
normalization of the pixel values, followed by the application of PCA for dimensionality reduction. 
Then, performances are compared among the conventional methods of Grid Search, Random Search, 
and Bayesian Optimization with the Plumber Optimizer.
Collected metrics include execution time, memory usage, CPU usage, and validation accuracy. 
All while performing the optimization effectively, the Plumber Optimizer was competitive in its accuracy rate, 
with an average achieved value of 94.71%, but with lower execution times compared to both Grid Search and BayesSearchCV. 
That is a good job done by the Plumber Optimizer in hyperparameter tuning, finding that sweet spot of accuracy and efficiency, 
along with resource usage. All extended results, including the rest of the metrics and a comparative analysis, are provided in the notebook.
