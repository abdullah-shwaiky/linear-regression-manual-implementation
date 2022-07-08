#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
class LinearRegressor:
    """
    Ordinary least squares Linear Regression.
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.
    
    Parameters
    ----------
    
    solver : string, default="normal"
        Specifies the type of solver for the linear regression model.
        The available options are:
            - normal : using the normal equation.
            - bgd    : using batch gradient descent
            - mbsgd  : using minibatch (stochastic) gradient descent
            - sgd    : using stochastic gradient descent
        Each is related to its own hyperparameters, see more in Parameters
        
    alpha : float, default=1.0
        Specifies the ElasticNet regularization coefficient.
        Default value specifies 100% of Lasso regularization
        and 0% of Ridge regularization.
        
    lambda_ : float, default=0.0
        Specifies total regularization coefficient.
        Default value depicts using no regularization.
        
    minibatch_size : int, default=20
        Specifies the size of each minibatch in case of using mbsgd
        solver method.
        
    learning_schedule : bool, default=False
        If true, the learning rate ``eta`` will be updated
        regularly, as the iterative methods go through more
        iterations and epochs.
        
    epochs : int, default=300
        Specifies the number of epochs of training in
        iterative methods of solving.
        This parameter is ignored when solver is set to ``normal``.
    
    t0 : int, default=200
        Specifies, along with t1, the initial learning rate ``eta``.
        The ratio between t0 and t1 specifies the rate of change
        in the learning rate ``eta``.
        These parameters are ignored when ``learning_schedule`` is
        set to ``False``.
    
    t1 : int, default=1000
        See t0 above.
        
    Attributes
    ----------
    theta : array of (n_features,1)
        The array theta is the array of calculated weights that
        define the calculated fit for the model.
    """
    def __init__(self, solver = 'normal', alpha = 1.0,lambda_ = 0.0, minibatch_size = 20,
                 learning_schedule = False,epochs = 300,t0 = 200, t1 = 1000):
        self.solver = solver;
        self.alpha = alpha;
        self.lambda_ = lambda_;
        self.learning_schedule = learning_schedule;
        self.epochs = epochs;
        self.minibatch_size = minibatch_size
        self.t0 = t0;
        self.t1 = t1;
        self.eta = self.t0/self.t1
    def __str__(self):
        """
        String print function: prints summary of model
        when object is printed (method is called).
        """
        s_l=("Linear Regression Model\nAttributes:\n")
        s_theta=(f"Theta: {self.theta}\n")
        s_alpha=(f"Alpha: {self.alpha}\n")
        s_lambda=(f"Lambda: {self.lambda_}\n")
        s_ls=(f"Learning Schedule: {self.learning_schedule}\n")
        return s_l+s_theta+s_alpha+s_lambda+s_ls
    def preprocessing(self,X):
        """
        Concatenates a vector of ones to the given dataset
        as an added bias, used to calculate model bias.
        
        Parameters
        ----------
        
        X : array of shape (n_samples, n_features)
            Training data.
        """
        return np.c_[np.ones([X.shape[0],1]),X]
    def calculate_gradient(self,X,y,theta):
        """
        Calculates gradient for changing of theta.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training Data.
            
        y : array of shape (n_samples,1)
            Target values.
        """
        y_pred = X.dot(theta);
        y_pred = y_pred.reshape(len(X),1)
        return X.T.dot(y_pred-y)
    def calculate_regularization(self,theta,lambda_,alpha):
        """
        Calculates regularization value using ElasticNet.
        This method is ignored when lambda is set to zero.
        
        Parameters
        ----------
        theta: array of shape (n_features+1,1)
            Contains weights of model for prediction.
        
        alpha : float, default=1.0
            Specifies the ElasticNet regularization coefficient.
            Default value specifies 100% of Lasso regularization
            and 0% of Ridge regularization.

        lambda_ : float, default=0.0
            Specifies total regularization coefficient.
            Default value depicts using no regularization.
        """
        return (lambda_) * (alpha * np.sign(theta) + ((1-alpha))*theta)
    def update_eta(self,t):
        """
        Updates the value of learning rate.
        
        Parameters
        ----------
        t0 : int, default=200
            Specifies, along with t1, the initial learning rate ``eta``.
            The ratio between t0 and t1 specifies the rate of change
            in the learning rate ``eta``.
            These parameters are ignored when ``learning_schedule`` is
            set to ``False``.
    
        t1 : int, default=1000
            See t0 above.
        
        t : int, default=0
            Resembles the number of current iterations.
            The more iterations done, the higher the value
            of t becomes, the slower the learning rate becomes.
        """
        return self.t0 / (t + self.t1) if self.learning_schedule else self.t0/self.t1
    
    
    def normal_equation(self, X, y):
        """
        Calculates weight through Linear Algebraic equation called
        the normal equation. It calculates the optimal weights for the
        problem without any iterations. It is the default solver
        of this model.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training Data.
            
        y : array of shape (n_samples,1)
            Target values.
        """
        self.theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),y)
    def bgd(self,X,y,epochs):
        """
        Calculates weight through Batch Gradient Descent, an
        iterative method that calculates gradients and updates
        weights in batches.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training Data.
            
        y : array of shape (n_samples,1)
            Target values.
        epochs : int, default=300
            Number of iterations over the entire data set.
        """
        m = len(X)
        for epoch in range(epochs):
            gradient = self.calculate_gradient(X,y,self.theta);
            regularization = self.calculate_regularization(self.theta,self.lambda_,self.alpha)
            self.theta = self.theta - (self.eta/m)*(gradient + regularization)
    def mbsgd(self,X,y,epochs,minibatch_size):
        """
        Calculates weight through MiniBatch Gradient Descent, an
        iterative method that calculates gradients and updates
        weights in batches of smaller size than the entire dataset.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training Data.
            
        y : array of shape (n_samples,1)
            Target values.
        epochs : int, default=300
            Number of iterations over the entire data set.
        """
        t = 0;
        for epoch in range(epochs):
            m = len(X)
            shuffled_indices = np.random.permutation(m)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            for i in range(0, m, minibatch_size):
                t += 1
                xi = X_shuffled[i:i+minibatch_size]
                yi = y_shuffled[i:i+minibatch_size]
                gradients = 2 * self.calculate_gradient(xi,yi,self.theta)
                regularization = self.calculate_regularization(self.theta,self.lambda_,self.alpha)
                self.eta = self.update_eta(t)
                self.theta = self.theta - (self.eta/minibatch_size) * (gradients + regularization)
    def sgd(self,X,y,epochs):
        """
        Calculates weight through MiniBatch Gradient Descent, an
        iterative method that calculates gradients and updates
        weights with each instance of the dataset.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training Data.
            
        y : array of shape (n_samples,1)
            Target values.
        epochs : int, default=300
            Number of iterations over the entire data set.
        """
        m = len(X);
        t = 0;
        for epoch in range(epochs):
            for i in range(m):   
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                regularization = self.calculate_regularization(self.theta,self.lambda_,self.alpha)
                gradients = 2 * self.calculate_gradient(xi,yi,self.theta)
                self.eta = self.update_eta(t)
                self.theta = self.theta - self.eta * (gradients + regularization/m)
    def fit(self, X,y):
        """
        Fit linear model.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training Data.
            
        y : array of shape (n_samples,1)
            Target values.
        """
        X = self.preprocessing(X);
        self.theta = np.random.randn(X.shape[1],1)
        if self.solver == 'normal':
            self.normal_equation(X,y);
        elif self.solver == 'bgd':
            self.bgd(X,y,self.epochs);
        elif self.solver == 'mbsgd':
            self.mbsgd(X,y,self.epochs,self.minibatch_size);
        else:
            self.sgd(X,y,self.epochs)
    def predict(self,X):
        """
        Return model predictions based on the calculated value of theta.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data points to predict value of.
        """
        X = self.preprocessing(X)
        return X.dot(self.theta)