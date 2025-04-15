import torch
import math
class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        return X @ self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores= self.score(X)
        return (scores > 0).float()

class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
        y_ = 2*y - 1
        """
        
        y_ = 2 * y - 1 
        scores = self.score(X)
        misclassified = (scores * y_ <= 0).float()
        return misclassified.mean()

    def grad(self, X, y):

        y_ = 2 * y - 1  #convert label to {-1, 1}
        s = X @ self.w  #compute score s_i = <w, x_i>
        mask = (s * y_ < 0).float()
        grad = -1 * mask[:, None] * (y_[:, None] * X)  # Apply update rule
        alpha =.01
        return alpha * grad.mean(axis=0)

        

class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        """
        self.model.loss(X, y)

        #compute gradient and update weights
        grad = self.model.grad(X, y)  # grad() gives the weight update
        self.model.w -= grad  # Update weights

class LogisticRegression(LinearModel):

    def loss(self, X, y):
        """
        Compute the missclassification rate at the current stage in the model by idnetifying which points are in the wrong section

        Arguments: 
        X: torch.tensor: an n x p feature matrix where n is the number of data points and p the number of features
        y: torch.tensor: the target vector of size n, where the values are 1 or 0 (binary classified)
        
        Returns:
        loss: the missclassification rate
        """
        if self.w is None:
            self.score(X)
        #y_= 2* y -1 # convert to (-1, 1)
        #use binary cross entropy formula from notes
        s = self.score(X) #compute scores
        p = torch.sigmoid(s)
        loss =-torch.mean(y * torch.log(p) + (1 - y) * torch.log(1 - p)) #calculate logistic loss
        return loss #mean across points

    def grad(self, X, y):
        """
        Computes the gradient of the logistic loss function with respect to the model weights.
        Arguments: 
        X: torch.tensor: an n x p feature matrix where n is the number of data points and p the number of features
        y: torch.tensor: the target vector of size n, where the values are 1 or 0 (binary classified)

        Returns:
        grad: a torch.tensor of size p representing the average gradient of the logistic loss function with respect to the model weights.
        """
        if self.w is None:
            self.score(X)
        s = self.score(X) #compute score
        
        # sigma = torch.sigmoid(s) #apply sigmoid function
        v= torch.sigmoid(s) - y
        v = v[:, None] # match dimensions of X
        # broadcast multiply and then sum over columns 
        grad = torch.sum(v * X, dim=0)
        #grad = -(X.T @ v) / X.size(0) # calculate gradient as average of x^t *v
        return grad
    
class GradientDescentOptimizer:
    def __init__(self, model):
        self.model= model
        self.prev_w = None #previous weight vector
    
    def step(self, X, y, alpha, beta):
        """
        Runs through one step of the gradient descent optimizer, updating the current and previous weights.
        This will optimize the weights used to classify the data.

        Arguments: 
        X: torch.tensor: an n x p feature matrix where n is the number of data points and p the number of features
        y: torch.tensor: the target vector of size n, where the values are 1 or 0 (binary classified)
        alpha: the learning rate, in between 0 and 1
        beta: the momentum term, in between 0 and 1
        """
        # Initialize weights with the correct shape
        if self.model.w is None:
            self.model.score(X)

        grad = self.model.grad(X, y) #compute gradient

        if self.prev_w is None:
            self.prev_w = self.model.w.clone()  # initialize first step
        #print("self.model.w" , self.model.w.size())
        #print("prev", self.prev_w.size())
        #print("grad", grad.size())
        new_w = self.model.w - ( alpha * grad ) + beta * (self.model.w - self.prev_w)
        #print(self.model.w.size())
        self.prev_w = self.model.w.clone()  # update prev weight
        self.model.w = new_w  # update current weight

class MyLinearRegression(LinearModel):
    
    def predict(self, X):
        """
        Return the score for each data point in X (no thresholding).
        """
        return self.score(X)

    def loss(self, X, y):
        """
        Compute the mean squared error (MSE) between predictions and targets.
        """
        preds = self.score(X) #predictions aka scores
        mse = torch.mean((preds - y) ** 2) #calculating error
        return mse

class OverParameterizedLinearRegressionOptimizer:

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        """
        Compute the optimal weights using the pseudoinverse.
        """
        X_pinv = torch.linalg.pinv(X)  # calculating the Moore-Penrose pseudoinverse
        self.model.w = X_pinv @ y

    def sig(x): 
        return 1/(1+torch.exp(-x))

    def square(x): 
        return x**2

class RandomFeatures:
    """
    Random sigmoidal feature map. This feature map must be "fit" before use, like this: 

    phi = RandomFeatures(n_features = 10)
    phi.fit(X_train)
    X_train_phi = phi.transform(X_train)
    X_test_phi = phi.transform(X_test)

    model.fit(X_train_phi, y_train)
    model.score(X_test_phi, y_test)

    It is important to fit the feature map once on the training set and zero times on the test set. 
    """

    def __init__(self, n_features, activation = sig):
        self.n_features = n_features
        self.u = None
        self.b = None
        self.activation = activation

    def fit(self, X):
        self.u = torch.randn((X.size()[1], self.n_features), dtype = torch.float64)
        self.b = torch.rand((self.n_features), dtype = torch.float64) 

    def transform(self, X):
        return self.activation(X @ self.u + self.b)