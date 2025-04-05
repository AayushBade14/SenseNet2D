import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01,epochs=1000):
        """Initialize the Logistic Regression model"""
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None # weights
        self.B = None # bias

    def sigmoid(self,z):
        """Sigmoid activation function to clamp the values between 0 and 1"""
        return 1/(1+np.exp(-z))

    def fit(self,X,y):
        """Train the model using gradient descent"""
        n_samples,n_features = X.shape
        self.W = np.zeros(n_features)
        self.B = 0

        for _ in range(self.epochs):
            # compute predictions
            linear_model = np.dot(X,self.W) + self.B
            y_predicted = self.sigmoid(linear_model)

            # compute gradient
            dw = (1/n_samples) * np.dot(X.T,(y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.W -= self.learning_rate * dw
            self.B -= self.learning_rate * db
        
    def predict_prob(self,X):
        """Predict probabilities for input data"""
        return self.sigmoid(np.dot(X,self.W) + self.B)
    
    def predict(self,X,threshold=0.5):
        """Convert probabilities to class labels (0 or 1)"""
        return (self.predict_prob(X) >= threshold).astype(int)

    def accuracy(self,y_true,y_pred):
        """Calculate accuracy score"""
        return np.mean(y_true == y_pred)
