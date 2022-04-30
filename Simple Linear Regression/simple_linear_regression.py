""" 
Simple linear Regression 
Python using sklearn 

"""

# improt library;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# loading the dataset

data = pd.read_csv("C:/Users/kavif/Documents/Git/Machine_learning_python/Simple Linear Regression/data/Salary_Data.csv")
x_data = data.iloc[:,0:1].values
y_data =  data.iloc[:,1:].values


# ploting the dataset 
def data_graph(x, y):
    plt.scatter(x, y)
    plt.show()



if __name__ == "__main__":
    
    x_train = x_data[0:20,:]
    x_test = x_data[20:,:]

    y_train = y_data[0:20,:]
    y_test = y_data[20:,:]

    regressor = LinearRegression()
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)

    plt.scatter(x_test,y_test)
    plt.plot(y_pred, color="red")
    plt.show()


