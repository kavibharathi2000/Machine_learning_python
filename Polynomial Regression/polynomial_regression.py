"""
Polynomial Regression Python 

"""

# importion
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


# main 
if __name__ == "__main__":
    
    # loading the dataset
    data_file = pd.read_csv("data/Position_Salaries.csv")
    x_data = data_file.iloc[:,1:2].values
    y_data = data_file.iloc[:,2:].values

    # data Preprocessing
    feature = PolynomialFeatures(degree=4)
    x_poly_data = feature.fit_transform(x_data)


    # fitting the model

    # data without preprocessing
    regressor = LinearRegression()
    regressor.fit(x_data, y_data)
    
    # plotting the model
    plt.scatter(x_data, y_data, color="red")
    plt.plot(x_data , regressor.predict(x_data), color="blue")
    plt.xlabel("position")
    plt.ylabel("salary")
    plt.title("Non Polynomail Feature")
    plt.show()
    
    
    # data with preprocessing
    poly_regressor = LinearRegression()
    poly_regressor.fit(x_poly_data , y_data)

    # plotting the model
    plt.scatter(x_data , y_data , color='red')
    plt.plot(x_data, poly_regressor.predict(x_poly_data), color="blue")
    plt.xlabel("position")
    plt.ylabel("salary")
    plt.title("Polynomial feature")
    plt.show()