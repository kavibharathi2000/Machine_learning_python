"""
Multiple Linear Regression
Python

"""




import pandas as pd
import numpy as np 


if __name__ == "__main__":

    # loading the dataset
    file = pd.read_csv("C:/Users/kavif/Documents/Git/Machine_learning_python/Multiple linear Regression/data/50_Startups.csv")
    x_data = file.iloc[:,0:3].values
    y_data = file.iloc[:,4:].values

    # train test data split
    from sklearn.model_selection import train_test_split
    x_train ,x_test , y_train , y_test = train_test_split(x_data , y_data , test_size=0.2 , random_state=0)

    # fitting the model
    from sklearn.linear_model import LinearRegression
    machine = LinearRegression()
    machine.fit(x_train , y_train)

    # predicting the data
    pred = machine.predict(x_test)
    output = np.concatenate((pred, y_test) , axis=1)
    print(output)





