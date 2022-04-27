
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression




if __name__ == "__main__":
    
    file_path = "C:/Users/kavif/Downloads/Salary_Data.csv"
    data = pd.read_csv(file_path)
    x_data = data.iloc[:,0:1].values
    y_data = data.iloc[:,:1].values

    machine = LinearRegression()
    machine.fit(x_data, y_data)
    machine.predict((1.5))
