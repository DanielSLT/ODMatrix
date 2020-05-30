Since we have signed the Data Confidentiality Agreement, the data are not allowed to be shared. But the structure of
 our deep learning model can be made available.

## code file description  
In the directory gravityModel, there are three Python files. File predictO.py and File predictD.py are used to predict 
inflow and outflow, respectively. The outputs of these two programs are used as the input of getODbyRegression.py. By 
running getODbyRegression.py, the output of this program is the output of the gravity model.

File final_model.py contains the whole structure of our model, which consists of Modified Gravity Model Component and 
Deep Learning Component with Convolutional Auto-Encoder. The input of the model is a series of OD matrices, time 
information, and the output of File getODbyRegression.py, which is the output of the gravity model.