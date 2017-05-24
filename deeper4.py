#Multiclass Classification of Flower Species
#Evaluating keras neural networks with scikit-learn
#Loading important libraries
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#Initializing random number generator to ensure reproducibility
seed = 7
numpy.random.seed(seed)

#Loading the dataset
dataframe = read_csv('iris.csv', header=None)
dataset = dataframe.values

#Separating inputs and outputs
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

#Converting the output variable into a matrix using one hot encoding
#Encoding clas values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#Converting integers to dummy variables ie one hot encoding
dummy_y = np_utils.to_categorical(encoded_Y)

#Defining the baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
    #Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
	
#Applying the keras classifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

#Evaluating the model using k-fold
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)"%(results.mean()*100, results.std()*100))