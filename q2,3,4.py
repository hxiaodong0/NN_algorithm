###########################Initialization
###########################Question 2,3,4,5.  2.2
#this code is the NN algorithm developed with library; for question2, run  function:k23410(); print_img() and error(); for question3
# run question3(); for question 4, run question4hog() and uncommon line 50, question4lbp() and uncommon line51 for 2.2 run cross(p,k);
from scipy.io import loadmat
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from skimage import feature
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from bokeh.plotting import figure, show,output_file

loadtemp = loadmat('YaleB_32x32.mat')
gnd = loadtemp["gnd"]  # 2414 X 1
fea = loadtemp["fea"]  # (2414, 1024)

def hog():
    fea_hog = []
    for i in range(len(fea)):
        img = fea[i].reshape((32, 32))
        hog_image = hog(img, orientations=8, cells_per_block=(1,1))
        fea_hog.append((hog_image))
    fea_hog = np.array(fea_hog)
    datasethog = pandas.DataFrame(fea_hog)
    return datasethog

def lbp():
    fea_lbp = []
    for i in range(len(fea)):
        img = fea[i].reshape((32, 32))
        lbp_img = local_binary_pattern(img, P =4, R = 4 )
        fea_lbp.append((lbp_img))
    fea_lbp = np.array(fea_lbp)
    kk = np.array(fea_lbp)
    fea_lbp = kk.reshape(2414,1024)
    datasetlbp = pandas.DataFrame(fea_lbp)
    return datasetlbp
dataset = pandas.DataFrame(fea)
#hog
index = pandas.DataFrame(gnd)
y = index
X = dataset
# X = hog()
# X= lbp()
Splitm = [(64-10)/64, (64-20)/64, (64-30)/64, (64-40)/64, (64-50)/64]
m = [10, 20, 30, 40, 50]
k = [2,3,5,10]
p = [ 1, 2, 3, 5, 10]

def cross_validation(k=1,p=2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(20/64), random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=k,p = p)
    classifier.fit(X_train, y_train.values.ravel())
    scores = cross_val_score(classifier, X, y.values.ravel(), cv=5)  # score array
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))  #The mean score and the 95% confidence interval
    y_pred = classifier.predict(X_test)
    count = 0
    correct = 0
    error_locations = []
    real_locations = []
    for i in range(len(y_pred)):
        try:
            if abs((int(y_test.iloc[i,0]) - int(y_pred[i]))) <= 6 :
                correct+=1
            else:
                error_locations.append(y_pred[i])
                real_locations = y_test.iloc[i,0]
            count += 1
        except:
            pass
    rate = (count-correct)/count
    return rate , scores

def main(n,k=1,p=2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n, random_state=None)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=k,p = p)
    classifier.fit(X_train, y_train.values.ravel())
    y_pred = classifier.predict(X_test)
    count = 0
    correct = 0
    error_locations = []
    real_locations = []
    for i in range(len(y_pred)):
        try:
            if abs((int(y_test.iloc[i,0]) - int(y_pred[i]))) <= 6 :
                correct+=1
            else:
                error_locations.append(y_pred[i])
                real_locations = y_test.iloc[i,0]
            count += 1
        except:
            pass
    rate = (count-correct)/count
    return n,rate ,error_locations , real_locations

#general Question2

def k23410():  #Question2
    lst = []
    for j in range(len(k)):
        for i in range(len(Splitm)):
            n,rate ,error_locations , real_locations = main(Splitm[i],k[j])
            lst.append((k[j], m[i], rate))
    df_out = pandas.DataFrame(lst)
    df_out.to_excel("question2.xlsx")
    return df_out

#images,wrong samples
################
# lst = []
def error(): #Question2
    n,rate ,error_locations , real_locations = main(Splitm[2] , 10)
    real_img = fea[real_locations].reshape((32, 32))
    false_img1 = fea[error_locations[4]].reshape((32, 32))
    imgplot = plt.imshow(false_img1)
    plt.show()

def print_img():
    n,rate ,error_locations , real_locations = main(Splitm[2] , 10)
    real_img = fea[real_locations].reshape((32, 32))
    false_img1 = fea[error_locations[4]].reshape((32, 32))
    imgplot = plt.imshow(false_img1)
    plt.show()

def k1():
    lst1= []
    for i in range(len(Splitm)):
        n,rate ,error_locations , real_locations = main(Splitm[i])
        lst1.append(( m[i], rate))
    df1_out = pandas.DataFrame(lst1)
    df1_out.to_excel("question111.xlsx")
    return df1_out
# add a line renderer
def question3():
    lst = []
    for i in range(len(p)):
        n, rate, error_locations, real_locations = main(Splitm[2], k[2], p[i])
        lst.append((p[i], rate))
    df1_out = pandas.DataFrame(lst)
    return df1_out

def question4lbp():
    lst = []
    X = lbp()
    for i in range(2):
        n, rate, error_locations, real_locations = main(Splitm[2], k[2], p[i])
        lst.append((p[i], rate))
    df1_out = pandas.DataFrame(lst)
    return df1_out
def question4hog():
    lst = []
    X = hog()
    for i in range(2):
        n, rate, error_locations, real_locations = main(Splitm[2], k[2], p[i])
        lst.append((p[i], rate))
    df1_out = pandas.DataFrame(lst)
    return df1_out

def cross(p,k):  #question 2.2 cross validation
    score = []
    for i in p:
        for j in k:
            x = cross_validation(j, i)
            score.append([j,i,x])
    return score
ll = cross(p,k)
# ll = pandas.DataFrame(ll)

