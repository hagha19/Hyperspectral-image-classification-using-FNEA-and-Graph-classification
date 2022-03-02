import numpy as np
import scipy.io as sco
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
import sklearn
from sklearn.svm import SVC
import spectral 
import pysptools.material_count as cnt
from sklearn import random_projection
from sklearn.metrics import accuracy_score, cohen_kappa_score
import skimage.segmentation as seg
import time
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
# ---------------------------------------------------------
# object-based Hyperspectral image classification
# Returns a classified image.
#
# Inputs:
#           data: input hyperspectral image [row*col*dim]
#           gt: ground_truth 
#           DR_method: dimension reduction method, Default= 1(Random Projection).
#           classification_method: supervised classier determination, Default =3(Random Forest classifier) .
#.          segmentation_method = Defaul value is 0. uses the output from ecognition software. in this case the segmented image direction should be addressed.
#
# Output:
#           Final classification map using pixelwise classifier + object based improvement.
# ------------------------------------------------------------


def DimMeasure(data):
    #######virtual dimensionality measure with HYsime
    row, col, dim = data.shape
    hy = cnt.HySime()
    kf, Ek = hy.count(data)
    print('Testing HySime')
    print('  Virtual dimensionality is: k =', kf)
    num_dim = kf
    return num_dim

def Do_dimensionReduction(data, num_dim, DR_method = 1):
     # Dimension Reduction method: [RandomProjection , Kernel PCA]
     row, col, dim = data.shape
     data = data.reshape(row*col , dim)
     if DR_method == 1:
         # GRP
            GRP = random_projection.GaussianRandomProjection(n_components=num_dim,eps = 0.5, random_state=2019)
            GRP.fit(data)
            data = GRP.transform(data)
    
     else:  
    #KPCA
        KPCA = sklearn.decomposition.KernelPCA(n_components=num_dim,eps = 0.5, random_state=2019)
        KPCA.fit(data)
        data = KPCA.transform(data)
     return data.reshape(row, col, num_dim)



def DO_classification(data, gt, classification_method =3 , Search_param=0):
    row, col, num_dim = data.shape
    data = data.reshape(row*col , num_dim)
    gt = gt.reshape(row*col,1)
    a = np.where(gt == 0)
    gt1 = np.delete(gt, a[0] , 0)
    data1 = np.delete(data, a[0] , 0) 
    sc = StandardScaler()
    data1 = sc.fit_transform(data1)
    data = sc.transform(data)
    X_train,  X_test, Y_train, Y_test = train_test_split(data1, gt1, train_size = 0.2, random_state=0,stratify=gt1)
#    ids = np.zeros((len(X_train),1))
#    for counts in range(len(X_train)):
#        ids[counts] = data.tolist().index(X_train[counts,:].tolist())
        
    ################################# classification
     # [SVM , KNN, RandomForest, ML]
    if classification_method == 1:    
        #clf = svm.SVC(kernel='rbf',max_iter= 1000)
        #clf.fit(X_train, Y_train)    
        #clf = sklearn.linear_model.SGDClassifier(max_iter=3000, tol=1e-3)
        #clf.fit(X_train, Y_train)
        print('.......................SVM Classifier............')
        Y_train.shape=(Y_train.size,)    
        cv = StratifiedKFold(n_splits=5,random_state=0).split(X_train,Y_train)
        grid = GridSearchCV(SVC(), param_grid=dict(gamma=2.0**sp.arange(-4,4), C=10.0**sp.arange(0,3)), cv=cv,n_jobs=-1)
        grid.fit(X_train, Y_train)
        clf = grid.best_estimator_

        clf.probability= True
        clf.fit(X_train,Y_train)
        print('.......................SVM............')
    elif classification_method == 2:
        print('.......................KNN Classifier............') 
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors= gt.max())
        clf.fit(X_train , Y_train)
         
     
    elif classification_method == 3:
        print('.......................RF Classifier............') 
        # Number of trees in random forest
        if Search_param:
            
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            # Use the random grid to search for best hyperparameters
            # First create the base model to tune
            rf = RandomForestClassifier()
            # Random search of parameters, using 3 fold cross validation, 
            # search across 100 different combinations, and use all available cores
            rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
            # Fit the random search model
            rf_random.fit(X_train, Y_train)
            clf = rf_random.best_estimator_
        else:    
            #best param
            """ best_param = {'n_estimators': 1000,
                 'min_samples_split': 2,
                 'min_samples_leaf': 1,
                 'max_features': 'auto',
                 'max_depth': 50,
                 'bootstrap': False}
            """
            clf = RandomForestClassifier(n_estimators=1000,
                 min_samples_split= 2,
                 min_samples_leaf= 1,
                 max_features= 'auto',
                 max_depth= 50,
                 bootstrap= False)
            clf.fit(X_train, Y_train)
    elif classification_method == 4:
        clf = GaussianNB()
        clf.fit(X_train, Y_train)
            
    classified_image = clf.predict(data)
    a = np.where(gt == 0)
    #gt_labels = np.delete(gt, a[0] , 0)
    #labels = np.delete(classified_image, a[0] , 0) 
    Y_pred = clf.predict(X_test)
    accu = accuracy_score(Y_test , Y_pred)
    Kappa = cohen_kappa_score(Y_test , Y_pred)
    print("Overal Acuuracy = " , accu)
    print("Kappa Coeficient = " , Kappa)
    return data.reshape(row,col,num_dim), clf, classified_image, X_train,  X_test, Y_train, Y_test, accu, Kappa


def segment_graph(data, gt, segmentation_method = 0):
    # segementation with felzenszwalb
    if segmentation_method:
        segmented_HSI = seg.felzenszwalb(data, scale = 100 , sigma= 1, min_size = 50)
    else:
        segmented_HSI = np.array(Image.open(r"C:\Users\hamid\Desktop\HSI classification with FNEA and KNN\SegmentedPaviaU.tif"))
    return segmented_HSI


def final_calssification(segmented_HSI, data, gt, clf):
    #Reshaped_segmented_HSI = np.reshape(segmented_HSI , segmented_HSI.size[0] * segmented_HSI.size[1] , 1)
   
    row, col, num_dim = data.shape
    data = data.reshape(row*col , num_dim)
    gt = gt.reshape(row*col , 1)
    Reshaped_segmented_HSI = segmented_HSI.reshape(row*col, 1)
    classified_final = gt.copy()
    for counts in range(Reshaped_segmented_HSI.max()+1):
        IDs = np.where(Reshaped_segmented_HSI == counts)
        values = data[IDs[0],:]
        labels = clf.predict(values)
        classLabel = np.bincount(labels).argmax()
        classified_final[IDs[0]] = classLabel
    gt_zeros = np.where(gt == 0)
    gt_labels = np.delete(gt, gt_zeros[0] , 0)
    labels = np.delete(classified_final, gt_zeros[0] , 0) 
    accu = accuracy_score(gt_labels , labels)
    Kappa = cohen_kappa_score(gt_labels , labels)
    print("ّFinal Overal Acuuracy = " , accu)
    print("Final Kappa Coeficient = " , Kappa)
    return classified_final, accu, Kappa



if __name__ == "__main__":
    
    data_name = 'PaviaU'
    start_time = time.time()
    ## Reading Envi data ***********
    if data_name == 'PaviaU':
        data_path = "E:/university/MSC/thesis/data/Pavia University scene/paviaU.hdr"
        gt_path = "C:/Users/hamid/Desktop/PaviaU_gt.mat"
    elif data_name == 'Salinas':         
        data_path = "E:/university/MSC/thesis/data/Salinas scene/salinas_corrected.dat.hdr"
        gt_path = "E:/university/MSC/thesis/data/Salinas scene/Salinas_gt.mat" 
    data = spectral.io.envi.open(data_path)
    #data = Image.open(r"C:\Users\hamid\Desktop\paviaU-8band.tif")   
    data = data[:,:,:]
    row, col, dim = data.shape    
    gt = sco.loadmat(gt_path)
    
    if data_name == 'PaviaU':
        gt = gt['paviaU_gt']
    elif data_name == 'Salinas':  
        gt = gt['salinas_gt']

    num_dim = DimMeasure(data)
    data = Do_dimensionReduction(data, num_dim, DR_method = 1)
    data, clf, classified_image, X_train,  X_test, Y_train, Y_test, accu, Kappa = DO_classification(
            data, gt)
    segmented_HSI = segment_graph(data, gt,1)
    classified_final, accuF, KappaF = final_calssification(segmented_HSI, data, gt, clf)
    
    elapsed_time = time.time() - start_time
    print(
        "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")
    
    gt = gt.reshape(row*col, 1)
    CopyClasified = classified_image.copy()
    gt_zeros = np.where(gt == 0)
    CopyClasified[gt_zeros[0]]= 0
    
    Copyclassified_final = classified_final.copy()
    Copyclassified_final[gt_zeros[0]]= 0
    
    fig , ax = plt.subplots(1,3, sharex=True , sharey=True)
    ax[0].imshow(CopyClasified.reshape(row,col))
    ax[0].set_title('classified image')
    ax[1].imshow(Copyclassified_final.reshape(row,col))
    ax[1].set_title('classified final')
    ax[2].imshow(gt.reshape(row,col))
    ax[2].set_title('ground truth')
