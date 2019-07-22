"""
Transfer Kernel Learning
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.model_selection import GridSearchCV
from TKL import TKL
from sklearn.svm import SVC
Study = ['FR-CRC', 'AT-CRC', 'CN-CRC', 'US-CRC', 'DE-CRC']
#Study = ['AT-CRC']
log_n0 = 1e-5
sd_min_q = 0.1
kernel = 'linear'


warnings.filterwarnings("ignore")
tag = 'species'
species_url = 'data//' + tag + '//feat_rel_crc.tsv'
meta_url = 'data//meta//meta_crc.tsv'
species = pd.read_csv(species_url, sep='\t', header=0)
meta = pd.read_csv(meta_url, sep='\t', header=0)
#row name
sample_name = meta['Sample_ID'].tolist()
"""
predict_matrix store the prediction result
"""
predict_matrix = pd.DataFrame(np.zeros(shape=(species.shape[1],6)),index=sample_name,columns=['FR-CRC','AT-CRC','CN-CRC','US-CRC','DE-CRC','LOSO'])
accuracy_matrix = pd.DataFrame(np.zeros(shape=(7,5)),index = ['FR-CRC', 'AT-CRC', 'CN-CRC', 'US-CRC', 'DE-CRC','LOSO','LOSO_CV']
                               ,columns=['FR-CRC', 'AT-CRC', 'CN-CRC', 'US-CRC', 'DE-CRC'])
"""
train model for study to study transfer
"""
for study in Study:
    print('training on {} with Study to Study Transfer'.format(study))
    # get sample_id about study
    train_data = pd.read_csv('pre_data//species_stst//{}//{}_data.csv'.format(study,study),header=0,index_col=0)
    train_sample_id = train_data._stat_axis.values.tolist()
    y = train_data['Label']
    x = train_data.drop(columns='Label')
    x = x.values
    y = y.values
    train_pre_matrix = pd.DataFrame(np.zeros(shape=(train_data.shape[0], 10)),
                                    index=train_sample_id, columns=['1', '2', '3','4', '5', '6', '7', '8', '9', '10'])
    """
    10 fold cross-validation repeated 10 times
    """
    right_num_study = 0
    for i in range(10):
        skf = StratifiedKFold(n_splits=10)
        parameters = {'C': [0.1, 0.5, 1, 5, 10, 50,100, 500,1000]}
        for train_index ,test_index in skf.split(x,y):
            train_x = x[train_index]
            train_y = y[train_index]
            test_x = x[test_index]
            test_y = y[test_index]
            """
            remove features with std=0
            """
            std_preprocess = np.std(train_x,axis=0)
            train_x = train_x[:,std_preprocess!=0]
            train_x = np.log10(train_x + log_n0)
            mean = np.mean(train_x,axis=0)
            std = np.std(train_x,axis=0)
            q = np.percentile(std,10)
            train_x = (train_x - mean) / (std + q)

            lr = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=-1)
            clf = GridSearchCV(lr, parameters, cv=10)
            clf.fit(train_x, train_y)

            lr = LogisticRegression(penalty='l1',solver='liblinear',n_jobs=-1,C=clf.best_params_['C'])
            lr.fit(train_x,train_y)
            #print(lr.n_iter_)
            test_x = test_x[:,std_preprocess!=0]
            test_x = np.log10(test_x + log_n0)
            test_x = (test_x - mean) / (std + q)
            proba = lr.predict_proba(test_x)[:,1]
            train_sample_id = np.array(train_sample_id)
            test_sample = train_sample_id[test_index]
            train_pre_matrix.loc[test_sample,'%d'%(i+1)] = proba
            right_num_study += np.sum(test_y == lr.predict(test_x))
            for study_test in Study:
                if study_test == study:
                    continue
                else:
                    test_stst = pd .read_csv('pre_data//species_stst//{}//{}_data.csv'.format(study_test,study_test),header=0,index_col=0)
                    test_stst_index = test_stst._stat_axis.values.tolist()
                    test_stst_y = test_stst['Label']
                    test_stst_x = test_stst.drop(columns='Label')
                    test_stst_x = test_stst_x.values
                    test_stst_x = test_stst_x[:,std_preprocess!=0]
                    test_stst_x = np.log10(test_stst_x+log_n0)
                    test_stst_x = (test_stst_x - mean) / (std + q)
                    train_space,test_space = TKL(train_x,test_stst_x,eta=2.0,kernel=kernel).tkl()

                    svc = SVC(kernel='precomputed',probability=True)
                    clf = GridSearchCV(svc,parameters,cv=10)
                    clf.fit(train_space,train_y)

                    svc = SVC(kernel='precomputed',probability=True,C=clf.best_params_['C'])
                    svc.fit(train_space,train_y)
                    proba_stst = svc.predict_proba(test_space)[:,1] / 100
                    predict_matrix.loc[test_stst_index,study] += proba_stst
                    score = svc.score(test_space,test_stst_y)
                    accuracy_matrix.loc[study,study_test] += score / 100
        train_pre_mean = train_pre_matrix.mean(1)
        predict_matrix.loc[train_pre_matrix._stat_axis.values.tolist(),study] = train_pre_mean
        accuracy_matrix.loc[study,study] += right_num_study / (len(x)*10)
        right_num_study = 0

 # LOSO
for study in Study:
    print('testing on study {} with LOSO'.format(study))
    train_data = pd.read_csv('pre_data//species_loso//{}//train_data.csv'.format(study),header=0,index_col=0)
    train_sample_id = train_data._stat_axis.values.tolist()
    y = train_data['Label']
    x = train_data.drop(columns='Label')
    x = x.values
    y = y.values
    for i in range(10):
        parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]}
        right_num_study = 0
        skf = StratifiedKFold(n_splits=10)
        for train_index, test_index in skf.split(x, y):
            train_x = x[train_index]
            train_y = y[train_index]
            std_preprocess = np.std(train_x, axis=0)
            train_x = train_x[:, std_preprocess != 0]
            train_x = np.log10(train_x + log_n0)
            mean = np.mean(train_x, axis=0)
            std = np.std(train_x, axis=0)
            q = np.percentile(std, 10)
            train_x = (train_x - mean) / (std + q)

            lr = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=-1)
            clf = GridSearchCV(lr, parameters, cv=10)
            clf.fit(train_x, train_y)

            test_x = x[test_index]
            test_y = y[test_index]
            lr = LogisticRegression(penalty='l1', solver='liblinear', n_jobs=-1,C = clf.best_params_['C'])
            lr.fit(train_x,train_y)

            test_x = test_x[:,std_preprocess!=0]
            test_x = np.log10(test_x + log_n0)
            test_x = (test_x - mean) / (std + q)
            right_num_study += np.sum(test_y == lr.predict(test_x))

            test_data = pd.read_csv('pre_data//species_loso//{}//test_data.csv'.format(study),header=0,index_col=0)
            test_sample_id = test_data._stat_axis.values.tolist()
            test_loso_y = test_data['Label']
            test_loso_y = test_loso_y.values
            test_loso_x = test_data.drop(columns='Label')
            #test_loso_x = scalar.transform(test_loso_x)
            test_loso_x = test_loso_x.values
            test_loso_x = test_loso_x[:,std_preprocess != 0]
            #print(test_loso_x.shape)
            test_loso_x = np.log10(test_loso_x + log_n0)
            test_loso_x = (test_loso_x - mean) / (std + q)
            #using trasfer kernel learning
            train_space, test_space = TKL(train_x , test_loso_x , eta = 2.0 , kernel=kernel).tkl()

            svc = SVC(kernel='precomputed', probability=True)
            clf = GridSearchCV(svc, parameters, cv=10)
            clf.fit(train_space, train_y)

            svc = SVC(kernel='precomputed', probability=True,C = clf.best_params_['C'])
            svc.fit(train_space, train_y)

            proba_loso = svc.predict_proba(test_space)[:,1]/100
            predict_matrix.loc[test_sample_id,'LOSO'] += proba_loso
            score = svc.score(test_space,test_loso_y)
            #accuracy_matrix_loso[study].append(score)
            accuracy_matrix.loc['LOSO',study] += score / 100
        accuracy_matrix.loc['LOSO_CV', study] += right_num_study / (len(x) * 10)

#print(predict_matrix)
predict_matrix.to_csv('predict_matrix_TKL.csv')
accuracy_matrix.to_csv('accuracy_matrix_TKL.csv')