"""
the classification result of 'Study to Study Transfer'
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
import warnings

Study = ['FR-CRC', 'AT-CRC', 'CN-CRC', 'US-CRC', 'DE-CRC']

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # read .tsv file
    tag = 'species'
    species_url = 'data//' + tag + '//feat_rel_crc.tsv'
    meta_url = 'data//meta//meta_crc.tsv'
    species = pd.read_csv(species_url, sep='\t', header=0)
    meta = pd.read_csv(meta_url, sep='\t', header=0)
    #row name
    sample_name = meta['Sample_ID'].tolist()
    #predict_matrix store the prediction result
    predict_matrix = pd.DataFrame(np.zeros(shape=(species.shape[1],5)),index=sample_name,columns=['FR-CRC','AT-CRC','CN-CRC','US-CRC','DE-CRC'])


    # train model for study to study transfer
    for study in Study:
        #study = 'FR-CRC'
        print('training on {} with Study to Study Transfer'.format(study))
        # get sample_id about study
        train_data = pd.read_csv('pre_data//species_stst//{}//{}_data.csv'.format(study,study),header=0,index_col=0)
        train_sample_id = train_data._stat_axis.values.tolist()
        y = train_data['Label']
        x = train_data.drop(columns='Label')
        # normalize using standard
        x = StandardScaler().fit_transform(x)
        train_pre_matrix = pd.DataFrame(np.zeros(shape=(train_data.shape[0], 10)), index=train_sample_id,columns=['1', '2', '3','4', '5', '6', '7', '8', '9', '10'])

        #10 fold cross-validation repeated 10 times
        for i in range(10):
            skf = StratifiedKFold(n_splits=10)
            for train_index , test_index in skf.split(x,y):

                train_x = x[train_index]
                train_y = y[train_index]
                test_x = x[test_index]
                test_y = y[test_index]

                lr = LogisticRegression(penalty='l1',solver='liblinear',n_jobs=-1)
                lr.fit(train_x,train_y)
                proba = lr.predict_proba(test_x)[:,1]
                train_sample_id = np.array(train_sample_id)
                test_sample = train_sample_id[test_index]
                train_pre_matrix.loc[test_sample,'%d'%(i+1)] = proba
                for study_test in Study:
                    if study_test is study:
                        continue
                    else:
                        test_stst = pd .read_csv('pre_data//species_stst//{}//{}_data.csv'.format(study_test,study_test),header=0,index_col=0)
                        test_stst_index = test_stst._stat_axis.values.tolist()
                        test_stst_y = test_stst['Label']
                        test_stst_x = test_stst.drop(columns='Label')
                        proba_stst = lr.predict_proba(test_stst_x)[:,1] / 100
                        predict_matrix.loc[test_stst_index,study] += proba_stst





        train_pre_mean = train_pre_matrix.mean(1)
        predict_matrix.loc[train_pre_matrix._stat_axis.values.tolist(),study] = train_pre_mean

    print(predict_matrix)
    predict_matrix.to_csv('predict_stst.csv')





    






