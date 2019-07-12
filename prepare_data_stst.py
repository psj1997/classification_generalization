"""
prepare data for stst
"""

import pandas as pd
import numpy as np
import warnings
import os
Study = ['FR-CRC', 'AT-CRC', 'CN-CRC', 'US-CRC', 'DE-CRC']
root_dir = 'E://classification_generalization//'

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    if not os.path.exists(root_dir+'pre_data//species_stst'):
        os.makedirs(root_dir+'pre_data//species_stst')
    root_dir = root_dir + 'pre_data//species_stst//'
    for study in Study:
        if not os.path.exists(root_dir+'{}'.format(study)):
            os.makedirs(root_dir+'{}'.format(study))


    tag = 'species'
    species_url = 'data//' + tag + '//feat_rel_crc.tsv'
    meta_url = 'data//meta//meta_crc.tsv'
    species = pd.read_csv(species_url, sep='\t', header=0)
    meta = pd.read_csv(meta_url, sep='\t', header=0)

    for study in Study:
        Sample_id = []
        label = []
        meta_row_id = []
        for i in range(len(meta[['Sample_ID', 'Study']])):
            if meta.loc[i, ['Study']].tolist()[0] == study:
                Sample_id.append(meta.loc[i, ['Sample_ID']].tolist()[0])
                meta_row_id.append(i)
                if meta.loc[i,['Group']].tolist()[0] == 'CRC':
                     label.append(1)
                else:
                     label.append(-1)

        species_study = species[Sample_id].T
        meta_study = meta.loc[meta_row_id,:]
        species_study['Label'] = label
        species_study.to_csv(root_dir+'{}'.format(study)+'//{}_data.csv'.format(study))


