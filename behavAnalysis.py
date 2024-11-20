import pandas as pd
import numpy as np
import os
import glob


root = '{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc2{0}'.format(
    os.sep)
global_dict = {'round': 1, 'mixed_round': 2, 'mixed_spiky': 3, 'spiky': 4}

subs_list = [os.path.basename(x) for x in glob.glob(os.path.join(root, 'sub*')) if os.path.isdir(x)]
subs_list.sort()

all_data = pd.DataFrame()
all_subjects_data = pd.DataFrame()

for sub in subs_list:
    path_to_data = root + '{1}{0}func{0}'.format(os.sep, sub)
    outdir = root + '{0}{1}first_level_outputs'.format(sub, os.sep)
    roi_outdir = outdir + '{0}ROI{0}'.format(os.sep)
    outputdir_glmsingle = os.path.join(outdir, 'GLMsingle')

    brain_masks_files = glob.glob(path_to_data + '*task-bouba*T1w*-brain_mask*.nii.gz')
    brain_masks_files.sort()

    nii_files = glob.glob(path_to_data + '*task-bouba*T1w*-preproc_bold*.nii.gz')
    nii_files.sort()

    path_to_events = root + '{1}{0}events{0}task{0}'.format(os.sep, sub)

    csv_files = glob.glob(path_to_events + '*.csv')
    csv_files.sort()

    log_files = glob.glob(path_to_events + '*.log')
    log_files.sort()
    # get the key responses, getting rid of baseline events
    cols = ['Condition', 'RespCorr', 'Shape', 'resp_stimulus.keys', 'resp_interv.keys']
    events = [pd.read_csv(csv_files[i], usecols=cols).dropna(subset=['Condition'], ignore_index=True) for i in range(len(csv_files))]
    # get response code and translate responses to shape then to global shape scale, keeping them for each run
    all_runs_data = pd.DataFrame()
    for i, e in enumerate(events):
        ev = e.loc[e['Shape'] != 'mixed', :].copy()
        # merge responses during stim with responses during resp time
        ev['Resp'] = ev['resp_stimulus.keys'].fillna(ev['resp_interv.keys'])
        # homogenize so that 1 is always round: figure out resp code of run

        # there's an issue with one 'mixed spiky' having CorrResp 4, which is always wrong; so let's make sure
        # CorrResp == resp code
        ev.loc[(ev['Shape'] == 'mixed_spiky') & (ev['RespCorr'] == 4), 'RespCorr'] = 2

        # check just in case
        shape_to_respcorr = ev.groupby('Shape')['RespCorr'].unique()

        if len(shape_to_respcorr[shape_to_respcorr.apply(len) > 1].index) > 0:
            print('error with response code of ' + sub + ', run ', str(i))

        # get what the response code was
        resp_code = dict(ev.groupby('RespCorr')['Shape'].unique().apply(lambda x: x[0]).loc[:])
        # check how well the subj responded, in case he mixed up the resp code
        r = np.corrcoef(ev['RespCorr'], ev['Resp'])[0][0]

        err = np.sum((ev['RespCorr'] - ev['Resp']) == 0) / len(ev)
        print('{} percentage of fully correct shape resps is {}% in run{}'.format(sub, int(np.round(err, 2) * 100),
                                                                                  i))
        if err < 0.10:
            print('sub {}, run {} has suspicious responses (err={})! inverted manually..'.format(sub, i, err))
            resp_code = {1: resp_code[4], 2: resp_code[3], 3: resp_code[2], 4: resp_code[1]}
        # translate from local code to global code (1->round, 4->spiky) KEEPING MEANINGFUL SHAPED SOUNDS

        ev['HomogResp'] = ev['Resp'].map(resp_code).map(global_dict)
        all_runs_data = pd.concat([all_runs_data, ev], ignore_index=True)
    # stack these into a single list across subjs
    sub_df = all_runs_data.groupby('Shape')['HomogResp'].mean().to_frame().T
    sub_df['sub'] = sub
    all_subjects_data = pd.concat([all_subjects_data, sub_df], ignore_index=True)

all_subjects_data.to_csv('behav.csv')

import seaborn as sns
import matplotlib.pyplot as plt

order = ['round', 'mixed_round', 'mixed_spiky', 'spiky']
sns.barplot(all_subjects_data, order=order, palette='tab10')
plt.ylim(1, 4)
plt.tight_layout()
plt.savefig('behav.png', dpi=300, bbox_inches='tight')