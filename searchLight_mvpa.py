from operator import itemgetter
from nilearn.decoding import SearchLight
import os
import nibabel as nib
from glmsingle import GLM_single
import glob
from sklearn.pipeline import make_pipeline
from nilearn.maskers import NiftiMasker
from ants import apply_transforms, image_read, from_nibabel
import pandas as pd
from nilearn import image
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, permutation_test_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from nilearn.masking import apply_mask, unmask

# Glm Single needs:

# a 'data' list of size N containing the 4d (x,y,z,time) data of N runs

# a 'design' cell  list of size N, with inside the timepoints by condition code matrices containing event info (
# timepoints by condition{say you have two conditions, then it would be a 0 1 or 1 0 at each timepoint, according to
# that event's conditions)

## PATHS, VARIABLES, FILENAMES, CONSTANTS
'''roots = ['{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc1{0}'.format(
    os.sep),
    '{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc2{0}'.format(
        os.sep)]'''
roots = ['{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc2{0}'.format(
    os.sep)]
root_n = 1  # 0
n_perms = 1000  # number of permutations for shuffling
condition_pairs = [('round', 'spiky'), ('mixed_round', 'mixed_spiky'), ('mixed_round', 'round'),
                   ('mixed_spiky', 'spiky')]
cv = LeaveOneGroupOut()
scaler = StandardScaler()
classifier = LinearSVC(penalty="l2", max_iter=int(1e4), dual='auto')
# classifier = LogisticRegression(penalty='l1', solver="liblinear")
selector = VarianceThreshold()
pipe = make_pipeline(scaler, classifier)
pipeline = make_pipeline(StandardScaler(), classifier)
radius = 3

# loop over each participant, per sample
for root in roots:
    root_n += 1
    subs_list = []
    [subs_list.append(x.split('/')[-1]) for x in list(set(glob.glob(root + 'sub*')) - set(glob.glob(root + '*.html')))]
    subs_list.sort()

    for sub in subs_list:
        path_to_data = root + '{1}{0}func{0}'.format(os.sep, sub)
        outdir = root + '{0}{1}first_level_outputs'.format(sub, os.sep)
        roi_outdir = outdir + '{0}ROI{0}'.format(os.sep)
        outputdir_glmsingle = os.path.join(outdir, 'GLMsingle')
        sl_outdir = outdir + '{0}searchlight'.format(os.sep)
        if not os.path.exists(sl_outdir):
            os.mkdir(sl_outdir)

        brain_masks_files = glob.glob(path_to_data + '*task-bouba*T1w*-brain_mask*.nii.gz')
        brain_masks_files.sort()

        nii_files = glob.glob(path_to_data + '*task-bouba*T1w*-preproc_bold*.nii.gz')
        nii_files.sort()

        path_to_events = root + '{1}{0}events{0}task{0}'.format(os.sep, sub)

        csv_files = glob.glob(path_to_events + '*.csv')
        csv_files.sort()

        log_files = glob.glob(path_to_events + '*.log')
        log_files.sort()

        native_to_MNI_path = glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*T1w_to-MNI152NLin6Asym_*.h5')[
            0]
        t1_MNI = image.load_img(glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*MNI152NLin6Asym*preproc*T1w.nii.gz')[0])
        t1_native = image.load_img(glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*_desc-preproc_T1w.nii.gz')[0])

        # IMAGES & ROIS
        # check if we have already estimated the betas for this participant
        if not os.path.exists(outputdir_glmsingle):
            images = [image.load_img(nii_files[i]) for i in range(len(nii_files))]
            brain_masks = [image.load_img(brain_masks_files[i]) for i in range(len(brain_masks_files))]
            masked_images = [unmask(apply_mask(img, mask), mask) for img, mask in zip(images, brain_masks)]

            # apply brainmask to images (may consider adding .astype(np.int16) to get fdata here)
            data = [i.get_fdata() for i in masked_images]
        else:  # otherwise just load a single image for the affine and shape
            masked_images = [image.load_img(nii_files[0]), image.load_img(nii_files[1])]

        rs_gm_mask = image.resample_img(image.binarize_img(
            image.load_img(glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*run-1_label-GM_probseg*.nii.gz')[0]),
            two_sided=False),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')
        sl = SearchLight(mask_img=rs_gm_mask, radius=radius, estimator=pipeline, cv=LeaveOneGroupOut())
        # check if we have already estimated the betas for this participant, else run glm single
        if not os.path.exists(outputdir_glmsingle):
            ## REWORK THE EVENTS FILES ##
            # for our design matrix, we want each row to have onset, duration and trialtype(condition)
            events = []
            events_to_rework = [pd.read_csv(csv_files[i]) for i in range(len(csv_files))]
            logs = [pd.read_table(log_files[i], header=None, names=['timestamp', 'label', 'message']) for i in
                    range(len(log_files))]

            # loop thru csv files
            duration = 1.4
            TR = 1.32
            for event_csv, log in zip(events_to_rework, logs):
                event_csv.dropna(subset=['Sound'], inplace=True)
                # empty df where we'll stash the onsets durations and conditions of trials
                trial_dfs = []
                if (log['message'] == 'Keypress: 5').any():
                    trgr_idx = np.where((log['message'] == 'Keypress: 5'))[0][0]
                else:
                    trgr_idx = np.where((log['message'] == 'text_2: autoDraw = True'))[0][0]

                t0 = log.iloc[trgr_idx, 0]
                # change shape 'none' to 'baseline' and 'mixed' to 'meaningful'
                event_csv.loc[event_csv['Shape'] == 'none', 'Shape'] = 'baseline'

                # loop thru trials
                for idx, row in event_csv.iterrows():
                    if row['Condition'] == 'Meaningfull':
                        trial_condition = 'meaningful'
                    else:
                        trial_condition = row['Shape']

                    if trial_condition != 'baseline':
                        trial_name = trial_condition
                    else:
                        continue  # if it's baseline, skip this row
                    # get onset and duration of trial
                    onset = row['stimulus.started'] - t0
                    # wrap up everything in the table
                    trial_df = pd.DataFrame({'onset': onset, 'duration': duration, 'trial_type': trial_name},
                                            index=[idx])
                    trial_dfs.append(trial_df)

                # stash events table in the list
                events.append(pd.concat(trial_dfs, ignore_index=True))
            # convert to scans
            # I NEED TO ADD A BEHAV RESP REGRESSOR (AFTER EACH PRESENTATION?)
            # get n scans for each run
            num_scans = [i[0].shape[-1] for i in data]
            design = []
            for run in range(len(data)):
                # get closest scan to each event for each run
                events[run]['scans'] = np.round(events[run]['onset'] / TR).astype(int)
                design_matrix = np.zeros((num_scans[run], 5))
                # 1 hot encoding of our conditions
                encoded_events = pd.get_dummies(events[run]['trial_type'])
                for index, _ in events[run].iterrows():
                    # get position of our event (rows = scans); -1 because of base 0 indexing
                    scan_index = events[run].iloc[index]['scans'] - 1
                    # copy our 1-hot encoded event
                    design_matrix[scan_index, :] = encoded_events.iloc[index].values
                design.append(design_matrix)
            ## RUN THE GLM ##
            xyz = data[0].shape[:3]
            xyzt = data[0].shape

            # consolidate design matrices
            designALL = np.concatenate(design, axis=0)

            # construct a vector containing 0-indexed condition numbers in chronological order
            corder = []
            for p in range(designALL.shape[0]):
                if np.any(designALL[p]):
                    corder.append(np.argwhere(designALL[p])[0, 0])

            corder = np.array(
                corder)  # here we have our conditions: order of columns is correct (alphabetical): encoded_events.columns

            run_length = 60
            num_runs = len(corder) // run_length
            runs = np.concatenate([np.full(run_length, i) for i in range(num_runs)])

            opt = dict()

            # set important fields for completeness (but these would be enabled by default)
            opt['wantlibrary'] = 1
            opt['wantglmdenoise'] = 1
            opt['wantfracridge'] = 1

            # for the purpose of this example we will keep the relevant outputs in memory
            # and also save them to the disk
            opt['wantfileoutputs'] = [0, 0, 0, 1]
            opt['wantmemoryoutputs'] = [0, 0, 0, 1]

            # running python GLMsingle involves creating a GLM_single object
            # and then running the procedure using the .fit() routine
            glmsingle_obj = GLM_single(opt)
            results_glmsingle = glmsingle_obj.fit(design, data, duration, TR, outputdir=outputdir_glmsingle)
            results_glmsingle = results_glmsingle['typed']
            mean_betas = np.nanmean(results_glmsingle['betasmd'], axis=-1)
            nib.save(nib.Nifti1Image(mean_betas, affine=masked_images[0].affine), roi_outdir + 'mean_onoff_betas.nii')
            r2_map = np.squeeze(results_glmsingle['R2'].reshape(xyz))
            nib.save(nib.Nifti1Image(r2_map, affine=masked_images[0].affine), roi_outdir + 'r2_map.nii')
            np.save(outputdir_glmsingle + os.sep + 'conditions.npy', corder)
            np.save(outputdir_glmsingle + os.sep + 'runs.npy', runs)
        else:
            results_glmsingle = np.load(os.path.join(outputdir_glmsingle, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                        allow_pickle=True).item()
            corder = np.load(os.path.join(outputdir_glmsingle, 'conditions.npy'))
            runs = np.load(os.path.join(outputdir_glmsingle, 'runs.npy'))

        conditions_dict = {
            'meaningful': np.where(corder == 0)[0],
            'mixed_round': np.where(corder == 1)[0],
            'mixed_spiky': np.where(corder == 2)[0],
            'round': np.where(corder == 3)[0],
            'spiky': np.where(corder == 4)[0],
        }

        ## RUN CLASSIFICATION ##
        for pair in condition_pairs:
            condition1, condition2 = pair
            pair_name = f'{condition1}_vs_{condition2}'
            mvpa_indices = []

            for k in pair:
                mvpa_indices.extend(conditions_dict[k])

            beta_maps = [nib.Nifti1Image(results_glmsingle['betasmd'][:, :, :, i], affine=masked_images[0].affine) for
                         i in
                         mvpa_indices]
            g = runs[mvpa_indices]
            y = LabelEncoder().fit_transform(corder[mvpa_indices])

            sl.fit(beta_maps, y, groups=g)
            # save in native space
            sl_image = image.new_img_like(masked_images[0], sl.scores_)
            nib.save(sl_image,
                     sl_outdir + os.sep + sub + '_' + pair_name + '_searchlight.nii.gz')
            # resample to t1 dimensions (native space)
            sl_t1 = image.resample_img(sl_image,
                                       target_affine=t1_native.affine, target_shape=t1_native.shape[0:3],
                                       interpolation='nearest')
            # shift to MNI space the resampled image and save it
            transformed = apply_transforms(fixed=from_nibabel(t1_MNI), moving=from_nibabel(sl_t1),
                                           transformlist=native_to_MNI_path,
                                           interpolator='nearestNeighbor').to_nibabel()
            nib.save(transformed,
                     sl_outdir + os.sep + sub + '_' + pair_name + '_searchlight_MNI.nii.gz')
