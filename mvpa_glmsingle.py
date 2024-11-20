import matplotlib.pyplot as plt
import os
import nibabel as nib
from glmsingle import GLM_single
import glob
from sklearn.pipeline import make_pipeline
from nilearn.maskers import NiftiMasker
from ants import apply_transforms, image_read
import pandas as pd
from random import sample
from nilearn import image
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, permutation_test_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from nilearn.masking import apply_mask, unmask

# Glm Single needs:

# a 'data' list of size N containing the 4d (x,y,z,time) data of N runs

# a 'design' cell  list of size N, with inside the timepoints by condition code matrices containing event info (
# timepoints by condition{say you have two conditions, then it would be a 0 1 or 1 0 at each timepoint, according to
# that event's conditions)

## PATHS, VARIABLES, FILENAMES, CONSTANTS
'''
roots = ['{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc1{0}'.format(
    os.sep),
    '{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc2{0}'.format(
        os.sep)]
'''
roots = ['{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc2{0}'.format(
    os.sep)]

root_n = 0  # 0
n_perms = 1000  # number of permutations for shuffling
condition_pairs = [('round', 'spiky'), ('mixed_round', 'mixed_spiky'), ('mixed_round', 'round'),
                   ('mixed_spiky', 'spiky'), ('meaningful', 'meaningless')]
cv = LeaveOneGroupOut()
scaler = StandardScaler()
classifier = LinearSVC(penalty="l2", max_iter=int(1e4), dual='auto')
# classifier = LogisticRegression(penalty='l1', solver="liblinear")
selector = VarianceThreshold()
pipe = make_pipeline(scaler, classifier)

this_run_results = []
rois = ['EVC', 'V1', 'V2', 'V3', 'LO1', 'LO2', 'V3A', 'V3B', 'AC', 'A1', 'A2', 'A3', 'PCG', 'HIP', 'BRC']
# 'UNC', 'SMT', 'MTR', 'IPS'
# rois = ['BRC']

# loop over each participant, per sample
for root in roots:
    root_n += 1
    subs_list = [os.path.basename(x) for x in glob.glob(os.path.join(root, 'sub*')) if os.path.isdir(x)]
    subs_list.sort()
    a1_roi_img_path = root + 'group_rois/audio_1_roi_mni.nii.gz'
    a2_roi_img_path = root + 'group_rois/audio_2_roi_mni.nii.gz'
    a3_roi_img_path = root + 'group_rois/audio_3_roi_mni.nii.gz'
    uncinate_roi_img_path = root + 'group_rois/uncinate_roi_mni.nii.gz'
    pcg_roi_img_path = root + 'group_rois/amygdala_roi_mni.nii.gz'
    hippo_roi_img_path = root + 'group_rois/hippo_roi_mni.nii.gz'
    ips_roi_img_path = root + 'group_rois/ip_roi_mni.nii.gz'
    somato_roi_img_path = root + 'group_rois/somato_2nd_roi_mni.nii.gz'
    motor_roi_img_path = root + 'group_rois/motor_roi_mni.nii.gz'
    speech_roi_img_path = root + 'group_rois/speech_roi_mni.nii.gz'

    path_to_result = '{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}mvpa_results{0}'.format(
        os.sep)
    output_filename_all = f'{path_to_result}perm_results_trial.csv'

    # THIS DOESN'T WORK IF ROOT1 IS INCLUDED!
    if os.path.exists(output_filename_all):
        prev_results = pd.read_csv(output_filename_all)
        done_subjects = prev_results['sub'].unique()
        done_subjects.sort()

        subs_list = list(set(done_subjects) - set(subs_list))
        subs_list.sort()

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
        if 'preproc1' in root:
            MNI_to_native_xfm_path = glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*MNI152NLin2009cAsym_to-T1w*.h5')[0]
        elif 'preproc2' in root:
            MNI_to_native_xfm_path = glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*MNI152NLin6Asym_to-T1w*.h5')[
                0]
        subject_template_path = glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*_desc-preproc_T1w.nii.gz')[0]

        # IMAGES & ROIS
        # check if we have already estimated the betas for this part.
        if not os.path.exists(outputdir_glmsingle):
            images = [image.load_img(nii_files[i]) for i in range(len(nii_files))]
            brain_masks = [image.load_img(brain_masks_files[i]) for i in range(len(brain_masks_files))]
            masked_images = [unmask(apply_mask(img, mask), mask) for img, mask in zip(images, brain_masks)]

            # apply brainmask to images (may consider adding .astype(np.int16) to get fdata here)
            data = [i.get_fdata() for i in masked_images]
        else:  # otherwise just load a single image for the affine and shape
            masked_images = [image.load_img(nii_files[0]), image.load_img(nii_files[1])]
        # load ROIs
        resampled = image.resample_img(image.load_img(roi_outdir + '{0}_visual_areas.nii'.format(sub)),
                                       target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
                                       interpolation='nearest')
        wang_path = '/Users/mococomac/Documents/data_backups/Bouba/original_format/bouba_scans/preproc2/sourcedata/freesurfer/{}/mri/wang15_converted.nii'.format(
            sub)
        rs_wang = image.resample_img(image.load_img(wang_path),
                                     target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
                                     interpolation='nearest')

        a1_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(a1_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        a2_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(a2_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        a3_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(a3_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        uncinate_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(uncinate_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        pcg_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(pcg_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        somato2_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(somato_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        motor_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(motor_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        hippo_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(hippo_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        ips_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(ips_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        speech_in_native_space = image.resample_img(
            apply_transforms(fixed=image_read(subject_template_path), moving=image_read(speech_roi_img_path),
                             transformlist=MNI_to_native_xfm_path, interpolator='nearestNeighbor').to_nibabel(),
            target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
            interpolation='nearest')

        rs_gm_mask = image.resample_img(image.math_img("img1 > 0", img1=image.load_img(glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*run-1_label-GM_probseg*.nii.gz')[0])),
                                        target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
                                        interpolation='nearest')
        ac_in_native_space = image.math_img("(img1+img2+img3) >= 1", img1=a1_in_native_space, img2=a2_in_native_space, img3=a3_in_native_space)

        roi_masks = {'EVC': image.new_img_like(resampled, np.isin(resampled.get_fdata(), [1, 2, 3])),
                     'V1': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 1)),
                     'V2': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 2)),
                     'V3': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 3)),
                     'LO1': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 14)),
                     'LO2': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 15)),
                     'V3A': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 16)),
                     'V3B': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 17)),
                     'AC': image.math_img('img1*img2', img1=ac_in_native_space, img2=rs_gm_mask),
                     'A1': image.math_img('img1*img2', img1=a1_in_native_space, img2=rs_gm_mask),
                     'A2': image.math_img('img1*img2', img1=a2_in_native_space, img2=rs_gm_mask),
                     'A3': image.math_img('img1*img2', img1=a3_in_native_space, img2=rs_gm_mask),
                     'UNC': image.math_img('img1*img2', img1=uncinate_in_native_space, img2=rs_gm_mask),
                     'PCG': image.math_img('img1*img2', img1=pcg_in_native_space, img2=rs_gm_mask),
                     'SMT': image.math_img('img1*img2', img1=somato2_in_native_space, img2=rs_gm_mask),
                     'MTR': image.math_img('img1*img2', img1=motor_in_native_space, img2=rs_gm_mask),
                     'HIP': image.math_img('img1*img2', img1=hippo_in_native_space, img2=rs_gm_mask),
                     'IPS': image.math_img('img1*img2', img1=ips_in_native_space, img2=rs_gm_mask),
                     'BRC': image.math_img('img1*img2', img1=speech_in_native_space, img2=rs_gm_mask)
                     }

        # evc_roi = image.new_img_like(resampled, np.isin(resampled.get_fdata(), [1, 2, 3]).astype(np.int16))

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

        # randomly sample trials from meaningless (3 per shape condition)
        meaningless_indices = []
        for condition in conditions_dict.keys():
            for nrun in np.unique(runs):
                if condition != 'meaningful':
                    indices = conditions_dict[condition][runs[conditions_dict[condition]] == nrun]
                    meaningless_indices.extend(sample(indices.tolist(), 3))

        # add meaningless condition (no need to sort, but nicer)
        meaningless_indices.sort()
        conditions_dict['meaningless'] = np.array(meaningless_indices)

        ## RUN CLASSIFICATION ##
        for pair in condition_pairs:
            condition1, condition2 = pair
            pair_name = f'{condition1}_vs_{condition2}'
            mvpa_indices = []

            for k in pair:
                mvpa_indices.extend(conditions_dict[k])
            mvpa_indices = np.array(mvpa_indices)

            beta_maps = [nib.Nifti1Image(results_glmsingle['betasmd'][:, :, :, i], affine=masked_images[0].affine) for
                         i in
                         mvpa_indices]

            for roi, mask in roi_masks.items():
                if roi in rois:
                    masker = NiftiMasker(mask_img=mask, standardize=False, smoothing_fwhm=None)

                    X = masker.fit_transform(beta_maps)
                    # remove voxels w no variance (doesn't change much)
                    X = selector.fit_transform(X)
                    g = runs[mvpa_indices]
                    # if we have multiclasses (bunch of randomly sampled meaningless) then binarize
                    if 'meaningful' in pair and 'meaningless' in pair:
                        y = LabelEncoder().fit_transform(corder[mvpa_indices] != 0)
                    else:
                        y = LabelEncoder().fit_transform(corder[mvpa_indices])

                    # CROSS VALIDATION
                    folds = cv.split(X, y, g)
                    accs = []
                    for fold in folds:
                        train_idx, test_idx = fold
                        x_train, x_test = X[train_idx, :], X[test_idx, :]
                        y_train, y_test = y[train_idx], y[test_idx]
                        pipe.fit(x_train, y_train)
                        accs.append(accuracy_score(pipe.predict(x_test), y_test))

                    # PERMUTATION SHUFFLING
                    '''
                    permuted_accuracies = np.zeros(n_perms)
                    for p in range(n_perms):
                        fold_accuracies = np.zeros(len(accs))
                        ii = 0
                        for train_idx, test_idx in cv.split(X, y, g):
                            # shuffle the labels
                            x_train, x_test = X[train_idx], X[test_idx]
                            y_train, y_test = y[train_idx], y[test_idx]
                            np.random.shuffle(y_train)
                            # train n test
                            pipe.fit(x_train, y_train)
                            fold_accuracies[ii] = accuracy_score(pipe.predict(x_test), y_test)
                            ii += 1
    
                        permuted_accuracies[p] = np.mean(fold_accuracies)
                    '''
                    this_run_results.append({
                        'sub': sub,
                        'sample': root_n,
                        'ROI': roi,
                        'pair_name': pair_name,
                        'accuracy': np.round(np.mean(accs) * 100, 1),
                        'nulldistr': 'no' #np.round(permuted_accuracies*100, 1)
                    })

this_run_results_df = pd.DataFrame(this_run_results)
if os.path.exists(output_filename_all):
    all_results_df = pd.concat([prev_results, this_run_results_df])
    all_results_df.to_csv(output_filename_all, index=False)
else:
    this_run_results_df.to_csv(output_filename_all, index=False)
