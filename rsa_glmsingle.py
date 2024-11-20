import os
import glob
import seaborn as sns
from scipy.optimize import nnls
from scipy.stats import kendalltau
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from nilearn.maskers import NiftiMasker
from sklearn.metrics import pairwise_distances
from nilearn import image, plotting, glm, masking
from scipy.spatial.distance import squareform
from ants import apply_transforms, image_read
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import LinearRegression

## SET PATHS AND DEFINE SOME CONSTANTS ##
root = '{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc2{0}'.format(
    os.sep)

path_to_result = '{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}rsa_results{0}'.format(
    os.sep)

# variable with names of conditions & constants
conditions_dict = {0: 'meaningful', 1: 'mixed_round', 2: 'mixed_spiky', 3: 'round', 4: 'spiky'}
global_dict = {'round': 1, 'mixed_round': 2, 'mixed_spiky': 3, 'spiky': 4}
smooth = None
standardization = False  # https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2013.00174/full
metrica = 'euclidean'
reg_nnls = LinearRegression(positive=True)
selector = VarianceThreshold()
cond_names = ['mixed_round', 'mixed_spiky', 'round', 'spiky']
subs_list = [os.path.basename(x) for x in glob.glob(os.path.join(root, 'sub*')) if os.path.isdir(x)]
subs_list.sort()
rois = ['V1', 'V2', 'V3', 'LO1', 'LO2', 'V3A', 'V3B', 'A1', 'A2', 'A3', 'AMY',  'HIP']
trialwise_ts = {roi: np.zeros(len(subs_list)) for roi in rois}
trialwise_zetas = {roi: np.zeros(len(subs_list)) for roi in rois}
conditionwise_ts = {roi: np.zeros(len(subs_list)) for roi in rois}
conditionwise_zetas = {roi: np.zeros(len(subs_list)) for roi in rois}
zs = {roi: [] for roi in rois}
kts = {roi: [] for roi in rois}
conditionwise_coefs = []
trialwise_coefs = []
sub_i = 0
sc = StandardScaler()

a1_roi_img_path = root + 'group_rois/audio_1_roi_mni.nii.gz'
a2_roi_img_path = root + 'group_rois/audio_2_roi_mni.nii.gz'
a3_roi_img_path = root + 'group_rois/audio_3_roi_mni.nii.gz'
uncinate_roi_img_path = root + 'group_rois/uncinate_roi_mni.nii.gz'
amygdala_roi_img_path = root + 'group_rois/amygdala_roi_mni.nii.gz'
hippo_roi_img_path = root + 'group_rois/hippo_roi_mni.nii.gz'
ips_roi_img_path = root + 'group_rois/ip_roi_mni.nii.gz'
somato_roi_img_path = root + 'group_rois/somato_2nd_roi_mni.nii.gz'
motor_roi_img_path = root + 'group_rois/motor_roi_mni.nii.gz'
## LOOP OVER EACH PARTICIPANT ##
for sub in subs_list:
    neural_rdvs = {roi: [] for roi in rois}
    feature_rdvs = []
    # paths and files
    outdir = root + '{0}{1}first_level_outputs'.format(sub, os.sep)
    roi_outdir = outdir + '{0}ROI{0}'.format(os.sep)
    outputdir_glmsingle = os.path.join(outdir, 'GLMsingle')
    nii_files = glob.glob(root + '{1}{0}func{0}'.format(os.sep, sub) + '*task-bouba*T1w*-preproc_bold*.nii.gz')
    nii_files.sort()
    path_to_events = root + '{1}{0}events{0}task{0}'.format(os.sep, sub)
    csv_files = glob.glob(path_to_events + '*.csv')
    csv_files.sort()
    log_files = glob.glob(path_to_events + '*.log')
    log_files.sort()

    MNI_to_native_xfm_path = glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*MNI152NLin6Asym_to-T1w*.h5')[0]
    subject_template_path = glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*preproc*T1w.nii.gz')[0]

    # get the key responses, getting rid of baseline events
    cols = ['Condition', 'RespCorr', 'Shape', 'resp_stimulus.keys', 'resp_interv.keys']
    events = [pd.read_csv(csv_files[i], usecols=cols).dropna(subset=['Condition']) for i in range(len(csv_files))]
    # get response code and translate responses to shape then to global shape scale, keeping them for each run
    runwise_scores = []
    for i, e in enumerate(events):
        ev = e.loc[e['Shape'] != 'mixed', :].copy()
        # merge responses during stim with responses during resp time
        ev['Resp'] = ev['resp_stimulus.keys'].fillna(ev['resp_interv.keys'])
        # get what the response code was
        resp_code = dict(ev.groupby('RespCorr')['Shape'].unique().apply(lambda x: x[0]).loc[:])
        # check how well the subj responded, in case he mixed up the resp code
        r = np.corrcoef(ev['RespCorr'], ev['Resp'])[0][0]

        err = np.sum((ev['RespCorr'] - ev['Resp']) == 0) / len(ev)
        print('{} percentage of fully correct shape resps is {}% in run{}'.format(sub, int(np.round(err, 2) * 100), i))
        if err < 0.10:
            print('sub {}, run {} has suspicious responses (err={})! inverted manually..'.format(sub, i, err))
            resp_code = {1: resp_code[4], 2: resp_code[3], 3: resp_code[2], 4: resp_code[1]}
        # translate from local code to global code (1->round, 4->spiky)
        for con in cond_names:
            int_resp = ev.loc[(ev['Condition'] != 'Meaningfull') & (ev['Shape'] == con)]
            runwise_scores.append(
                {'run': i, 'con': con, 'score': int_resp['Resp'].map(resp_code).map(global_dict).mean()})

    '''
    # load the ROI mask
    masked_images = [image.load_img(nii_files[0]), image.load_img(nii_files[1])]
    resampled = image.resample_img(image.load_img(roi_outdir + '{0}_visual_areas.nii'.format(sub)),
                                   target_affine=masked_images[0].affine, target_shape=masked_images[0].shape[0:3],
                                   interpolation='nearest')

    evc_roi = image.new_img_like(resampled, np.isin(resampled.get_fdata(), [1, 2, 3]).astype(np.int16))
    '''
    # load the betas and their condition/run info
    # load the ROI mask
    masked_images = [image.load_img(nii_files[0]), image.load_img(nii_files[1])]
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

    amygdala_in_native_space = image.resample_img(
        apply_transforms(fixed=image_read(subject_template_path), moving=image_read(amygdala_roi_img_path),
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
    '''
    roi_masks = {'V1': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 1)),
                 'V2': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 2)),
                 'V3': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 3)),
                 'LO1': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 14)),
                 'LO2': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 15)),
                 'V3A': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 16)),
                 'V3B': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 17)),
                 'A1': a1_in_native_space, 'A2': a2_in_native_space, 'A3': a3_in_native_space,
                 'UNC': uncinate_in_native_space, 'AMY': amygdala_in_native_space,
                 'SMT': somato2_in_native_space, 'MTR': motor_in_native_space, 'HIP': hippo_in_native_space,
                 'IPS': ips_in_native_space
                 }
    '''
    roi_masks = {'V1': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 1)),
                 'V2': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 2)),
                 'V3': image.new_img_like(resampled, np.isin(resampled.get_fdata(), 3)),
                 'LO1': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 14)),
                 'LO2': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 15)),
                 'V3A': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 16)),
                 'V3B': image.new_img_like(rs_wang, np.isin(rs_wang.get_fdata(), 17)),
                 'A1': a1_in_native_space, 'A2': a2_in_native_space, 'A3': a3_in_native_space,
                 'AMY': amygdala_in_native_space, 'HIP': hippo_in_native_space
                 }

    results_glmsingle = np.load(os.path.join(outputdir_glmsingle, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                allow_pickle=True).item()

    beta_maps = [nib.Nifti1Image(results_glmsingle['betasmd'][:, :, :, i], affine=masked_images[0].affine) for
                 i in range(results_glmsingle['betasmd'].shape[-1])]
    corder = np.load(os.path.join(outputdir_glmsingle, 'conditions.npy'))
    rorder = np.load(os.path.join(outputdir_glmsingle, 'runs.npy'))
    R = {roi: np.zeros((len(cond_names), len(cond_names))) for roi in roi_masks.keys()}
    B = pairwise_distances(pd.DataFrame(runwise_scores).pivot(index='con', columns='run', values='score'),
                           metric=metrica)
    stacked_rdvs = []
    runwise_feature_rdvs = {roi: [] for roi in roi_masks}
    all_neural_rdvs = {roi: [] for roi in roi_masks}

    for roi, mask in roi_masks.items():
        rsa_masker = NiftiMasker(mask_img=mask, standardize=standardization, smoothing_fwhm=smooth)
        betas = rsa_masker.fit_transform(beta_maps)
        betas = selector.fit_transform(betas)

        roi_feature_rdvs = []
        roi_neural_rdvs = []

        for run in np.unique(rorder):
            run_data = betas[np.where(rorder == run)[0]]
            conds = corder[rorder == run]
            # exclude trials w no resp
            run_events = events[run][conds != 0]
            run_events['resp'] = run_events['resp_stimulus.keys'].fillna(run_events['resp_interv.keys'])

            no_resp_indices = np.isnan(run_events['resp'])
            rr = run_data[conds != 0][~no_resp_indices]
            cc = conds[conds != 0][~no_resp_indices]
            shape_judgements = run_events['resp'][~no_resp_indices]
            resp_code = dict(run_events.groupby('RespCorr')['Shape'].unique().apply(lambda x: x[0]).loc[:])
            # transform conditions from numbers to names
            mapped_cc = [resp_code[val] for val in shape_judgements]
            # translate names to overall response code
            shape_judgements = np.array([global_dict[val] for val in mapped_cc])

            trialwise_neural_rdv = squareform(pairwise_distances(rr, metric=metrica).round(5))
            trialwise_feature_rdv = squareform(pairwise_distances(shape_judgements.reshape(-1, 1), metric=metrica).round(5))

            roi_feature_rdvs.append(trialwise_feature_rdv)
            roi_neural_rdvs.append(trialwise_neural_rdv)

            t, _ = kendalltau(trialwise_feature_rdv, trialwise_neural_rdv)
            z = np.arctanh(np.sin(t * 0.5 * np.pi))

            kts[roi].append(t)
            zs[roi].append(z)
        # stash the trial wise correlations between shape judgments and neural activity
        trialwise_ts[roi][sub_i] = np.mean(kts[roi])
        trialwise_zetas[roi][sub_i] = np.mean(zs[roi])

        runwise_feature_rdvs[roi] = roi_feature_rdvs
        all_neural_rdvs[roi] = roi_neural_rdvs

    # conditionwise distances
        for i in np.unique(corder):
            if i == 0:
                continue
            for j in np.unique(corder):
                if j == 0:
                    continue
                betas_cond1 = np.mean(betas[np.where(corder == i)], axis=0).reshape(1, -1)
                betas_cond2 = np.mean(betas[np.where(corder == j)], axis=0).reshape(1, -1)
                R[roi][i-1][j-1] = pairwise_distances(betas_cond1, betas_cond2, metric=metrica)[0][0]

        neural_rdv = squareform(np.round(R[roi], 4))
        feature_rdv = squareform(np.round(B, 4))

        tt, _ = kendalltau(neural_rdv, feature_rdv)
        conditionwise_ts[roi][sub_i] = tt
        zz = np.arctanh(np.sin(tt * 0.5 * np.pi))
        conditionwise_zetas[roi][sub_i] = zz
        stacked_rdvs.append(neural_rdv)

        # reweighting (conditionwise)
        stacked_matrix = np.column_stack(stacked_rdvs)
        intercept_column = np.ones((stacked_matrix.shape[0], 1))

        A = np.hstack((intercept_column, stacked_matrix))
        b = feature_rdv
        # TODO NORMALIZE DATA
        # conditionwise_coefs.append(nnls(A, b)[0])
        try:
            conditionwise_coefs.append(kendalltau(b,reg_nnls.fit(A, b.reshape(-1, 1)).predict(A))[0])
        except RuntimeError:
            continue

    # reweighting (trialwise)
    runwise_coefs = []
    for run in np.unique(rorder):
        b = runwise_feature_rdvs['V1'][run]  # ROI doesn't matter here, they're all the same
        A = np.insert(np.array([all_neural_rdvs[roi][run] for roi in rois]).T, 0, 1, axis=1)
        try:
            # runwise_coefs.append(nnls(A, b)[0])
            runwise_coefs.append(kendalltau(b,reg_nnls.fit(A, b.reshape(-1, 1)).predict(A))[0])
        except LinAlgWarning:
            continue  # skip this run

    trialwise_coefs.append(np.mean(runwise_coefs, axis=0))
    sub_i += 1


pd.DataFrame(trialwise_zetas).to_csv(path_to_result+'trialwise_zetas.csv')
pd.DataFrame(conditionwise_zetas).to_csv(path_to_result+'conditionwise_zetas.csv')
pd.DataFrame(trialwise_zetas).to_csv(path_to_result+'trialwise_zetas.csv')
pd.DataFrame(pd.DataFrame(conditionwise_coefs).iloc[:, 1:]).to_csv(path_to_result+'conditionwise_coefs.csv')
pd.DataFrame(pd.DataFrame(trialwise_coefs).iloc[:, 1:]).to_csv(path_to_result+'trialwise_coefs.csv')

breakpoint()
print('yeyo')