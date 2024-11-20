import os
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting, glm, masking
from ants import apply_transforms, image_read

root = '{0}Users{0}mococomac{0}Documents{0}data_backups{0}Bouba{0}original_format{0}bouba_scans{0}preproc2{0}'.format(
    os.sep)

subs_list = [os.path.basename(x) for x in glob.glob(os.path.join(root, 'sub*')) if os.path.isdir(x)]
subs_list.sort()
imgs = []

for sub in subs_list:
    path_to_img = root + '{0}{1}first_level_outputs{1}searchlight{1}'.format(sub, os.sep)
    native_to_MNI_xfm_path = glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*T1w_to-MNI152NLin6Asym*.h5')[0]
    subject_template_path = \
        glob.glob(root + '{1}{0}anat{0}'.format(os.sep, sub) + '*MNI152NLin6Asym*preproc*T1w.nii.gz')[0]
    img_MNI = apply_transforms(fixed=image_read(subject_template_path), moving=image_read(glob.glob(path_to_img + '*round_vs_spiky_searchlight.nii.gz')[0]),
                               transformlist=native_to_MNI_xfm_path, interpolator='nearestNeighbor').to_nibabel()
    img_MNI_data = img_MNI.get_fdata()
    img_MNI_data *= 100  # multiply by 100 to bring to scale
    img_MNI_data[img_MNI_data != 0] -= 50 # subtract chance to normalize all voxels (where there is signal)
    img_MNI = nib.Nifti1Image(img_MNI_data, img_MNI.affine, img_MNI.header)
    imgs.append(img_MNI)

avg = image.mean_img(imgs)
#nib.save(avg, root + 'group_rois/r_v_s_SL.nii')

design_matrix = pd.DataFrame(
    [1] * len(imgs),
    columns=["intercept"],
)

second_level_model = glm.second_level.SecondLevelModel(smoothing_fwhm=4, n_jobs=2)
second_level_model.fit(
    imgs,
    design_matrix=design_matrix,
)
z_map = second_level_model.compute_contrast(
    second_level_contrast="intercept",
    output_type="z_score",
)

plotting.plot_stat_map(z_map, bg_img=avg)


out_dict = glm.second_level.non_parametric_inference(
    imgs,
    design_matrix=design_matrix,
    model_intercept=True,
    n_perm=10000,
    two_sided_test=False, # only clusters above chance are considered
    smoothing_fwhm=2.0,
    n_jobs=4,
    threshold=0.001,
)

p_val = second_level_model.compute_contrast(output_type="p_value")
n_voxels = np.sum(image.get_data(second_level_model.masker_.mask_img_))
# Correcting the p-values for multiple testing and taking negative logarithm
neg_log_pval = image.math_img(
    f"-np.log10(np.minimum(1, img * {str(n_voxels)}))",
    img=p_val,
)
'''
threshold = 1  # p < 0.1
vmax = 2.69  # ~= -np.log10(1 / 500)

cut_coords = [0]

IMAGES = [
    neg_log_pval,
    out_dict["logp_max_t"],
    out_dict["logp_max_size"],
    out_dict["logp_max_mass"],
]
TITLES = [
    "Parametric Test",
    "Permutation Test\n(Voxel-Level Error Control)",
    "Permutation Test\n(Cluster-Size Error Control)",
    "Permutation Test\n(Cluster-Mass Error Control)",
]

fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
for img_counter, (i_row, j_col) in enumerate(
    itertools.product(range(2), range(2))
):
    ax = axes[i_row, j_col]
    plotting.plot_glass_brain(
        IMAGES[img_counter],
        colorbar=True,
        vmax=vmax,
        display_mode="z",
        plot_abs=False,
        cut_coords=cut_coords,
        threshold=threshold,
        figure=fig,
        axes=ax,
    )
    ax.set_title(TITLES[img_counter])
fig.suptitle("Group round vs spiky searchlight\n(negative log10 p-values)")
plt.show()
'''