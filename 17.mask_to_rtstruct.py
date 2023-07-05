from rtstruct_helper import load_dcm_images_from_path, generate_base_dataset, add_study_and_series_information, add_patient_information, add_refd_frame_of_ref_sequence, RTStruct

import numpy as np
import SimpleITK as sitk
from os.path import join
from os import mkdir

ct_path = 'U:\\temp\\New folder\\ct_dicom'
ct_mhd = '.\\data1\\00001500\CT.mhd'
mask_mhd = '.\\data1\\00001500\Rectum.mhd'
rs_path = 'U:\\temp\\New folder\\RS.new.dcm'


mask_img = sitk.ReadImage(mask_mhd)
mask_np = sitk.GetArrayFromImage(mask_img).astype('bool')

new_shape = (mask_np.shape[2], mask_np.shape[1], mask_np.shape[0])
mask_np_trans = np.empty(new_shape, dtype=bool)

print('mask_np_trans.shape=', mask_np_trans.shape)

print('new_shape=', new_shape)

# reorder from z,y,x->z,y,x (interesting... but this is the way it works with rt-util for now).
for k in range(mask_np.shape[0]):
    for j in range(mask_np.shape[1]):
        for i in range(mask_np.shape[2]):
            mask_np_trans[j, i, k] = bool(mask_np[k, j, i])

print('mask_np.shape=', mask_np.shape)

series_data = load_dcm_images_from_path(ct_path)
series_data.sort(key=lambda ds: ds.SliceLocation, reverse=False)

print('len(series_data)=', len(series_data))
ds0 = series_data[0]
print('series_data[0]=', ds0)
print('FrameOfReferenceUID=', ds0.FrameOfReferenceUID)

# file dataset
rtstruct_fds = generate_base_dataset()

# add infomation to the fileDataset based on ct series data
add_study_and_series_information(rtstruct_fds, series_data)
add_patient_information(rtstruct_fds, series_data)
add_refd_frame_of_ref_sequence(rtstruct_fds, series_data)

rtstruct = RTStruct(series_data, ds=rtstruct_fds)


rtstruct.add_roi(
    mask=mask_np_trans,
    color=[255, 0, 255],
    name="Rectum"
)

rtstruct.save(rs_path)


print('done')
