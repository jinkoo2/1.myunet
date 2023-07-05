from workermanager import WorkerManager
from param import param_from_txt
from helper import s2f
from rect import rect
from image_helper import extract_largest_connected_compoment, save_image
wm = WorkerManager('.\\workers')

##############
# input param
img_file = '.\\sample_data\\test\\00000020\\CT.mhd'
out_file = '.\\_test3\\Rectum.seg.mhd'

# debug param
out_dir_for_debug = '.\\_test3'
out_file_unet = '.\\_test3\\Rectum.unet.seg.mhd'

# ##############################################
# # LocNetWorker to get the organ rect
rectum_locnet_worker = wm.create_worker("rectum_locnet_train4_202108")
# rectum_locnet_worker.train()
roi_rect_w, _, _ = rectum_locnet_worker.locate(img_file)
print('roi_rect_w=', roi_rect_w)

##############################################
# UNetWorker to get the organ segmentation
rectum_unet_worker = wm.create_worker("rectum_unet_train3_202108")
# rectum_unet_worker.train()
img_seg = rectum_unet_worker.segment(img_file, roi_rect_w, out_file_unet)

#############################
# pick the largest compoment
img_largest_cc = extract_largest_connected_compoment(img_seg)

save_image(img_largest_cc, out_file)

print('done')
