from workermanager2 import WorkerManager2
from param import param_from_txt
from helper import s2f
from rect import rect

# this to be loaded from configuration file
config={
    'webservice_url': 'http://roweb3.uhmc.sbuh.stonybrook.edu:3000/api'
}

wm = WorkerManager2(config)
wm.run()

#rectum_locnet_worker = wm.create_worker("rectum_locnet_train4_202108")
# # rectum_locnet_worker.train()
# img_file = '.\\sample_data\\test\\00000020\\CT.mhd'
# rect_w,_,_ = rectum_locnet_worker.locate(img_file)

# # the true values
# rectum_info = '.\\sample_data\\test\\00000020\\Rectum.info'
# bbox = s2f(param_from_txt(rectum_info)['bbox'].split(','))
# rect_true = rect(low=[bbox[0], bbox[2], bbox[4]],
#                  high=[bbox[1], bbox[3], bbox[5]])
# print('rect_true=', rect_true)

# rect_diff = rect(low=rect_w.low - rect_true.low,
#                  high=rect_w.high - rect_true.high)
# print('rect_diff=', rect_diff)

print('done')
