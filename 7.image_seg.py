from dataset import read_and_normalize
from helper import add_a_dim_to_tensor
img_path = './data_test/train/00000000/CT.mhd'
img, img_coord = read_and_normalize(img_path)

add_a_dim_to_tensor(img)

print('img.shape=', img.shape)
print('img.rect_o=', img_coord.rect_o())
print('img.rect_o.size()=', img_coord.rect_o().size())
print('img_coord=', img_coord)

print('done')