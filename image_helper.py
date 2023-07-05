
import SimpleITK as sitk
import numpy as np
from image_coord import image_coord
from rect import rect

def save_image(img, path):
    print(f'save_image(img, {path})')
    sitk.WriteImage(img, path, True)  # useCompression:True


def read_image(path):
    print(f'read_image({path})')
    return sitk.ReadImage(path)


def sample_image(img_path, grid_coord, margin=3, defaultPixelValue=0, interpolator=sitk.sitkLinear, sampled_image_path=None):
    # print(
    #     f'sample_image({img_path}, {grid_coord}, {margin}, {defaultPixelValue}, {interpolator}, {sampled_image_path})')
    img = read_image_partial(img_path, grid_coord, margin)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(grid_coord.spacing.tolist())
    resample.SetSize(grid_coord.size.tolist())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(grid_coord.origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(defaultPixelValue)
    resample.SetInterpolator(interpolator)

    img_sampled = resample.Execute(img)
    #print_image_prop(img_sampled, 'img_sampled')

    # save image
    if sampled_image_path:
        save_image(img_sampled, sampled_image_path)

    return img_sampled


def read_image_partial(img_path, grid_coord, margin):

    #print(f'read_image_partial({img_path}, {grid_coord}, {margin})')

    # image file reader
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(img_path)

    # read the image information without reading the bulk data, compute ROI start and size and read it.
    file_reader.ReadImageInformation()

    img_coord_full = image_coord(size=file_reader.GetSize(
    ), origin=file_reader.GetOrigin(), spacing=file_reader.GetSpacing())

    # image index of the grid origin
    grid_org_imgI = img_coord_full.w2I(grid_coord.origin).astype(int)

    # grid size w.r.t image I
    grid_size_imgI = np.round(
        grid_coord.rect_o().size()/img_coord_full.spacing).astype(int)

    # grid rect in the image I coordinates
    grid_rect_I = rect(grid_org_imgI, grid_org_imgI+grid_size_imgI)
    #print('grid_rect_I=', grid_rect_I)

    # expand the grid rect by 3
    read_rect_I = grid_rect_I.expand(margin)
    #print('read_rect_I expanded=', read_rect_I)

    # make sure the read_rect is within the image rect
    read_rect_I = read_rect_I.intersect(img_coord_full.rect_I())
    #print('read_rect_I=', read_rect_I)
    start_index = read_rect_I.low
    extract_size = read_rect_I.size()

    #print('start_index=', start_index)
    #print('extract_size=', extract_size)

    file_reader.SetExtractIndex(start_index.tolist())
    file_reader.SetExtractSize(extract_size.tolist())

    img = file_reader.Execute()
    #print_image_prop(img, 'img (partial)')

    return img


def get_grid_list_to_cover_rect(organ_rect_w, grid_size, grid_spacing, n_border_pixels):

    valid_grid_size = 64 - n_border_pixels * 2
    valid_grid_size_mm = valid_grid_size * grid_spacing

    print('organ_rect_w=', organ_rect_w)
    print('organ_rect_w.size()=', organ_rect_w.size())

    print('num of blocks to inference (fraction) =',
          organ_rect_w.size()/valid_grid_size_mm)

    N_sub_images = np.array(
        np.floor(organ_rect_w.size()/valid_grid_size_mm)+[1.0]*3).astype(int)

    print('num of sub images =', N_sub_images)

    grid_coord_000 = None

    list = []

    for k in range(N_sub_images[2]):
        for j in range(N_sub_images[1]):
            for i in range(N_sub_images[0]):
                print('======================')
                print(f'rect_sub[{i}][{j}][{k}]')

                # a sub block in the organ u coordinate system
                sub_block_size_organ_u = [1.0] * 3 / N_sub_images
                print('sub_block_size=', sub_block_size_organ_u)
                center_organ_u = sub_block_size_organ_u / \
                    2.0 + sub_block_size_organ_u * [i, j, k]
                print('center_organ_u=', center_organ_u)

                # sub block ceneter in w coordiante system
                sub_block_center_w = organ_rect_w.low + center_organ_u * organ_rect_w.size()
                print('sub_block_center_w=', sub_block_center_w)

                # sample grid coordinate
                grid_org_w = sub_block_center_w - \
                    [grid_size * grid_spacing / 2.0]*3
                grid_coord = image_coord(origin=grid_org_w, size=[
                                         grid_size]*3, spacing=[grid_spacing]*3)
                print('grid_org_w=', grid_org_w)
                print('grid_coord=', grid_coord)

                # keep if ijk = 000
                if grid_coord_000 is None:
                    grid_coord_000 = grid_coord

                # get the nearist image pixel of the first grid (ijk=000)
                grid_org_wrt_grid000_I = grid_coord_000.w2I(grid_org_w)
                print('grid_org_wrt_grid000_I=', grid_org_wrt_grid000_I)

                # adjust the origin such that the grid orgin is aligned to first grid orgin
                grid_org_w = grid_coord_000.I2w(grid_org_wrt_grid000_I)
                grid_coord = image_coord(origin=grid_org_w, size=[
                                         grid_size]*3, spacing=[grid_spacing]*3)
                print('grid_org_w=', grid_org_w)
                print('grid_coord=', grid_coord)

                # tuple of grid_org_wrt_grid000_I,  grid_coord
                list.append((grid_coord, grid_org_wrt_grid000_I))
    return list


def extract_largest_connected_compoment(binary_input_image):

    # select the largest object
    connected_component_image = sitk.ConnectedComponent(binary_input_image)
    print('type(connected_components)=', type(connected_component_image))
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(connected_component_image)

    label_of_largest_object = -1
    n_pixels_of_largest_objects = 0
    for label in stats.GetLabels():
        print('================')
        print('label=', label)
        print('type(label)=', type(label))
        n_pixels = stats.GetNumberOfPixels(label)
        print('N pixels=', n_pixels)

        if n_pixels > n_pixels_of_largest_objects:
            n_pixels_of_largest_objects = n_pixels
            label_of_largest_object = label
    print('label_of_largest_object=', label_of_largest_object)
    print('n_pixels_of_largest_objects=', n_pixels_of_largest_objects)

    img_th = sitk.Cast(sitk.Threshold(connected_component_image,
                                      label_of_largest_object-0.5, label_of_largest_object+0.5, 0.0), sitk.sitkUInt8)

    return img_th
