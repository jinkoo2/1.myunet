from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian,generate_uid, PYDICOM_IMPLEMENTATION_UID 
from pydicom.sequence import Sequence
from typing import List, Union
import datetime
import os
from dataclasses import dataclass
import numpy as np
import cv2 as cv


COLOR_PALETTE= [
    [255, 0, 255],
    [0, 235, 235],
    [255, 255, 0],
    [255, 0, 0],
    [0, 132, 255],
    [0, 240, 0],
    [255, 175, 0],
    [0, 208, 255],
    [180, 255, 105],
    [255, 20, 147],
    [160, 32, 240],
    [0, 255, 127],
    [255, 114, 0],
    [64, 224, 208],
    [0, 178, 47],
    [220, 20, 60],
    [238, 130, 238],
    [218, 165, 32],
    [255, 140, 190],
    [0, 0, 255],
    [255, 225, 0]
]


class SOPClassUID:
    RTSTRUCT_IMPLEMENTATION_CLASS = PYDICOM_IMPLEMENTATION_UID  # TODO find out if this is ok
    DETACHED_STUDY_MANAGEMENT = '1.2.840.10008.3.1.2.3.1'
    RTSTRUCT = '1.2.840.10008.5.1.4.1.1.481.3'

@dataclass
class ROIData:
    """Data class to easily pass ROI data to helper methods."""
    mask: str
    color: Union[str, List[int]]
    number: int
    name: str
    frame_of_reference_uid: int
    description: str = ''
    use_pin_hole: bool = False
    approximate_contours: bool = True

    def __post_init__(self):
        self.validate_color()
        self.add_default_values()

    def add_default_values(self):
        if self.color is None:
            self.color = COLOR_PALETTE[(self.number - 1) % len(COLOR_PALETTE)]

        if self.name is None:
            self.name = f"ROI-{self.number}"

    def validate_color(self):
        if self.color is None:
            return

        # Validating list eg: [0, 0, 0]
        if type(self.color) is list:
            if len(self.color) != 3:
                raise ValueError(f'{self.color} is an invalid color for an ROI')
            for c in self.color:
                try:
                    assert 0 <= c <= 255
                except:
                    raise ValueError(f'{self.color} is an invalid color for an ROI')

        else:
            self.color: str = str(self.color)
            self.color = self.color.strip('#')

            # fff -> ffffff
            if len(self.color) == 3:
                self.color = ''.join([x*2 for x in self.color])

            if not len(self.color) == 6:
                raise ValueError(f'{self.color} is an invalid color for an ROI')

            try:
                self.color = [int(self.color[i:i+2], 16) for i in (0, 2, 4)]
            except Exception as e:
                raise ValueError(f'{self.color} is an invalid color for an ROI')

def load_dcm_images_from_path(dicom_series_path: str) -> List[Dataset]:
    series_data = []
    for root, _, files in os.walk(dicom_series_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file))
                if hasattr(ds, 'pixel_array'):
                    series_data.append(ds)

            except Exception:
                # Not a valid DICOM file
                continue

    return series_data

def get_file_meta() -> FileMetaDataset:
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SOPClassUID.RTSTRUCT
    file_meta.MediaStorageSOPInstanceUID = generate_uid() # TODO find out random generation is fine
    file_meta.ImplementationClassUID = SOPClassUID.RTSTRUCT_IMPLEMENTATION_CLASS
    return file_meta

    
def add_required_elements_to_ds(ds: FileDataset):
    dt = datetime.datetime.now()
    # Append data elements required by the DICOM standarad
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.InstanceCreationDate = dt.strftime('%Y%m%d')
    ds.InstanceCreationTime = dt.strftime('%H%M%S.%f')
    ds.StructureSetLabel = 'RTstruct'
    ds.StructureSetDate = dt.strftime('%Y%m%d')
    ds.StructureSetTime = dt.strftime('%H%M%S.%f')
    ds.Modality = 'RTSTRUCT'
    ds.Manufacturer = 'Qurit'
    ds.ManufacturerModelName = 'rt-utils'
    ds.InstitutionName = 'Qurit'
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    # Set values already defined in the file meta
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    ds.ApprovalStatus = 'UNAPPROVED'

def add_sequence_lists_to_ds(ds: FileDataset):
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()

def generate_base_dataset() -> FileDataset:
    file_name = 'rt-utils-struct'
    file_meta = get_file_meta()
    ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
    add_required_elements_to_ds(ds)
    add_sequence_lists_to_ds(ds)
    return ds

def add_study_and_series_information(ds: FileDataset, series_data):
    reference_ds = series_data[0] # All elements in series should have the same data
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, 'SeriesDate', '')
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, 'SeriesTime', '')
    ds.StudyDescription = getattr(reference_ds, 'StudyDescription', '')
    ds.SeriesDescription = getattr(reference_ds, 'SeriesDescription', '')
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid() # TODO: find out if random generation is ok
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = "1" # TODO: find out if we can just use 1 (Should be fine since its a new series)

def add_patient_information(ds: FileDataset, series_data):
    reference_ds = series_data[0] # All elements in series should have the same data
    ds.PatientName = getattr(reference_ds, 'PatientName', '')
    ds.PatientID = getattr(reference_ds, 'PatientID', '')
    ds.PatientBirthDate = getattr(reference_ds, 'PatientBirthDate', '')
    ds.PatientSex = getattr(reference_ds, 'PatientSex', '')
    ds.PatientAge = getattr(reference_ds, 'PatientAge', '')
    ds.PatientSize = getattr(reference_ds, 'PatientSize', '')
    ds.PatientWeight = getattr(reference_ds, 'PatientWeight', '')

def add_refd_frame_of_ref_sequence(ds: FileDataset, series_data):
    refd_frame_of_ref = Dataset()
    #FrameOfReferenceUID = getattr(series_data[0], 'FrameOfReferenceUID', generate_uid())
    #print('add_refd_frame_of_ref_sequence()->FrameOfReferenceUID=', FrameOfReferenceUID)
    refd_frame_of_ref.FrameOfReferenceUID =  getattr(series_data[0], 'FrameOfReferenceUID', generate_uid())
    refd_frame_of_ref.RTReferencedStudySequence = create_frame_of_ref_study_sequence(series_data)

    # Add to sequence
    ds.ReferencedFrameOfReferenceSequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence.append(refd_frame_of_ref)

def create_frame_of_ref_study_sequence(series_data) -> Sequence:
    reference_ds = series_data[0] # All elements in series should have the same data
    rt_refd_series = Dataset()
    rt_refd_series.SeriesInstanceUID = reference_ds.SeriesInstanceUID
    rt_refd_series.ContourImageSequence = create_contour_image_sequence(series_data)

    rt_refd_series_sequence = Sequence()
    rt_refd_series_sequence.append(rt_refd_series)

    rt_refd_study = Dataset()
    rt_refd_study.ReferencedSOPClassUID = SOPClassUID.DETACHED_STUDY_MANAGEMENT
    rt_refd_study.ReferencedSOPInstanceUID = reference_ds.StudyInstanceUID
    rt_refd_study.RTReferencedSeriesSequence = rt_refd_series_sequence

    rt_refd_study_sequence = Sequence()
    rt_refd_study_sequence.append(rt_refd_study)
    return rt_refd_study_sequence

def create_contour_image_sequence(series_data) -> Sequence:
    contour_image_sequence = Sequence()

    # Add each referenced image
    for series in series_data:
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = series.file_meta.MediaStorageSOPClassUID
        contour_image.ReferencedSOPInstanceUID = series.file_meta.MediaStorageSOPInstanceUID
        contour_image_sequence.append(contour_image)
    return contour_image_sequence

def get_contour_sequence_by_roi_number(ds, roi_number):
    for roi_contour in ds.ROIContourSequence:

        # Ensure same type
        if str(roi_contour.ReferencedROINumber) == str(roi_number):
            return roi_contour.ContourSequence

    raise Exception(f"Referenced ROI number '{roi_number}' not found")


def load_sorted_image_series(dicom_series_path: str):
    """
    File contains helper methods for loading / formatting DICOM images and contours
    """

    series_data = load_dcm_images_from_path(dicom_series_path)

    if len(series_data) == 0:
        raise Exception("No DICOM Images found in input path")

    # Sort slices in ascending order
    series_data.sort(key=lambda ds: ds.SliceLocation, reverse=False)

    return series_data


def load_dcm_images_from_path(dicom_series_path: str) -> List[Dataset]:
    series_data = []
    for root, _, files in os.walk(dicom_series_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file))
                if hasattr(ds, 'pixel_array'):
                    series_data.append(ds)

            except Exception:
                # Not a valid DICOM file
                continue

    return series_data


def get_contours_coords(mask_slice: np.ndarray, series_slice: Dataset, roi_data: ROIData):
    # Create pin hole mask if specified
    if roi_data.use_pin_hole:
        mask_slice = create_pin_hole_mask(mask_slice, roi_data.approximate_contours)

    # Get contours from mask
    contours, _ = find_mask_contours(mask_slice, roi_data.approximate_contours)
    validate_contours(contours)
    
    # Format for DICOM
    formatted_contours = []
    for contour in contours:
        contour = np.array(contour) # Type cannot be a list
        translated_contour = translate_contour_to_data_coordinants(contour, series_slice)
        dicom_formatted_contour = format_contour_for_dicom(translated_contour, series_slice)
        formatted_contours.append(dicom_formatted_contour)

    return formatted_contours 


def find_mask_contours(mask: np.ndarray, approximate_contours: bool):
    approximation_method = cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE 
    contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
    # Format extra array out of data
    for i, contour in enumerate(contours):
        contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
    hierarchy = hierarchy[0] # Format extra array out of data

    return contours, hierarchy



def create_pin_hole_mask(mask: np.ndarray, approximate_contours: bool):
    """
    Creates masks with pin holes added to contour regions with holes.
    This is done so that a given region can be represented by a single contour.
    """

    contours, hierarchy = find_mask_contours(mask, approximate_contours)
    pin_hole_mask = mask.copy()

    # Iterate through the hierarchy, for child nodes, draw a line upwards from the first point
    for i, array in enumerate(hierarchy):
        parent_contour_index = array[Hierarchy.parent_node]
        if parent_contour_index == -1: continue # Contour is not a child

        child_contour = contours[i]
        
        line_start = tuple(child_contour[0])

        pin_hole_mask = draw_line_upwards_from_point(pin_hole_mask, line_start, fill_value=0)
    return pin_hole_mask


def draw_line_upwards_from_point(mask: np.ndarray, start, fill_value: int) -> np.ndarray:
    line_width = 2
    end = (start[0], start[1] - 1)
    mask = mask.astype(np.uint8) # Type that OpenCV expects
    # Draw one point at a time until we hit a point that already has the desired value
    while mask[end] != fill_value:
        cv.line(mask, start, end, fill_value, line_width)

        # Update start and end to the next positions
        start = end
        end = (start[0], start[1] - line_width)
    return mask.astype(bool)


def validate_contours(contours: list):
    if len(contours) == 0:
        raise Exception("Unable to find contour in non empty mask, please check your mask formatting")


def translate_contour_to_data_coordinants(contour, series_slice: Dataset):
    offset = series_slice.ImagePositionPatient
    spacing = series_slice.PixelSpacing
    contour[:, 0] = (contour[:, 0]) * spacing[0] + offset[0]
    contour[:, 1] = (contour[:, 1]) * spacing[1] + offset[1]
    return contour


def translate_contour_to_pixel_coordinants(contour, series_slice: Dataset):
    offset = series_slice.ImagePositionPatient
    spacing = series_slice.PixelSpacing
    contour[:, 0] = (contour[:, 0] - offset[0]) / spacing[0]
    contour[:, 1] = (contour[:, 1] - + offset[1]) / spacing[1] 

    return contour


def format_contour_for_dicom(contour, series_slice: Dataset):
    # DICOM uses a 1d array of x, y, z coords
    z_indicies = np.ones((contour.shape[0], 1)) * series_slice.SliceLocation
    contour = np.concatenate((contour, z_indicies), axis = 1)
    contour = np.ravel(contour)
    contour = contour.tolist()
    return contour


def create_series_mask_from_contour_sequence(series_data, contour_sequence: Sequence):
    mask = create_empty_series_mask(series_data)

    # Iterate through each slice of the series, If it is a part of the contour, add the contour mask
    for i, series_slice in enumerate(series_data):
        slice_contour_data = get_slice_contour_data(series_slice, contour_sequence)
        if len(slice_contour_data):
            mask[:, :, i] = get_slice_mask_from_slice_contour_data(series_slice, slice_contour_data)
    return mask


def get_slice_contour_data(series_slice: Dataset, contour_sequence: Sequence):
    slice_contour_data = []
    
    # Traverse through sequence data and get all contour data pertaining to the given slice
    for contour in contour_sequence:
        for contour_image in contour.ContourImageSequence:
            if contour_image.ReferencedSOPInstanceUID == series_slice.SOPInstanceUID:
                slice_contour_data.append(contour.ContourData)

    return slice_contour_data


def get_slice_mask_from_slice_contour_data(series_slice: Dataset, slice_contour_data):
    slice_mask = create_empty_slice_mask(series_slice)
    for contour_coords in slice_contour_data:    
        fill_mask = get_contour_fill_mask(series_slice, contour_coords)
        # Invert values in the region to be filled. This will create holes where needed if contours are stacked on top of each other
        slice_mask[fill_mask == 1] = np.invert(slice_mask[fill_mask == 1])
    return slice_mask


def get_contour_fill_mask(series_slice: Dataset, contour_coords):
    # Format data
    reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
    translated_contour_data  = translate_contour_to_pixel_coordinants(reshaped_contour_data, series_slice)
    polygon = [np.array([translated_contour_data[:, :2]], dtype=np.int32)]

    # Create mask for the region. Fill with 1 for ROI
    fill_mask = create_empty_slice_mask(series_slice).astype(np.uint8)
    cv.fillPoly(img=fill_mask, pts=polygon, color=1)
    return fill_mask


def create_empty_series_mask(series_data):
    ref_dicom_image = series_data[0]
    mask_dims = (int(ref_dicom_image.Columns), int(ref_dicom_image.Rows), len(series_data))
    mask = np.zeros(mask_dims).astype(bool)
    return mask


def create_empty_slice_mask(series_slice):
    mask_dims = (int(series_slice.Columns), int(series_slice.Rows))
    mask = np.zeros(mask_dims).astype(bool)
    return mask



def create_structure_set_roi(roi_data: ROIData) -> Dataset:
    # Structure Set ROI Sequence: Structure Set ROI 1
    structure_set_roi = Dataset()
    structure_set_roi.ROINumber = roi_data.number
    structure_set_roi.ReferencedFrameOfReferenceUID = roi_data.frame_of_reference_uid
    structure_set_roi.ROIName = roi_data.name
    structure_set_roi.ROIDescription = roi_data.description
    structure_set_roi.ROIGenerationAlgorithm = 'MANUAL'
    return structure_set_roi


def create_roi_contour(roi_data: ROIData, series_data) -> Dataset:
    roi_contour = Dataset()
    roi_contour.ROIDisplayColor = roi_data.color
    roi_contour.ContourSequence = create_contour_sequence(roi_data, series_data)
    roi_contour.ReferencedROINumber = str(roi_data.number)
    return roi_contour


def create_contour_sequence(roi_data: ROIData, series_data) -> Sequence:
    """
    Iterate through each slice of the mask
    For each connected segment within a slice, create a contour
    """

    contour_sequence = Sequence()
    for i, series_slice in enumerate(series_data):
        mask_slice = roi_data.mask[:,:,i]
        # Do not add ROI's for blank slices
        if np.sum(mask_slice) == 0:
            print("Skipping empty mask layer")
            continue

        contour_coords = get_contours_coords(mask_slice, series_slice, roi_data)
        for contour_data in contour_coords:
            contour = create_contour(series_slice, contour_data)
            contour_sequence.append(contour)

    return contour_sequence


def create_contour(series_slice: Dataset, contour_data: np.ndarray) -> Dataset:
    contour_image = Dataset()
    contour_image.ReferencedSOPClassUID = series_slice.file_meta.MediaStorageSOPClassUID
    contour_image.ReferencedSOPInstanceUID = series_slice.file_meta.MediaStorageSOPInstanceUID

    # Contour Image Sequence
    contour_image_sequence = Sequence()
    contour_image_sequence.append(contour_image)

    contour = Dataset()
    contour.ContourImageSequence = contour_image_sequence
    contour.ContourGeometricType = 'CLOSED_PLANAR' # TODO figure out how to get this value
    contour.NumberOfContourPoints = len(contour_data) / 3  # Each point has an x, y, and z value
    contour.ContourData = contour_data

    return contour


def create_rtroi_observation(roi_data: ROIData) -> Dataset:
    rtroi_observation = Dataset()
    rtroi_observation.ObservationNumber = roi_data.number
    rtroi_observation.ReferencedROINumber = roi_data.number
    # TODO figure out how to get observation description
    rtroi_observation.ROIObservationDescription = 'Type:Soft,Range:*/*,Fill:0,Opacity:0.0,Thickness:1,LineThickness:2,read-only:false'
    rtroi_observation.private_creators = 'Qurit Lab'
    rtroi_observation.RTROIInterpretedType = ''
    rtroi_observation.ROIInterpreter = ''
    return rtroi_observation


class RTStruct:
    """
    Wrapper class to facilitate appending and extracting ROI's within an RTStruct
    """

    def __init__(self, series_data, ds: FileDataset):
        self.series_data = series_data
        self.ds = ds
        self.frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[-1].FrameOfReferenceUID  # Use last strucitured set ROI

    def set_series_description(self, description: str):
        """
        Set the series description for the RTStruct dataset
        """

        self.ds.SeriesDescription = description

    def add_roi(
        self,
        mask: np.ndarray,
        color: Union[str, List[int]] = None,
        name: str = None,
        description: str = '', 
        use_pin_hole: bool = False,
        approximate_contours: bool = True,
        ):
        """
        Add a ROI to the rtstruct given a 3D binary mask for the ROI's at each slice
        Optionally input a color or name for the ROI
        If use_pin_hole is set to true, will cut a pinhole through ROI's with holes in them so that they are represented with one contour
        If approximate_contours is set to False, no approximation will be done when generating contour data, leading to much larger amount of contour data
        """

        # TODO test if name already exists
        self.validate_mask(mask)
        roi_number = len(self.ds.StructureSetROISequence) + 1
        roi_data = ROIData(
            mask,
            color,
            roi_number,
            name,
            self.frame_of_reference_uid,
            description,
            use_pin_hole,
            approximate_contours
            )

        self.ds.ROIContourSequence.append(create_roi_contour(roi_data, self.series_data))
        self.ds.StructureSetROISequence.append(create_structure_set_roi(roi_data))
        self.ds.RTROIObservationsSequence.append(create_rtroi_observation(roi_data))

    def validate_mask(self, mask: np.ndarray) -> bool:
        if mask.dtype != bool:
            raise RTStruct.ROIException(f"Mask data type must be boolean. Got {mask.dtype}")

        if mask.ndim != 3:
            raise RTStruct.ROIException(f"Mask must be 3 dimensional. Got {mask.ndim}")

        if len(self.series_data) != np.shape(mask)[2]:
            raise RTStruct.ROIException(
                "Mask must have the save number of layers (In the 3rd dimension) as input series. " +
                f"Expected {len(self.series_data)}, got {np.shape(mask)[2]}"
            )

        if np.sum(mask) == 0:
            raise RTStruct.ROIException("Mask cannot be empty")

        return True

    def get_roi_names(self) -> List[str]:
        """
        Returns a list of the names of all ROI within the RTStruct
        """

        if not self.ds.StructureSetROISequence:
            return []

        return [structure_roi.ROIName for structure_roi in self.ds.StructureSetROISequence]

    def get_roi_mask_by_name(self, name) -> np.ndarray:
        """
        Returns the 3D binary mask of the ROI with the given input name
        """

        for structure_roi in self.ds.StructureSetROISequence:
            if structure_roi.ROIName == name:
                contour_sequence = get_contour_sequence_by_roi_number(self.ds, structure_roi.ROINumber)
                return create_series_mask_from_contour_sequence(self.series_data, contour_sequence)

        raise RTStruct.ROIException(f"ROI of name `{name}` does not exist in RTStruct")

    def save(self, file_path: str):
        """
        Saves the RTStruct with the specified name / location
        Automatically adds '.dcm' as a suffix
        """

        # Add .dcm if needed
        file_path = file_path if file_path.endswith('.dcm') else file_path + '.dcm'

        try:
            file = open(file_path, 'w')
            # Opening worked, we should have a valid file_path
            print("Writing file to", file_path)
            self.ds.save_as(file_path)
            file.close()
        except OSError:
            raise Exception(f"Cannot write to file path '{file_path}'")

    class ROIException(Exception):
        """
        Exception class for invalid ROI masks
        """

        pass