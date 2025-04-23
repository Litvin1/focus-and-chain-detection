#############################################################
# Authors: Dr. Daniel Dar and Vadim Litvinov
# Date: 1 September 2024
#############################################################
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as sks
import torch.cuda
from skimage.filters import threshold_otsu
import napari
import copy
import warnings
from skimage.measure import regionprops
from nd2_grabber import grab_nd2
from cellpose import models
import cv2
import pandas as pd
from tqdm import tqdm
from time import time
from scipy import ndimage
from numpy.typing import NDArray
from typing import Union, List, Tuple, Dict, Callable, Any


def useBrushToAddAndDrop(phase_image: Union[List, NDArray[int]], masks_relabel: Union[List, NDArray[int]],
                         focus_masks_from_algorithm: Union[List, NDArray[int]]) -> NDArray[int]:
    """
    Manually drop masks from focus masks or add masks to focus masks in Napari using brush
    @param phase_image: 3D phase image
    @param masks_relabel: all the masks that produced by the CNN
    @param focus_masks_from_algorithm: the focus masks that were found based on the properties
    @return: updated version of the focus masks
    """
    viewer = napari.Viewer(title=f'Correcting focus masks results')
    viewer.add_image(phase_image, name='phase', blending='additive', colormap='gray')
    focus_layer = viewer.add_labels(focus_masks_from_algorithm, name="Focus masks")
    masks_layer = viewer.add_labels(masks_relabel, name="All masks")
    focus_masks_proj = np.max(focus_masks_from_algorithm, axis=0)
    empty_seg1, empty_seg2 = np.zeros_like(focus_masks_from_algorithm), np.zeros_like(focus_masks_from_algorithm)
    add_layer = viewer.add_labels(empty_seg1, name='Manually added masks')  # type: ignore
    drop_layer = viewer.add_labels(empty_seg2, name='Manually dropped masks')  # type: ignore
    focus_masks_proj_layer = viewer.add_labels(focus_masks_proj, name='Focus masks projection')  # type: ignore

    # _... before unused parameter
    @add_layer.mouse_drag_callbacks.append
    def add_mask(_, event):
        """Inner function that assigns mask as focus mask"""
        if add_layer.mode == 'paint':
            coordinates = tuple(map(int, event.position))
            if masks_layer.data[coordinates] != 0:
                label_to_add = masks_relabel[coordinates]
                mask_location = (masks_relabel == label_to_add)
                #print(mask_location.shape, focus_layer.data.shape)
                focus_layer.data[mask_location] = label_to_add  # type: ignore
                focus_masks_from_algorithm[mask_location] = label_to_add
                flatten_proj_location = np.max(mask_location, axis=0)
                focus_masks_proj_layer.data[flatten_proj_location] = label_to_add  # type: ignore

    @drop_layer.mouse_drag_callbacks.append
    def drop_mask(_, event):
        """Inner function that removes mask from being a focus mask"""
        if drop_layer.mode == 'paint':
            coordinates = tuple(map(int, event.position))
            label_to_drop = focus_masks_from_algorithm[coordinates]
            mask_location = (focus_masks_from_algorithm == label_to_drop)
            focus_layer.data[mask_location] = 0  # type: ignore
            focus_masks_from_algorithm[mask_location] = 0
            flatten_proj_location = np.max(mask_location, axis=0)
            focus_masks_proj_layer.data[flatten_proj_location] = 0  # type: ignore

    napari.run()
    return focus_masks_from_algorithm


def addNewMetrics(mean_prob: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 2 metrics based on the pixel probabilities
    @param mean_prob: probabilities data structure
    @return: probabilities data structure with new metrics.
    """
    mean_prob['combined'] = (mean_prob['mean'] + mean_prob['max']) / 2
    mean_prob['SNR'] = mean_prob['mean'] / mean_prob['sd']
    # Add metrics for probabilities for non-mask 0
    mean_prob.loc[len(mean_prob)] = [0, -10, -10, -10, -10, -10, -10]
    return mean_prob


def GammaCorrection(img, gamma=1.0):
    image_8bit = cv2.convertScaleAbs((img - img.min()) * (255.0 / (img.max() - img.min())))
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    image_8bit_gamma = cv2.LUT(image_8bit, table)
    # Convert to 16bit
    image_16bit = np.uint16(np.round((image_8bit_gamma / 255.0) * 65535))
    return image_16bit


def NonLocalMeansDenoising(img, h=5, template_window_size=7, search_window_size=21):
    # Dummy
    s = h + template_window_size + search_window_size
    image_8bit = cv2.convertScaleAbs((img - img.min()) * (255.0 / (img.max() - img.min())))
    # Apply Non-Local Means Denoising on the 8-bit image
    denoised_image_8bit = cv2.fastNlMeansDenoising(image_8bit, None, h=3, templateWindowSize=7, searchWindowSize=21)
    # Adjust to 16 bit
    denoised_image_16bit = np.uint16(np.round((denoised_image_8bit / 255.0) * 65535))
    return denoised_image_16bit, s


def findHighestProbaSlice(proba_stack: NDArray) -> NDArray:
    """ Given all probability slices, find highest"""
    mean_values = np.mean(proba_stack, axis=2)
    max_mean_index = np.argmax(np.mean(mean_values, axis=1))
    return max_mean_index


def cancelSegmentation(masked_image: NDArray, segmentation_values_to_cancel: List) -> NDArray:
    """Turn mask to non mask"""
    # Create a mask to identify the segmentation values to cancel
    cancel_mask = np.isin(masked_image, segmentation_values_to_cancel)
    # Set the elements to 0 where the cancel mask is True
    masked_image[cancel_mask] = 0
    return masked_image


def slice3dImage(segmented_images: NDArray, cell_id: int) -> Tuple[NDArray, int, int]:
    """Given image and mask, slice 3D box around it"""
    z_start, z_end = 0, 0
    cell_slices = np.zeros((0, segmented_images.shape[1], segmented_images.shape[2]), dtype=segmented_images.dtype)
    for z_index in range(segmented_images.shape[0]):
        counter, z_start, z_end = 0, 0, 0
        z_slice = segmented_images[z_index]
        if np.any(z_slice == cell_id):
            if counter == 0:
                z_start = z_index
            cell_mask = (z_slice == cell_id)
            cell_image = np.zeros_like(z_slice)
            cell_image[cell_mask] = z_slice[cell_mask]
            cell_slices = np.concatenate((cell_slices, cell_image[np.newaxis, :, :]), axis=0)
        elif counter > 0:
            z_end = z_index
    return cell_slices, z_start, z_end


def checkIfNoOtherMaskSelectedInSpan(mask_sliced_region: np.ndarray, centroid_r: int, centroid_c: int, radius: int) \
        -> bool:
    """
    Given sliced 3d region, this function checks if there is any other mask in this region.
    @param mask_sliced_region: 3d sliced region
    @param centroid_r: row value of the centroid
    @param centroid_c: column value of the centroid
    @param radius: the radius around the centroid to look at
    @return: True or false
    """
    row_start, row_end, col_start, col_end, centroid_region = returnBoundariesForSlicedRegion(centroid_r, radius,
                                                                                              mask_sliced_region,
                                                                                              centroid_c)
    # Check if all elements in the region are zero
    return np.all(centroid_region == 0)


def findMaskSlice(seg_mask: NDArray, unkept_label: int) -> Union[int, None]:
    """This function finds the z of the mask"""
    for z_slice in range(seg_mask.shape[2]):
        # Check if the unkept_label is present in the current z-slice
        if unkept_label in seg_mask[:, :, z_slice]:
            return z_slice
    # If the unkept_label is not found in any z-slice, return None
    return None


def boundingBoxesToMasks(seg_mask: NDArray, zi: int) -> Dict:
    """Create mapping from label to mask properties"""
    zi_seg_mask = seg_mask[zi, :, :]
    props = regionprops(zi_seg_mask)
    bounding_boxes = {prop.label: prop.bbox for prop in props}
    return bounding_boxes


def keepLabel(seg_mask: NDArray, label_id: int, np_labels_to_keep: NDArray, labels_to_keep: List) -> None:
    """Given label to keep, assign it to data structures"""
    label_loc = (seg_mask == label_id)
    np_labels_to_keep[label_loc] = label_id
    labels_to_keep.append(label_id)


def find_most_in_focus_slice(image_stack, use_sharpness=True, use_darkness=False, sharpness_weight=1.0,
                             darkness_weight=1.0, focus_thresh=0):
    """
    Find the most in-focus slice in an image stack using sharpness and/or darkness measures,
    without using sub-functions for calculation of focus and darkness measures.
    Parameters:
    - image_stack: A numpy array of images (z, y, x).
    - use_sharpness: Boolean, whether to use sharpness measure.
    - use_darkness: Boolean, whether to use darkness measure.
    - sharpness_weight: Float, the weight of the sharpness measure in the combined score.
    - darkness_weight: Float, the weight of the darkness measure in the combined score.
    Returns:
    - Index of the most in-focus slice.
    """
    scores_sharpness = []
    scores_darkness = []
    for z_slice in image_stack:
        score_sharpness = 0.0
        score_darkness = 0.0
        if use_sharpness:
            # Calculate the variance of the Laplacian of the slice
            laplacian_var = cv2.Laplacian(z_slice, cv2.CV_64F).var()
            score_sharpness += sharpness_weight * laplacian_var
        if use_darkness:
            # Calculate the mean intensity (negated) of the slice
            mean_intensity = np.mean(z_slice)
            score_darkness += darkness_weight * (-mean_intensity)  # Negate to align with the focus measure convention
        scores_sharpness.append(score_sharpness)
        scores_darkness.append(score_darkness)
    sharpest_idx = np.argmax(scores_sharpness)
    if scores_sharpness[sharpest_idx] > focus_thresh:
        most_in_focus_index = sharpest_idx
    else:
        most_in_focus_index = -1
    return most_in_focus_index


def median_smooth_op_mat(optimal_layer_mat: NDArray, win_size: int = 128) -> NDArray:
    op = np.copy(optimal_layer_mat)
    for i in range(0, len(optimal_layer_mat), win_size):
        for j in range(0, len(optimal_layer_mat), win_size):
            # For each region, we want to look at the median value of neighboring tiles
            i_fr = max(i-win_size, 0)  # Above
            i_to = min(i+win_size*2, len(optimal_layer_mat)-1)  # below
            j_fr = max(j-win_size, 0)  # Left
            j_to = min(j+win_size*2, len(optimal_layer_mat)-1)  # right
            window_neighborhood = list(optimal_layer_mat[i_fr:i_to, j_fr:j_to].flatten())  # A list of
            # 9*win_size*win_size pixel values
            med_val = int(np.median(window_neighborhood))
            i_fr = i  # Above
            i_to = min(i+win_size, len(optimal_layer_mat)-1)  # Below
            j_fr = j  # Left
            j_to = min(j+win_size, len(optimal_layer_mat)-1)  # Right
            op[i_fr:i_to+1, j_fr:j_to+1] = med_val
    return op


def get_phase_projection(img, chunk_size=128, pixels_per_cell=20, is_show_proj=False, return_optimal_layer=False,
                         is_fluor=False, is_smooth=False, seg_mask=np.array([]), sig_percentile=0.95):
    """
    Written by Yedidya Ben-Eliyahu, 05.25.2022
    Purpose: Find the local best plane of focus in curved 3D phase images.
    Approach: Divide the image into windows (chunks) and find the optimal plate of focus for each.
    Generate a 2D projection using the intensity values from the chosen planes.
    input:
    - img_phase = multi-plane phase image
    - chunk_size = window pixel size for calculating median plane of focus; num must divide
    is_fluor - project on a single image
    Output:
    - phase projection = 2D corrected image
    """
    if not is_fluor:
        img_phase = img[:, 0, :, :]
        img_dapi = img[:, 1, :, :]
    else:
        img_phase = img[:, 0, :, :]
        img_dapi = img[:, 0, :, :]
    #F ind regions with objects to identify best z-section
    #Threhold phase
    img_min_int_proj = np.min(img_phase, axis=0) if not is_fluor else np.max(img_phase, axis=0)
    # Standard min intensity projection
    thresh = threshold_otsu(img_min_int_proj)  # Find the best threshold value
    img_thresh_bw = img_min_int_proj > thresh  # Background = 1, cells = 0
    if not is_fluor:
        img_thresh_bw = np.invert(img_thresh_bw)
    # Threshold dapi
    dapi_max_int_proj = np.max(img_dapi, axis=0)  # standard min intensity projection
    thresh_dapi = threshold_otsu(dapi_max_int_proj)  # find the best threshold value
    dapi_thresh_bw = dapi_max_int_proj > thresh_dapi  # background = 1, cells = 0
    # Intersect
    comb = dapi_thresh_bw.astype(int) + img_thresh_bw.astype(int)  # intersect
    comb = comb < 2
    # use seg mask? process for analysis of fluor data:  seg mask is supplied
    if seg_mask.size > 0:
        comb = seg_mask == 0
    img_min_int_zslice_mat = img_phase.argmin(axis=0).astype(float) if not is_fluor else img_phase.argmax(
        axis=0).astype(float)   # For each pixel, find the z-plane with min intensity
    if is_fluor:  # only look at the regions with high signal
        img_min_int_zslice_mat[img_min_int_zslice_mat < np.percentile(img_min_int_zslice_mat, sig_percentile)] = np.nan
    masked_layers = img_min_int_zslice_mat
    masked_layers[comb] = np.nan  # keep only the cell data, nan important as 0 vals mess with median calc downstream
    if is_fluor:
        masked_layers[dapi_max_int_proj < np.percentile(dapi_max_int_proj, sig_percentile)] = np.nan  # only look at
        # bright pixels
    img_size = masked_layers.shape[0]  # find the num of chunks
    optimal_layer_mat = np.zeros(masked_layers.shape)  # median layer values in chunk per pixel -> used for projection
    # downstream
    i = img_size
    windows = []
    while i > chunk_size - 1:
        windows.append(int(i))
        i = i / 2
    phase_projections = []
    opts = []
    op = []
    phase_projection = 0
    for window in windows:
        for i in range(0, img_size - 1, window):
            for j in range(0, img_size - 1, window):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    chunk_median_z_layer = np.nanmedian(masked_layers[i:i + window, j:j + window])
                    # nan insensitive median of entire chunk
                if np.isnan(chunk_median_z_layer) or np.count_nonzero(
                        ~np.isnan(masked_layers[i:i + window, j:j + window])) < pixels_per_cell:
                    chunk_median_z_layer = np.median(op[i:i + window, j:j + window])
                optimal_layer_mat[i:i + window, j:j + window] = chunk_median_z_layer
        phase_projection = np.take_along_axis(img_phase, optimal_layer_mat.astype(int)[np.newaxis], axis=0)[0]
        # save projection per window size
        pp = copy.deepcopy(phase_projection)
        phase_projections.append(pp)
        op = copy.deepcopy(optimal_layer_mat)
        opts.append(op)
    #average transition regions
    # downstream
    width = 10
    if is_smooth:
        num_of_rounds = 5
        if chunk_size > 128:
            num_of_rounds = 3
        for i in range(0, num_of_rounds):  # median smooth multiple times
            optimal_layer_mat = median_smooth_op_mat(optimal_layer_mat, win_size=chunk_size)
        optimal_layer_mat = optimal_layer_mat.astype(int)  # int
        # redo the phase projection
    # average seams
    opt1 = copy.deepcopy(optimal_layer_mat)  # copy.deepcopy(opts[-1])
    ph1 = copy.deepcopy(phase_projection)
    for i in range(chunk_size, img_size - chunk_size + 1, chunk_size):
        for j in range(chunk_size, img_size - chunk_size + 1, chunk_size):
            i_box = opts[-1][i - chunk_size: i, j - width: j + width]
            j_box = opts[-1][i - width: i + width, j - chunk_size: j]
            i_lay = np.sort(np.unique(i_box).astype(int))
            j_lay = np.sort(np.unique(j_box).astype(int))
            if i_lay[-1] - i_lay[0] > .1:
                i_slice = img_phase[i_lay[0]:i_lay[-1]+1, i - chunk_size: i, j - width: j + width]
                men = np.mean(i_slice, axis=0)
                ph1[i - chunk_size: i, j - width: j + width] = men
                opt1[i - chunk_size: i, j - width: j + width] = 14
            if j_lay[-1] - j_lay[0] > .1:
                j_slice = img_phase[j_lay[0]:j_lay[-1]+1, i - width: i + width, j - chunk_size: j]
                men = np.mean(j_slice, axis=0)
                ph1[i - width: i + width, j - chunk_size: j] = men
                opt1[i - width: i + width, j - chunk_size: j] = 14
    if is_show_proj:
        plt.imshow(opt1)
        plt.colorbar()
        plt.title('test')
        plt.show()
        viewer = napari.Viewer()
        viewer.add_image(phase_projection, name='old_projection')
        viewer.add_image(ph1, name='corrected(mean)')
        viewer.add_image(opts[-1], name='optimal_layer')  # type: ignore
        viewer.add_image(opt1, name='optimal_layer with borders')  # type: ignore
        viewer.add_image(img_phase, name='img_phase')
        viewer.show(block=True)
    return phase_projection, return_optimal_layer


def calculate_per_cell(img: NDArray, seg_mask: NDArray) -> Tuple[pd.DataFrame, float]:
    """Calculate different statistics for CellPose output probabilities"""
    start = time()
    # Find unique cell IDs and initialize a list to hold calculation results, maybe we will need it in the future.
    # Each cell has a few IDs, per slice.
    cell_ids = np.unique(seg_mask)
    cell_ids = cell_ids[cell_ids != 0]  # Exclude background if necessary
    calculations = []
    # Perform calculations for each cell
    for cell_id in tqdm(cell_ids, total=len(cell_ids), desc='Calculating probabilities statistics per cell mask'):
        cell_mask = seg_mask == cell_id
        # Choose only the current cell pixels
        cell_pixels = img[cell_mask]
        # Calculate statistics
        mean_signal = np.mean(cell_pixels)
        max_signal = np.max(cell_pixels)
        sd_signal = np.std(cell_pixels)
        median_signal = np.median(cell_pixels)
        # Append results for this cell
        calculations.append({
            'cell_id': cell_id,
            'mean': np.round(mean_signal, 2),
            'max': np.round(max_signal, 2),
            'sd': np.round(sd_signal, 2),
            'median': np.round(median_signal, 2)
        })
    # Convert calculations to DataFrame
    total_signal_df = pd.DataFrame(calculations)
    # Add case of background
    return total_signal_df, start


def relabelMasks(masks: NDArray) -> NDArray:
    """Assign new unique names for each mask in all z slices"""
    # Generate independently labeled mat
    masks_relabel = np.zeros_like(masks)
    offset = 0
    for zi in tqdm(range(masks.shape[0]), total=masks.shape[0], desc='Relabel per z-slice'):
        labels_zi = sorted([x for x in np.unique(masks[zi, :, :]) if x != 0])
        for label in labels_zi:  # tqdm(labels_zi, total=len(labels_zi), desc='labels in zi'):
            indices = np.where(masks[zi, :, :] == label)
            masks_relabel[zi, indices[0], indices[1]] = label + offset
            # break
        offset = offset + len(labels_zi)  # To give unique name for each cell in each slice
    print('Found uniques: ', len(np.unique(masks_relabel)))
    return masks_relabel


def loadNd2SaveTiff(data_dir: str, fov: int, hyb: int, img_slice: List) -> NDArray:
    """Load microscope file and save .tif file"""
    img = grab_nd2(img_path=fr'{data_dir}/fov_{fov}_hyb_{hyb}.nd2',
                   channels_to_grab=['phase'])  # ['A488','dapi'] DIC
    img = img[:, :, img_slice[0]:img_slice[1], img_slice[2]:img_slice[3]]  # z (depth),c (channel),x,y
    return img


def loadImgTrainModel(fov: int, hyb: int, data_dir: str, img_slice: List) \
        -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Load image, fix seed for reproducibility, define model and predict segmentation masks"""
    for fov in range(fov, fov + 1):
        img = loadNd2SaveTiff(data_dir, fov, hyb, img_slice)
        # params = {'channels':[0,1], # always define this with the model
        #     'rescale': None, # upscale or downscale your images, None = no rescaling
        #     'mask_threshold': 0, # erode or dilate masks with higher or lower values
        #     'flow_threshold': 0, # default is .4, but only needed if there are spurious masks to clean up; slows
        #     down output
        #     'transparency': True, # transparency in flow output
        #     'omni': False, # we can turn off Omnipose mask reconstruction, not advised
        #     'cluster': True, # use DBSCAN clustering
        #     'resample': True, # whether or not to run dynamics on rescaled grid or original grid
        #     # 'verbose': False, # turn on if you want to see more output
        #     'tile': False, # average the outputs from flipped (augmented) images; slower, usually not needed
        #     'niter': None, # None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it
        #     for over/under segmentation
        #     'augment': False, # Can optionally rotate the image and average outputs, usually not needed
        #     'affinity_seg': False, # new feature, stay tuned...
        #     'net_avg' : True,
        #     'gpu' : True,
        #     # 'do_3D' : True,
        #     # 'stitch_threshold' : 0.4,
        #     }
        # Ensure there is no stochasticity
        # Produce the same "random" output each runm in random and in numpy
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        '''
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            print('Warning: cudnn attributes not available. Ensure CUDA is installed for GPU determinism.')
        '''
        cellpose_model = models.CellposeModel(model_type='cyto3', gpu=True, nchan=2)      # pretrained_model=model_path
        print('Using GPU for parallelization in model:', torch.cuda.is_available())
        #print(torch.cuda.device_count())
        #print(torch.cuda.current_device())
        # masks: segmentation of pixels, flows:
        # (list of lists 2D arrays or list of 3D arrays):
        # flows[k][0] = XY flow in HSV 0-255
        # flows[k][1] = XY flows at each pixel
        # flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
        # cross entropy, this is why they negative [log(p)]
        # flows[k][3] = final pixel locations after Euler integration
        # , styles: rows - slices, columns - vector of 256 colors summarizing?
        # styles (list of 1D arrays of length 256 or single 1D array): Style vector summarizing each image, also used
        # to estimate size of objects in image.
        masks, model_flows, model_styles = cellpose_model.eval(img, diameter=cellpose_model.diam_mean,
                                                               flow_threshold=0.0, cellprob_threshold=0.90,
                                                               do_3D=False, augment=True, stitch_threshold=0,
                                                               channels=[1, 2], normalize={'norm3D': False})
        # channels=None for nuclei,
        # cellprob_threshold = DECREASE to find more and larger masks
        # flow_threshold = all cells with errors below threshold are kept. INCREASE to find more
        # dapi for cytoplasm
        print(f'Found {len(np.unique(masks)) - 1} masks in FOV before relabeling')
        proba = model_flows[2]
        return img, masks, model_flows, model_styles, proba


def relabelMasksConsecutive(masked_array: np.ndarray) -> NDArray:
    """Given non-consecutive order mask names, fix it"""
    print('Renaming the masks in consecutive order...')
    # Get unique labels, excluding 0 (background)
    unique_labels = []
    for z in masked_array:
        z_unique_labels = np.unique(z)
        z_unique_labels = list(z_unique_labels[z_unique_labels != 0])  # Exclude background (assumed to be 0)
        if len(z_unique_labels) > 0:
            unique_labels.extend(z_unique_labels)
    relabeled_array = masked_array.copy()
    # Apply the new labels
    for idx, old_label in enumerate(unique_labels):
        new_label = idx + 1
        relabeled_array[relabeled_array == old_label] = new_label
    return relabeled_array


def createFocusMasks(masks_relabel: NDArray, labels_to_keep: List) -> NDArray:
    """Delete masks that are not focus mask"""
    focus_masks_to_save = np.copy(masks_relabel)
    focus_masks_to_save[~np.isin(focus_masks_to_save, labels_to_keep)] = 0
    return focus_masks_to_save


def loadFocusMasks(focus_masks_path: str) -> NDArray:
    """loads the focus masks"""
    print('Loading focus masks...')
    loaded_focus_masks = np.load(focus_masks_path)
    return loaded_focus_masks


def showInNapari(img: Union[NDArray, List], focus_masks_for_napari: Union[NDArray, List],
                 filtered_masks: Union[NDArray, List], masks_relabel: NDArray,
                 proba: Union[NDArray, List]) -> None:
    """Show cells in Napari viewer"""
    focus_masks_proj = np.max(focus_masks_for_napari, axis=0)
    viewer = napari.Viewer()
    viewer.add_image(proba, blending='additive', name='img', colormap='twilight_shifted')
    viewer.add_image(img, blending='additive', name='DIC')
    viewer.add_labels(focus_masks_proj, blending='additive', name='focus_masks_proj')  # type: ignore
    viewer.add_labels(filtered_masks, blending='additive', name='filtered _masks')
    viewer.add_labels(masks_relabel, blending='additive', name='masks_relab')  # type: ignore
    viewer.add_labels(focus_masks_for_napari, blending='additive', name='focus_masks')
    napari.run()


def detectPerimeterByErosionAndCalculateMeanIntensity(img: NDArray, segmentation: NDArray, mask_name: int,
                                                      kernel_size: int) -> float:
    """Calculate the mean perimeter intensity in phase"""
    binary_mask = (segmentation == mask_name)
    # Create the kernel for erosion
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Erode the image to shrink the object
    eroded_mask = ndimage.binary_erosion(binary_mask, structure=kernel)
    # Detect the perimeter by subtracting the eroded image from the original
    mask_binary_perimeter = binary_mask ^ eroded_mask
    image_perimeter = np.where(mask_binary_perimeter, img, 0)
    mean_perimeter_intensity = image_perimeter[image_perimeter != 0].mean()
    return mean_perimeter_intensity


def meanMaskIntensity(img: NDArray, segmentation: NDArray, label: int) -> NDArray:
    """Calculate mean intensity of phase in the mask"""
    labeled_region = (segmentation == label)
    mask_pixels = img[labeled_region]
    return np.mean(mask_pixels)


def assureProbabilityThresholds(probability_df: pd.DataFrame, focus_mask: int, operator: Callable,
                                prob_thresh_snr: float, prob_thresh_median: float, prob_thresh_combined: float) -> int:
    """Check if the CellPose output probabilities in the pixels of the mask are above some thresholds"""
    mask_proba_combined = probability_df.loc[probability_df['cell_id'] == focus_mask, 'combined'].values[0]
    mask_proba_median = probability_df.loc[probability_df['cell_id'] == focus_mask, 'median'].values[0]
    mask_proba_snr = probability_df.loc[probability_df['cell_id'] == focus_mask, 'SNR'].values[0]
    if not operator([mask_proba_snr > prob_thresh_snr,
                     mask_proba_median > prob_thresh_median,
                     mask_proba_combined > prob_thresh_combined]):
        focus_mask = 0
    return focus_mask


def vegetativeCellProcedure(focus_mask: int, kept_masks_sliced_region: NDArray, sliced_centroid_r: int,
                            sliced_centroid_c: int, radius: int, arr_labels_to_keep: NDArray, labels_to_keep: List,
                            masks: NDArray) -> None:
    """Check that there is no other mask in span. If so, keep the label"""
    check1 = checkIfNoOtherMaskSelectedInSpan(kept_masks_sliced_region, sliced_centroid_r,
                                              sliced_centroid_c, radius)
    if focus_mask != 0 and check1:
        keepLabel(masks, focus_mask, arr_labels_to_keep, labels_to_keep)


def initializeVariables(masks: NDArray, edge_z: int) -> Tuple[List, NDArray, int]:
    """Initialize lists, numpy arrays, and the depth dimensions"""
    labels_to_keep = []
    arr_labels_to_keep = np.zeros_like(masks)
    z_max = masks.shape[0]
    # No need for looping over the extreme z slices
    if edge_z > 0:
        masks[:edge_z, :, :] = 0
        masks[-edge_z:, :, :] = 0
    return labels_to_keep, arr_labels_to_keep, z_max


def filterByProba(masks: NDArray, proba_df: pd.DataFrame, operator: str, combined_min: float, snr_min: float,
                  median_min: float, start: float) -> NDArray:
    """filter bad low probability masks"""
    def check_conditions(row: pd.DataFrame) -> bool:
        conditions = [row['combined'] > combined_min,
                      row['SNR'] > snr_min,
                      row['median'] > median_min]
        return all(conditions) if operator == 'and' else any(conditions)

    print('Filtering by probability statistics...')
    valid_mask = proba_df[proba_df.apply(check_conditions, axis=1)]['cell_id'].values
    valid_mask_set = set(valid_mask)
    mask_indices = np.isin(masks, list(valid_mask_set))
    masks = np.where(mask_indices, masks, 0)
    end = time()
    print(len(np.unique(masks)) - 1, 'masks after probability filtering')
    print('Filtering by probability time:', end - start, '\n')
    return masks


def returnBoundariesForSlicedRegion(centroid_r: int, radius: int, masks_sliced_region: NDArray, centroid_c: int) \
        -> Tuple[int, int, int, int, NDArray]:
    """Calculate the boundaries of the sliced region"""
    row_start = max(centroid_r - radius, 0)
    row_end = min(centroid_r + radius + 1, masks_sliced_region.shape[1])
    col_start = max(centroid_c - radius, 0)
    col_end = min(centroid_c + radius + 1, masks_sliced_region.shape[2])
    centroid_region = masks_sliced_region[:,
                                          row_start:row_end,
                                          col_start:col_end]
    return row_start, row_end, col_start, col_end, centroid_region


def reduceLayersWithoutMasks(img_sliced_region: NDArray, masks_sliced_region: NDArray, centroid_r: int, centroid_c: int,
                             radius: int) -> Tuple[NDArray, NDArray]:
    """Cancel z slices that have no segmentation masks in them"""
    row_start, row_end, col_start, col_end, centroid_region = returnBoundariesForSlicedRegion(centroid_r, radius,
                                                                                              masks_sliced_region,
                                                                                              centroid_c)
    valid_z = np.any(centroid_region != 0, axis=(1, 2))
    filtered_img_sliced_region = np.zeros_like(img_sliced_region)
    filtered_img_sliced_region[valid_z] = img_sliced_region[valid_z]
    return filtered_img_sliced_region, masks_sliced_region


def filterExtremelySmallAndLarge(masks: NDArray, min_size: int, max_size: int) -> Tuple[NDArray, Dict]:
    """Drop masks that are smaller than minimum or larger them maximum"""
    start = time()
    props = regionprops(masks)
    areas = {prop.label: prop.area for prop in props}
    for mask, area in tqdm(areas.items(), desc='Filtering extremely small and large masks'):
        if area < min_size or area > max_size:
            masks[masks == mask] = 0
    print(len(np.unique(masks)), 'masks after size filtering')
    props = regionprops(masks)
    areas = {prop.label: prop.area for prop in props}
    areas[0] = np.inf
    end = time()
    print('Filtering by size time:', end - start, '\n')
    return masks, areas


def calculateEccentricity(masks: NDArray) -> Dict:
    """Calculate the roundness of the mask"""
    general_eccentricity = {}
    for z in tqdm(range(masks.shape[0]), desc='Calculating eccentricity'):
        props = regionprops(masks[z])
        eccentricity = {prop.label: prop.eccentricity for prop in props}
        general_eccentricity.update(eccentricity)
    return general_eccentricity


def filterByEccentricity(masks: NDArray, max_e: float) -> NDArray:
    """Drop masks that are not round enough"""
    start = time()
    eccentricity = calculateEccentricity(masks)
    for mask, e in tqdm(eccentricity.items(), desc='Filtering by eccentricity'):
        if e > max_e:
            masks[masks == mask] = 0
    end = time()
    print(len(np.unique(masks)), 'masks after eccentricity filtering')
    print('Filtering by eccentricity time:', end - start, '\n')
    return masks


def filterByRatioInteriorToPerimeterMeanIntensity(img: NDArray, masks: NDArray, min_interior_perimeter_ratio: float,
                                                  kernel_size: int) -> NDArray:
    """Drop masks of cells that don't have ring darker than interior (implies focus)"""
    start = time()
    eroded_img = np.zeros_like(img)
    label_to_perimeter_mean_intensity = {}
    for z in tqdm(range(masks.shape[0]), desc='Calculating mean interior and mean perimeter intensities'):
        masks_2d = masks[z]
        img_2d = img[z]
        labels = np.unique(masks_2d)
        labels = labels[labels != 0]
        for label in labels:
            mask = (masks_2d == label)
            kernel = np.ones((kernel_size, kernel_size), bool)
            eroded_mask = ndimage.binary_erosion(mask, structure=kernel, iterations=1)
            eroded_img[z, eroded_mask] = img_2d[eroded_mask]
            perimeter_mean_intensity = detectPerimeterByErosionAndCalculateMeanIntensity(img_2d, masks_2d,
                                                                                         label,
                                                                                         kernel_size)
            label_to_perimeter_mean_intensity[label] = perimeter_mean_intensity
    eroded_props = regionprops(masks, intensity_image=eroded_img)
    label_to_interior_mean_intensity = {prop.label: prop.intensity_mean for prop in eroded_props}
    for mask in tqdm(label_to_perimeter_mean_intensity.keys(), desc='Filtering by interior to perimeter intensity'
                                                                    ' ratio'):
        interior = label_to_interior_mean_intensity[mask]
        perimeter = label_to_perimeter_mean_intensity[mask]
        ratio = interior / perimeter
        if ratio < min_interior_perimeter_ratio:
            masks[masks == mask] = 0
    end = time()
    print(len(np.unique(masks)), 'masks after perimeter to interior mean intensity ratio filtering')
    print('Filtering by intensity ratio time:', end - start, '\n')
    return masks


def createSlicedRegionForLabel(masks: NDArray, bounding_boxes: Dict, label: int, img_phase: NDArray, z, z_span: int,
                               z_max: int, xy_span: int) -> Tuple[NDArray, NDArray, int, int, int, int, int, int, int,
                                                                  int]:
    """Slice a bounding 3D box of the label in both phase image and the masks data structures"""
    min_row, min_col, max_row, max_col = bounding_boxes[label]
    boundary_row, boundary_col = img_phase.shape[1], img_phase.shape[2]
    # Create sliced region and find its most in focus slice, and take the mask that is there
    img_sliced_region = img_phase[max(z - z_span, 0):min(z + z_span + 1, z_max),
                                  max(min_row - xy_span, 0):min(max_row + xy_span, boundary_row),
                                  max(min_col - xy_span, 0):min(max_col + xy_span, boundary_col)]
    masks_sliced_region = masks[max(z - z_span, 0):min(z + z_span + 1, z_max),
                                max(min_row - xy_span, 0):min(max_row + xy_span, boundary_row),
                                max(min_col - xy_span, 0):min(max_col + xy_span, boundary_col)]
    sliced_centroid_r = round((masks_sliced_region.shape[1]) / 2)
    sliced_centroid_c = round((masks_sliced_region.shape[2]) / 2)
    return img_sliced_region, masks_sliced_region, sliced_centroid_r, sliced_centroid_c, min_row, max_row,\
        boundary_row, min_col, max_col, boundary_col


def DifferentiateFocusLayer(proba: NDArray, img_phase: NDArray, masks: NDArray, z_span: int, xy_span: int,
                            radius: int, prob_thresh_snr: float, prob_thresh_median: float,
                            prob_thresh_combined: float, operator: str) -> Tuple[List, NDArray]:
    """Find the focus layer for each cell"""
    labels_to_keep, arr_labels_to_keep, z_max = initializeVariables(masks, edge_z=0)
    masks = filterByEccentricity(masks, max_e=2.0)
    masks, general_areas = filterExtremelySmallAndLarge(masks, min_size=300, max_size=np.inf)
    mean_prob_df, start_time = calculate_per_cell(proba, masks)
    mean_prob_df = addNewMetrics(mean_prob_df)
    masks = filterByProba(masks, mean_prob_df, operator, prob_thresh_combined, prob_thresh_snr,
                          prob_thresh_median, start_time)
    masks = filterByRatioInteriorToPerimeterMeanIntensity(img_phase, masks,
                                                          min_interior_perimeter_ratio=1.0,
                                                          kernel_size=2)
    for z in tqdm(range(0, z_max), desc='Looping over z slices and its masks'):
        bounding_boxes = boundingBoxesToMasks(masks, z)
        for label, bbox in bounding_boxes.items():
            img_sliced_region, masks_sliced_region, sliced_centroid_r, sliced_centroid_c, min_row, max_row, \
                boundary_row, min_col, max_col, boundary_col = createSlicedRegionForLabel(masks, bounding_boxes, label,
                                                                                          img_phase, z, z_span, z_max,
                                                                                          xy_span)
            img_sliced_region, masks_sliced_region = reduceLayersWithoutMasks(img_sliced_region, masks_sliced_region,
                                                                              sliced_centroid_r, sliced_centroid_c,
                                                                              radius)
            sliced_region_focus_z = find_most_in_focus_slice(img_sliced_region)
            general_focus_z = max(z-z_span, 0) + sliced_region_focus_z
            # To find the general focus z, we need to add to sliced_region_focus_z the leftover from the beginning,
            # which is the bottom boundary of the z sliding window
            kept_masks_sliced_region = arr_labels_to_keep[max(general_focus_z-z_span, 0):
                                                          min(general_focus_z+z_span+1, z_max),
                                                          max(min_row-xy_span, 0):
                                                          min(max_row+xy_span, boundary_row),
                                                          max(min_col-xy_span, 0):
                                                          min(max_col+xy_span, boundary_col)]
            focus_mask = masks_sliced_region[sliced_region_focus_z, sliced_centroid_r, sliced_centroid_c] if\
                sliced_region_focus_z != -1 else 0
            vegetativeCellProcedure(focus_mask, kept_masks_sliced_region, sliced_centroid_r, sliced_centroid_c, radius,
                                    arr_labels_to_keep, labels_to_keep, masks)
    print(len(labels_to_keep), 'masks kept after first round')
    return labels_to_keep, masks


def saveModelSegmentationsAndProps(masks_unfilt: NDArray, masks_final: Union[None, NDArray],
                                   props_df: Union[None, pd.DataFrame], proba: Union[NDArray, None],
                                   fov: int, hyb: int, to_final_folder: bool, dir_name: str) -> None:
    """Save the focus masks to file"""
    np.save(f'{dir_name}\\fov_{fov}_hyb_{hyb}.seg.unfilt.npy', masks_unfilt)
    if to_final_folder:
        print('Saving .seg and props file...')
        np.save(f'{dir_name}\\fov_{fov}_hyb_{hyb}.seg.npy', masks_final)
        props_df.to_csv(f'{dir_name}\\fov_{fov}_hyb_{hyb}.props.txt', sep='\t', index=False)
    # Else save to the current directory, it is not final for the analysis folder on Wexac
    else:
        np.save(f'{dir_name}\\probabilities_' + str(fov) + '.npy', proba)


def loadModelSegmentations(filename: str, fov: int) -> Tuple[NDArray, NDArray]:
    """Load CellPose outputs"""
    print('Loading model segmentations and probabilities...')
    loaded_masks = np.load(f'{filename}\\{filename}.seg.unfilt.npy')
    loaded_probabilities = np.load('probabilities_' + str(fov) + '.npy', allow_pickle=True)
    return loaded_masks, loaded_probabilities


def getRegionProps(seg_mask: Union[None, NDArray], file_name: Union[None, str], load: bool) -> pd.DataFrame:
    """Load, or create region properties for masks, including a few circularity properties"""
    if load:
        properties_df_merged = pd.read_csv(file_name, sep='\t')
        properties_df_merged['centroid'] = properties_df_merged['centroid'].apply(
            lambda x: tuple(map(float, x.strip('()').split(','))))
    else:
        properties_df_merged = pd.DataFrame()
        for z in tqdm(range(seg_mask.shape[0]), desc='Calculating region properties'):
            seg_mask_z = seg_mask[z, :, :]
            cell_props = regionprops(seg_mask_z)
            props = ['label', 'area', 'perimeter', 'perimeter_crofton',
                     'eccentricity', 'major_axis_length',
                     'minor_axis_length', 'centroid', 'orientation', 'solidity', 'convex_area']  # Base properties
            data = []
            for prop in cell_props:
                cell_data = {p: getattr(prop, p) for p in props}
                # Add z-slice information to the cell data
                cell_data['z_slice'] = z
                mean_radius, sd_radius = calculateRadiusDistributionMeasures(prop, seg_mask_z)
                cell_data['mean_radius'] = mean_radius
                cell_data['SD_radius'] = sd_radius
                cell_data['Haralick_circularity'] = mean_radius / sd_radius
                data.append(cell_data)
            prop_df = pd.DataFrame(data)
            properties_df_merged = pd.concat([properties_df_merged, prop_df])
        properties_df_merged['circularity'] = \
            (4*np.pi*properties_df_merged['area']) / (properties_df_merged['perimeter']**2)
        properties_df_merged['heterocyst'] = np.nan
        properties_df_merged.sort_values(by=['label'], inplace=True)
    return properties_df_merged


def calculateRadiusDistributionMeasures(region: Any, cell_masks: NDArray) -> Tuple[float, float]:
    """Calculate measures for Haralick-Circularity"""
    bool_cell_mask = (cell_masks == region.label)
    centroid_row, centroid_col = region.centroid
    bool_boundaries_mask = sks.find_boundaries(bool_cell_mask, mode='inner', background=0)
    bound_rows_idx, bound_cols_idx = np.where(bool_boundaries_mask)
    a_s = (centroid_row - bound_rows_idx)**2
    b_s = (centroid_col - bound_cols_idx)**2
    a_s_plus_b_s = a_s + b_s
    radii = np.sqrt(a_s_plus_b_s)
    radius_mean, radius_sd = radii.mean(), radii.std()
    return radius_mean, radius_sd


def addHeterocystColumn(img: Union[NDArray, List], masks_in_focus: Union[NDArray, List], props: pd.DataFrame)\
        -> pd.DataFrame:
    """Open Napari viewer, and assign labels of heterocyst or vegetative to masks"""
    viewer = napari.Viewer(title='Manual labeling of Hetrocysts')
    viewer.add_image(img, name='Phase', blending='additive', colormap='gray')
    focus_layer = viewer.add_labels(masks_in_focus, name='Focus masks')
    empty_arr1, empty_arr2 = np.zeros_like(masks_in_focus), np.zeros_like(masks_in_focus)
    heterocyst_layer = viewer.add_labels(empty_arr1, name='Heterocysts')  # type: ignore
    vegetative_layer = viewer.add_labels(empty_arr2, name='Vegetative')  # type: ignore
    h_count, v_count = 0, 0

    @heterocyst_layer.mouse_drag_callbacks.append
    def labelHeterocyst(_, event):
        """Add clicked mask to Heterocyst list"""
        nonlocal h_count
        if heterocyst_layer.mode == 'paint':
            coordinates = tuple(map(int, event.position))
            if focus_layer.data[coordinates] != 0:
                heterocyst_mask = focus_layer.data[coordinates]
                props.loc[props['label'] == heterocyst_mask, 'heterocyst'] = 1
                h_count += 1
                print(h_count, 'heterocysts labeled')

    @ vegetative_layer.mouse_drag_callbacks.append
    def labelVegetative(_, event):
        """Add clicked mask to Vegetative list"""
        nonlocal v_count
        if vegetative_layer.mode == 'paint':
            coordinates = tuple(map(int, event.position))
            if focus_layer.data[coordinates] != 0:
                vegetative_mask = focus_layer.data[coordinates]
                props.loc[props['label'] == vegetative_mask, 'heterocyst'] = 0
                v_count += 1
                print(v_count, 'vegetative labeled')

    napari.run()
    return props
