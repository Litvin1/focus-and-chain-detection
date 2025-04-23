#############################################################
# Author: Vadim Litvinov
# Date: 31 July 2024
#############################################################
import os
import pickle
import math
import time
import napari
import numpy as np
import networkx as nx
import pandas as pd
from nd2_grabber import grab_nd2
from tqdm import tqdm
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, skeletonize
#from focus_differentiator import FILE_NAME, DATA_DIR, FOCUS_MASKS_PATH, IMG_SLICE, HYB, FOV, getRegionProps
from scipy.spatial import distance
from typing import List, Dict, Tuple, Union
#from numpy.typing import NDArray
from qtpy.QtWidgets import QApplication
import sys


def openMasksFile(focus_masks_path: str) -> np.ndarray:
    """Loads the focus masks file."""
    cwd = os.getcwd()
    # Full path to the subdirectory
    target_dir = cwd + '\\' + focus_masks_path
    return np.load(target_dir)


def checkPixelNumInNeighborhood(sliced_region: np.ndarray, neighbors: np.ndarray, pixel_thresh: int) -> List:
    """Assign neighbor relation if there is more lan some threshold of pixels"""
    properties = dictOfProps(sliced_region)
    neighbors = list(filter(lambda neighbor: properties[neighbor].area > pixel_thresh, neighbors))
    return neighbors


def findNeighborsInImage(graph: nx.Graph, segmentations: np.ndarray, row_dim: int, col_dim: int, depth_dim: int, z: int,
                         bounding_boxes: Dict, z_span: int, xy_span: int, pixel_thresh: int) -> None:
    """Convert neighboring cells in Image to Graph representation"""
    z_min = max(0, z-z_span)
    z_max = min(depth_dim, z + z_span)
    for label, bbox in bounding_boxes.items():
        graph.add_node(label)
        min_row, min_col, max_row, max_col = bbox
        # Enlarge the bounding box by 1
        min_row, min_col = max(0, min_row-xy_span), max(0, min_col-xy_span)
        max_row, max_col = min(row_dim, max_row + xy_span), min(col_dim, max_col + xy_span)
        sliced_region = segmentations[z_min:z_max + 1,
                                      min_row:max_row,
                                      min_col:max_col]
        neighbors = np.unique(sliced_region)
        self_background_indexes = np.where(np.isin(neighbors, [0, label]))
        neighbors = np.delete(neighbors, self_background_indexes)
        neighbors = checkPixelNumInNeighborhood(sliced_region, neighbors, pixel_thresh)
        # We do not want edge from the cell to itself or to background
        edges = [(label, neighbor) for neighbor in neighbors]
        graph.add_edges_from(edges)


def dropShortConnectedComponent(graph: nx.Graph, size: int = 2) -> nx.Graph:
    """Do not use any component which is smaller than "size" variable"""
    print('Dropping non-chains connected components...')
    connected_components = list(nx.connected_components(graph))
    nodes_to_remove = set()
    for component in connected_components:
        if len(component) <= size:
            nodes_to_remove.update(component)
    graph.remove_nodes_from(nodes_to_remove)
    return graph


def findUniqueCellsInSlice(slice_mask: np.ndarray) -> np.ndarray:
    """Return all masks in slice (without background)"""
    unique_cells = np.unique(slice_mask)
    return unique_cells[unique_cells != 0]


def masksToGraph(segmentations: np.ndarray, graph: nx.Graph, row_dimen: int, col_dimen: int, z_dimen: int,
                 z_span: int = 1, xy_span: int = 1, pixel_thresh: int = 1) -> nx.Graph:
    """Given all masks, convert them to nx.Graph object with neighboring masks as nodes with edges between them"""
    for z in tqdm(range(segmentations.shape[0]), desc='Converting to Graph representation...'):
        z_slice = segmentations[z, :, :]
        properties = regionprops(z_slice)
        bounding_boxes = {prop.label: prop.bbox for prop in properties}
        findNeighborsInImage(graph, segmentations, row_dimen, col_dimen, z_dimen, z, bounding_boxes, z_span, xy_span,
                             pixel_thresh)
    return graph


def loadAndInitialize(data_dir: str, file_name: str, masks_path: str, img_slice: np.ndarray)\
        -> Tuple[np.ndarray, np.ndarray, int, int, int, nx.Graph]:
    """Load .nd2 file and masks .npy file, and initialize Graph object"""
    print('Loading image and segmentation masks...')
    image = grab_nd2(img_path=fr'{data_dir}/{file_name}.nd2', channels_to_grab=['phase'])
    image = image[:, :, img_slice[0]:img_slice[1], img_slice[2]:img_slice[3]]
    loaded_masks = openMasksFile(masks_path)
    z_dimen, y_dim, x_dim = loaded_masks.shape
    g = nx.Graph()
    return image, loaded_masks, z_dimen, y_dim, x_dim, g


def dictOfProps(focus_masks: np.ndarray) -> Dict:
    """Create mapping of mask label to its image properties"""
    all_z_props = {}
    for z in range(focus_masks.shape[0]):
        properties = regionprops(focus_masks[z, :, :])
        all_z_props.update({prop.label: prop for prop in properties})
    return all_z_props


def atan2Angle(node: int, neighbor: int, all_z_props: Dict) -> float:
    """Calculate the angle between two centroids of cells and teh x-axis"""
    centroid1 = all_z_props[node].centroid
    centroid2 = all_z_props[neighbor].centroid
    delta_x = centroid2[1] - centroid1[1]
    delta_y = centroid2[0] - centroid1[0]
    angle_rad = math.atan2(delta_y, delta_x)
    # Convert range from [-π, π] to [0, 2π]
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def singleNeighborProcedure(node: int, neighbors: List, all_z_props: Dict, prev_angle: float)\
        -> Tuple[int, float, float]:
    """If cell is a top of chain"""
    cur_angle = atan2Angle(node, neighbors[0], all_z_props)
    angles_diff = abs(cur_angle - prev_angle)
    if angles_diff > 180:
        angles_diff = 360 - angles_diff
    return neighbors[0], angles_diff, cur_angle


def dilateAndCountIntersection(cell1_mask: int, cell2_mask: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate neighboring masks contact interface"""
    structuring_element = np.ones((3, 3, 3))
    dilated_cell1_mask = binary_dilation(cell1_mask, structuring_element)
    dilated_cell2_mask = binary_dilation(cell2_mask, structuring_element)
    contact_interface = dilated_cell1_mask & dilated_cell2_mask
    dilation_intersection_skeleton = skeletonize(contact_interface)
    dilation_intersection_skeleton = np.array(dilation_intersection_skeleton, dtype=bool)
    # Calculate the number of contact voxels
    contact_voxels = np.sum(contact_interface)
    skeleton_len = np.sum(dilation_intersection_skeleton)
    return contact_voxels, skeleton_len


def tiebreaker(segmentations: np.ndarray, cur_node: int, neighbor_a: int, neighbor_b: int, all_z_props: Dict,
               prev_angle: float) -> Tuple[int, float, float]:
    """If there is a tie looking at angles, use contact as tiebreaker"""
    cur_node_mask = (segmentations == cur_node)
    a_mask = (segmentations == neighbor_a)
    b_mask = (segmentations == neighbor_b)
    contact_a, skeleton_len_a = dilateAndCountIntersection(cur_node_mask, a_mask)
    contact_b, skeleton_len_b = dilateAndCountIntersection(cur_node_mask, b_mask)
    potential_chain_neighbor = neighbor_a if skeleton_len_a > skeleton_len_b else neighbor_b
    angle_potential_chain = atan2Angle(cur_node, potential_chain_neighbor, all_z_props)
    potential_chain_dif = abs(angle_potential_chain - prev_angle)
    return potential_chain_neighbor, potential_chain_dif, angle_potential_chain


def findPotentialChainNeighborCentroidsAngle(segmentations: np.ndarray, node: int, neighbors: List, all_z_props: Dict,
                                             prev_angle: float, angle_diff_tie_breaker: float)\
        -> Tuple[int, float, float]:
    """By angle continuity property, find potential neighbor"""
    if len(neighbors) == 1:
        return singleNeighborProcedure(node, neighbors, all_z_props, prev_angle)
    # Otherwise there are multiple neighbors, and we need to decide which one is the best candidate
    else:
        smallest_dif = math.inf
        angle_of_smallest_dif = math.nan
        potential_chain_neighbor = math.nan
        for neighbor in neighbors:
            cur_angle = atan2Angle(node, neighbor, all_z_props)
            angle_diff = abs(cur_angle - prev_angle)
            # Only if significant difference. if not, go to tiebreaker
            if angle_diff < (smallest_dif - angle_diff_tie_breaker):
                smallest_dif = angle_diff
                angle_of_smallest_dif = cur_angle
                potential_chain_neighbor = neighbor
            elif angle_diff < (smallest_dif + angle_diff_tie_breaker):
                potential_chain_neighbor, smallest_dif, angle_of_smallest_dif = tiebreaker(segmentations, node,
                                                                                           potential_chain_neighbor,
                                                                                           neighbor, all_z_props,
                                                                                           prev_angle)
        return potential_chain_neighbor, smallest_dif, angle_of_smallest_dif


def leafAndNeighborAngle(unassociated_nodes_g: nx.Graph, leaf: int, all_z_props: Dict) -> Tuple[float, int]:
    """Calculate the angle between mask and its given neighbor"""
    leafs_neighbor = list(unassociated_nodes_g.neighbors(leaf))[0]
    unassociated_nodes_g.remove_node(leaf)
    return atan2Angle(leaf, leafs_neighbor, all_z_props), leafs_neighbor


def addEdgeToChainAndUpdateCurNode(chain_g: nx.Graph, cur_node: int, potential_chain_next_node: int) -> int:
    """Given neighbor node, create edge between them"""
    chain_g.add_edge(cur_node, potential_chain_next_node)
    return potential_chain_next_node


def closingChainProcedure(unassociated_nodes_g: nx.Graph, cur_node: int, chains_graph_list: List, chain_g: nx.Graph,
                          deg_one_nodes: List) -> List:
    """Find "leaves", i.e., one degree nodes cells, and the assigning algorithm wil start from them"""
    nodes_to_remove = chain_g.nodes()
    unassociated_nodes_g.remove_nodes_from(nodes_to_remove)
    chains_graph_list.append(chain_g)
    # Check if the last node that associated is in leaves
    checkIfExistAndRemove(deg_one_nodes, cur_node)
    updated_leaves = detectLeaves(unassociated_nodes_g)
    return updated_leaves


def detectChains(segmentations: np.ndarray, deg_one_nodes: List, adjacency_graph: nx.Graph, angle_eps: float,
                 angle_diff_tie_breaker: float) -> List:
    """The main function that assigns masks to different chains"""
    unassociated_nodes_g, chains_graph_list, all_z_props = initForDetectingChain(adjacency_graph, segmentations)
    print('Using angle continuity algorithm for detecting chains.')
    while deg_one_nodes:
        print(len(deg_one_nodes), 'leaves remaining...')
        leaf = deg_one_nodes.pop(0)
        chain_g = nx.Graph()
        # Leaf has only 1 neighbor
        try:
            prev_angle, cur_node = leafAndNeighborAngle(unassociated_nodes_g, leaf, all_z_props)
        except IndexError:
            print(f"leaf {leaf} has no unassociated neighbors")
            continue
        chain_g.add_edge(leaf, cur_node)
        # Run over the chain
        neighbors = list(unassociated_nodes_g.neighbors(cur_node))
        while len(neighbors) > 0:
            # Return the most similar orientation node
            potential_chain_next_node, angles_diff, potential_angle = findPotentialChainNeighborCentroidsAngle(
                segmentations, cur_node, neighbors, all_z_props, prev_angle, angle_diff_tie_breaker)

            if angles_diff < angle_eps:
                neighbors = list(unassociated_nodes_g.neighbors(potential_chain_next_node))
                neighbors.remove(cur_node)
                cur_node = addEdgeToChainAndUpdateCurNode(chain_g, cur_node, potential_chain_next_node)
                prev_angle = potential_angle
            else:
                neighbors.remove(potential_chain_next_node)
        deg_one_nodes = closingChainProcedure(unassociated_nodes_g, cur_node, chains_graph_list, chain_g, deg_one_nodes)
    return chains_graph_list


def checkIfExistAndRemove(main_list: List, value: int) -> List:
    """If value exists in list, remove it"""
    try:
        main_list.remove(value)
    except ValueError:
        pass
    return main_list


def initForDetectingChain(adjacency_graph: nx.Graph, segmentations: np.ndarray) -> Tuple[nx.Graph, List, Dict]:
    all_z_props = dictOfProps(segmentations)
    return adjacency_graph.copy(), [], all_z_props


def detectLeaves(g: nx.Graph) -> List:
    """Find masks with one neighbor"""
    nodes = list(g.nodes)
    adjacency_matrix = nx.to_numpy_array(g, nodelist=nodes)
    degrees = np.sum(adjacency_matrix, axis=1)
    leaf_indices = np.where(degrees == 1)[0]
    leaf_nodes = [nodes[i] for i in leaf_indices]
    return leaf_nodes


def manuelChainCorrection(chain_g: nx.Graph, segmentations: np.ndarray, image: np.ndarray, begin_from: int,
                          end_at: int, chains_file_name: str, chain_num: int) -> List:
    """If the chain detection algorithm isn't producing perfect results, fix it manually"""
    # Get existing Qt app or create a new one
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    #if end_at == -1:
    #    end_at = len(chains_graphs)
    masks_proj = np.max(segmentations, axis=0)
    #for chain_num in range(begin_from, end_at):
    #chain_g = chains_graphs[chain_num]
    nodes = list(chain_g.nodes)
    nodes_bool_masks = np.isin(segmentations, nodes)
    only_chain_masks = segmentations.copy()
    only_chain_masks[~nodes_bool_masks] = 0
    if np.all(only_chain_masks == 0):
        chain_g.clear()
        #saveListOfGraphsToPklAndChains3D(chains_graphs, None, chains_file_name, None)
    viewer = napari.Viewer(title='Correcting chains assignment')
    nodes_proj = np.max(only_chain_masks, axis=0)
    print(type(nodes_proj))
    chain_layer_proj = viewer.add_labels(
        nodes_proj, blending='additive', name=f'chain {chain_num} projection')  # type: ignore
    chain_layer = viewer.add_labels(only_chain_masks, blending='additive', name=f'chain {chain_num}')
    viewer.add_image(image, blending='additive', name='DIC')  # type: ignore
    viewer.add_labels(masks_proj, blending='additive', name='Focus masks proj', visible=False)  # type: ignore
    focus_masks_layer = viewer.add_labels(
        segmentations, blending='additive', name='Focus masks', visible=False)  # type: ignore
    empty_seg1 = np.zeros_like(segmentations)
    empty_seg2 = np.zeros_like(segmentations)
    print(type(empty_seg1))
    add_layer = viewer.add_labels(empty_seg1, name='Manually added masks')  # type: ignore
    drop_layer = viewer.add_labels(empty_seg2, name='Manually dropped masks')  # type: ignore

    @add_layer.mouse_drag_callbacks.append
    def add_mask_to_chain(_, event):
        if add_layer.mode == 'paint':
            coordinates = tuple(map(int, event.position))
            if focus_masks_layer.data[coordinates] != 0:
                label_to_add = segmentations[coordinates]
                mask_location = (segmentations == label_to_add)
                chain_g.add_node(label_to_add)
                chain_layer.data[mask_location] = label_to_add  # type: ignore
                flatten_proj_location = np.max(mask_location, axis=0)
                nodes_proj[flatten_proj_location] = label_to_add
                chain_layer_proj.data[flatten_proj_location] = label_to_add  # type: ignore

    @drop_layer.mouse_drag_callbacks.append
    def drop_mask_from_chain(_, event):
        if drop_layer.mode == 'paint':
            coordinates = tuple(map(int, event.position))
            if chain_layer.data[coordinates] != 0:
                label_to_drop = chain_layer.data[coordinates]
                mask_location = (chain_layer.data == label_to_drop)
                chain_g.remove_node(label_to_drop)
                chain_layer.data[mask_location] = 0  # type: ignore
                flatten_proj_location = np.max(mask_location, axis=0)
                chain_layer_proj.data[flatten_proj_location] = 0  # type: ignore

    napari.run()
    #while viewer.window._qt_window.isVisible():
    #    time.sleep(0.1)
    #saveListOfGraphsToPklAndChains3D(chains_graphs, None, chains_file_name, None)
    # BLOCK until the window is closed
    viewer.window._qt_window.raise_()
    viewer.window._qt_window.activateWindow()
    while viewer.window._qt_window.isVisible():
        app.processEvents()
        time.sleep(0.005)
    return chain_g


def removeDuplicateChains(chains_list: List) -> Tuple[List, List]:
    """if there are two chains the same - drop  one of them"""
    used_nodes = set([])
    chain_num = 1
    for chain in tqdm(chains_list, desc='Checking that each node appears only in one chain'):
        intersection = used_nodes.intersection(chain.nodes)
        if len(intersection) > 0:
            print('Chain', chain_num, 'has dropped because', intersection, 'appears in 2 chains.')
            chain.clear()
        used_nodes = used_nodes.union(chain.nodes)
        chain_num += 1
    return chains_list, list(used_nodes)


def saveListOfGraphsToPklAndChains3D(chains_graphs: Union[np.ndarray, List], chains_3d: Union[None, np.ndarray],
                                     list_of_chains_path: str, chains_as_3d_path: Union[None, str]) -> None:
    """Save the data structures to pickle and .npy"""
    print('Saving chains in Graph list and as 3D array...')
    with open(list_of_chains_path, 'wb') as f:
        pickle.dump(chains_graphs, f)
        print('Chains are saved in .pkl file')
    if chains_3d is not None and chains_as_3d_path is not None:
        np.save(chains_as_3d_path, chains_3d)


def loadPkl(path: str) -> List:
    """Load the chains file"""
    with open(path, 'rb') as f:
        chains_graphs = pickle.load(f)
    print('Chains loaded from .pkl file')
    return chains_graphs


def removeEmptyChains(chains_graphs: List) -> List:
    """If there are Graphs with 0 nodes - drop them"""
    print('Number of chains before removing duplicated and empty chains:', len(chains_graphs))
    chains_graphs = [g for g in chains_graphs if g.number_of_nodes() > 0]
    print('Number of chains after removing duplicated and empty chains:', len(chains_graphs))
    return chains_graphs


def showNapariWithColoredChainsAndAddChainFromScratch(image: np.ndarray, chains_graphs: List, segmentations: np.ndarray,
                                                      assigned_segmentations: List) -> Tuple[List, np.ndarray]:
    """Manually add chain from beginning to the end, in case algorithm did not get it"""
    cur_chain = nx.Graph()
    # Load and show all of them, colored nicely
    viewer = napari.Viewer()
    viewer.add_image(image, blending='additive', name='Phase')  # type: ignore
    focus_masks_layer = viewer.add_labels(segmentations, blending='additive', name='Focus masks')  # type: ignore
    focus_masks_proj = np.max(segmentations, axis=0)
    chain_num = 1
    all_chains_proj = np.zeros_like(segmentations.shape[0])
    all_chains_3d_array = np.zeros_like(segmentations.shape[0])
    for chain in tqdm(chains_graphs, desc='Showing chains with Napari'):
        nodes = list(chain.nodes)
        nodes_bool_masks = np.isin(segmentations, nodes)
        filtered_nodes = np.where(nodes_bool_masks, chain_num, 0)
        all_chains_3d_array = np.where(all_chains_3d_array == 0, filtered_nodes, all_chains_3d_array)
        nodes_proj = np.max(filtered_nodes, axis=0)
        all_chains_proj = np.where(nodes_proj > 0, nodes_proj, all_chains_proj)
        chain_num += 1
    viewer.add_labels(focus_masks_proj, name='Focus masks projection')  # type: ignore
    viewer.add_labels(all_chains_3d_array, blending='additive', name='All chains')  # type: ignore
    viewer.add_labels(all_chains_proj, blending='additive', name='All chains projection')  # type: ignore
    empty_array1 = np.zeros_like(segmentations)
    add_chain_layer = viewer.add_labels(empty_array1, name='Add chain')  # type: ignore

    @add_chain_layer.mouse_drag_callbacks.append
    def add_nodes_to_chain(_, event):
        if add_chain_layer.mode == 'paint':
            (z_pos, c_pos, r_pos) = tuple(map(int, event.position))
            z_max, c_max, r_max = segmentations.shape[0], segmentations.shape[1], segmentations.shape[2]
            is_outside_image = c_pos < 0 or r_pos < 0 or c_pos > c_max or r_pos > r_max
            if is_outside_image and z_pos == 0:
                # Begin clean chain
                if cur_chain.number_of_nodes() != 0:
                    print('Close previous chain before starting a new one.')
                    return
                #cur_chain.clear()
                print('New chain has opened. \n', cur_chain.nodes)
            elif is_outside_image and z_pos == (z_max-1):
                # Add to list of chains
                if cur_chain.number_of_nodes() == 0:
                    print('Cannot append empty chain.')
                copied_chain = cur_chain.copy()
                chains_graphs.append(copied_chain)
                print('Final chain:', cur_chain.nodes, '\nMasks assignment to new chain has finished.')
                cur_chain.clear()
            elif not is_outside_image:
                # Add to opened chain
                mask_to_add = focus_masks_layer.data[z_pos, c_pos, r_pos]
                if mask_to_add != 0 and mask_to_add not in assigned_segmentations:
                    cur_chain.add_node(mask_to_add)
                    print(cur_chain.nodes)
                else:
                    print('The mask you tried to assign is already assigned, or it is not a mask.')
    napari.run()
    return chains_graphs, all_chains_3d_array


def assurePositionsInChains(chains: List, properties: pd.DataFrame) -> List:
    """Check that indeed two consecutive positioning have the closest proximity in the image"""
    label_to_centroid = dict(zip(properties['label'], properties['centroid']))
    success_flag = True
    for chain_idx, chain in enumerate(chains):
        nodes = list(chain.nodes())
        for i in range(len(nodes)-1):
            cur_label, adjacent_label = nodes[i], nodes[i+1]
            cur_centroid, adjacent_centroid = label_to_centroid[cur_label], label_to_centroid[adjacent_label]
            for later_label in nodes[i+2:]:
                later_centroid = label_to_centroid[later_label]
                adjacent_dist = distance.euclidean(cur_centroid, adjacent_centroid)
                non_adjacent_dist = distance.euclidean(cur_centroid, later_centroid)
                if non_adjacent_dist < adjacent_dist:
                    print(f"Contradiction in chain {chain_idx}:")
                    print(f"    {cur_label} and {adjacent_label} are adjacent in list, but {later_label} is closer")
                    success_flag = False
    if success_flag:
        print('All cell positions in chain are consistent with centroid locations.')
    return chains


'''
# Angle continuity algorithm parameters
Z_SPAN = 7
XY_SPAN = 2
ANGLE_EPS = 85.0
ANGLE_DIFF_TIE_BREAKER = 65.0
CHAIN_LENGTH_LOWER_BOUND = 1
PIXEL_THRESH = 9
# Other parameters
LOAD_CHAINS_GRAPHS = False
PROPS_FILE_NAME = f'fov_{FOV}_hyb_{HYB}'
LIST_OF_CHAIN_GRAPHS_FILE_NAME = f'fov_{FOV}_hyb_{HYB}.list_of_chain_Graphs'
CHAINS_AS_3D_FILENAME = f'fov_{FOV}_hyb_{HYB}.chains_as_3d_array'
BEGIN_MANUAL_CORRECTION_FROM_CHAIN = 300
END_MANUAL_CHAIN_CORRECTION_AT_CHAIN = 300

if __name__ == "__main__":
    # Change directory to current FOV
    os.chdir(f'C:\\Users\\LITVINOV\\PycharmProjects\\anabaenaSeg\\fov_{FOV}_hyb_{HYB}')
    if not LOAD_CHAINS_GRAPHS:
        img, masks, z_dim, r_dim, c_dim, adjacency_g = loadAndInitialize(DATA_DIR, FILE_NAME, FOCUS_MASKS_PATH,
                                                                         IMG_SLICE)
        #masks = masks[:, IMG_SLICE[0]:IMG_SLICE[1], IMG_SLICE[2]:IMG_SLICE[3]]  # z (depth),c (channel),x,y
        # Only phase
        img = img[:, 0, :, :]
        adjacency_g = masksToGraph(masks, adjacency_g, r_dim, c_dim, z_dim, z_span=Z_SPAN, xy_span=XY_SPAN,
                                   pixel_thresh=PIXEL_THRESH)
        adjacency_g = dropShortConnectedComponent(adjacency_g, CHAIN_LENGTH_LOWER_BOUND)
        leaves = detectLeaves(adjacency_g)
        chains_g = detectChains(masks, leaves, adjacency_g, ANGLE_EPS, ANGLE_DIFF_TIE_BREAKER)
    else:
        img, masks, _, _, _, _ = loadAndInitialize(DATA_DIR, FILE_NAME, FOCUS_MASKS_PATH, IMG_SLICE)
        img = img[:, 0, :, :]
        chains_g = loadPkl(LIST_OF_CHAIN_GRAPHS_FILE_NAME)
    chains_g = manuelChainCorrection(chains_g, masks, img, BEGIN_MANUAL_CORRECTION_FROM_CHAIN,
                                     END_MANUAL_CHAIN_CORRECTION_AT_CHAIN, LIST_OF_CHAIN_GRAPHS_FILE_NAME)
    chains_g, assigned_masks = removeDuplicateChains(chains_g)
    chains_g = removeEmptyChains(chains_g)
    chains_g, all_chains_3d, = showNapariWithColoredChainsAndAddChainFromScratch(img, chains_g, masks, assigned_masks)
    props = getRegionProps(None, file_name=PROPS_FILE_NAME, load=True)
    chains_g = assurePositionsInChains(chains_g, props)
    saveListOfGraphsToPklAndChains3D(chains_g, all_chains_3d, LIST_OF_CHAIN_GRAPHS_FILE_NAME, CHAINS_AS_3D_FILENAME)
'''
