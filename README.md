# Focus Mask Differentiation and Cell Chain Detection
This package handles 3 dimensional microscopy imaging of filamentuous cells.
## Overview
The overview provides detailed instructions for the initial run (creating the focus masks and chains). Loading existing results are avalible as well.
## Focus Differentiator
Takes as input a 3D `.nd2` file and performs those steps:
  - **Inference**: Using pretrained CellPose convolutional neural network model for segmenting the cells.
  - **Filtering**: Each cell will have multiple masks. The algorithm will differentiate the masks that are in the focus z-plane using:
    1. * *Eccentricity* : the roundness of the mask. Very odd looking masks will be filtered.
    2. * *Size*: minimum and maximum. Extremely small and large masks will be filtered.
    3. CellPose output segmentation probability features (can be combined with logical AND/OR):
       - Minimum `(mean_pixel_probability + median_pixel_probability) / 2`
       - Minimum `median_pixel_probability`
       - Minimum `mean_pixel_probability / probability_standard_deviation`
    4. * *Interior to perimeter ratio*: Minimum. Good focus masks will have high ratio because their perimeter will be darker.
  - **Choosing**: After the filtering, focus mask is chosen based on the variance of the Laplacian of the mask, where high variance implies sharper image.
  - **Manual correction**: Napari viewer will be opened with the layers: all_masks, focus_masks, add_layer, drop_layer. Using the Napari brush, the user can add or discard focus masks.
  - **Relabeling**: Consecutive relabeling of the remaining masks.
  - **Mask properties**: Calculate and save the region props (area, centroid, orientation, etc.) for all the selected masks.
  - **Heterocysts**: using Napari viewer, manually label vegetative or heterocyst columns (in the case of Anabaena)
## Chain Detector
Takes focus masks from previous step and `.nd2` file with phase channel and performs those steps:
  - **Masks to Graphs**: Converting the image to Graph representation where adjacent masks are nodes with edge between them.
  - **Chain tips detection**: Detecting the "leaves" (the tips of the chains, masks with only one neighbor)
  - **Angle continuity algorithm**: Running over those leaf nodes and assigning cells to its chains using similarity of angles between the mask centroid and the x-axis.
  - **Tiebreaker**: If there are two candidates with similar angles, the algorithm will assign to chain the mask with the larger contact interface.
  - **Manual correction**: Napari viewer will open for each chain, and the user will be able to manually correct the chain         assignments.
## Installation
The python dependencies are managed via conda and can be installed using the provided environment.yaml file:
```
conda env create -f focus_and_chain_env.yaml
conda activate focus_and_chain_detection
```
## Example
Two notebooks are provided: `focus_diff_run_notebook.ipynb`, `chain_det_run_notebook.ipynb`. One for finding the focus masks and the second for associating the masks to filaments.

![Figures of workflow](Res_F6_segmentation.svg)
