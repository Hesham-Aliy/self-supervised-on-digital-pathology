# Medical_segmentation

# MoNuSeg Dataset Segmentation Pipeline

This repository contains a segmentation pipeline implemented on the MoNuSeg dataset using relative positioning and multiscale unsupervised methods. The pipeline can be easily modified to perform classification or detection tasks as well.

## Dataset

The MoNuSeg dataset is used for training and evaluation. It is a widely-used benchmark dataset in the field of medical image segmentation. You can find the dataset [https://monuseg.grand-challenge.org/Data/].

## Methodology

The segmentation pipeline consists of two main components: Loading the unssupervised weights and doing the segmentation.

### Relative Positioning

The relative positioning technique is used to enhance the segmentation accuracy by considering the spatial relationships between different objects in the image. This approach takes into account the relative positions of neighboring pixels to improve the localization of boundaries.

### Multiscale Unsupervised Methods

The multiscale unsupervised methods help in capturing information at different scales and resolutions. By analyzing the image at multiple scales, the pipeline can extract features and patterns that are relevant for accurate segmentation. This approach enables the model to capture both fine-grained details and high-level contextual information.

## Getting Started

To use this segmentation pipeline, follow these steps:

1. Download the MoNuSeg dataset from https://monuseg.grand-challenge.org/Data/ and preprocess it according to the provided instructions.

2. Clone this repository to your local machine:
   ```
   git clone [https://github.com/your-username/your-repo.git](https://github.com/Hesham-Aliy/Medical_segmentation.git)
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the starting weights for relative positioning and multiscale methods from the following links:
   - Relative positioning starting weights: [[link-to-weights]](https://drive.google.com/file/d/1ldnFmniYk0f6uYhWir-UuTWXhXxkavRF/view?usp=sharing)
   - Multiscale starting weights: [[link-to-weights]](https://drive.google.com/file/d/1MCG1AqG9U5S8Bo_wuNmH_6Zp_G9WeLLO/view?usp=sharing)

5. change the path of the dataset and the weights in the Main.py code.

## Results

The segmentation pipeline achieves state-of-the-art performance on the MoNuSeg dataset. The results are comparable or even surpass the performance of existing methods reported in recent papers.

## References

To learn more about the techniques and methodologies used in this segmentation pipeline, please refer to the following papers:

1. Paper 1: [Relative positioning - Link to the paper](https://arxiv.org/abs/1505.05192)
2. Paper 2: [Multi-scale (Ours) - Link to the paper](https://www.springerprofessional.de/a-multi-scale-self-supervision-method-for-improving-cell-nuclei-/23303116) 


## Contributing

Contributions to this segmentation pipeline are welcome! If you find any issues or have ideas for improvement, please open an issue or submit a pull request on the GitHub repository.

## Contact

For any questions or inquiries, please contact [Hesham Ali](mailto:He.ali@nu.edu.eg).

## Note
1- The code of our unsupervised Multi-Scale approach will be relased soon. 

Happy segmentation!
