# 3D Reconstruction from Depth Estimation

This project demonstrates a pipeline for 3D reconstruction of objects and scenes using depth estimation from 2D images. By leveraging deep learning models and point cloud processing libraries, the workflow generates 3D meshes ready for visualization and analysis.

## Features

- **Depth Estimation:** Uses a pre-trained transformer model (`vinvino02/glpn-nyu`) for generating depth maps from RGB images.
- **Custom Point Cloud Creation:** Converts depth maps and RGB images into 3D point clouds.
- **Point Cloud Post-Processing:** Includes outlier removal and normal estimation for improved 3D geometry accuracy.
- **3D Mesh Generation:** Produces surface meshes from processed point clouds using Poisson surface reconstruction.
- **Custom Utility Functions:** Includes custom implementations for rotation, normal orientation, and other geometric operations to ensure robustness when standard methods encounter issues.

## Requirements

To run the project, make sure you have the following installed:

- **Python**
- Libraries:
  - `torch`
  - `transformers`
  - `numpy`
  - `matplotlib`
  - `Pillow`
  - `open3d`

## Workflow

1. **Load Input Image:** Start with an RGB image and preprocess it for depth estimation.
2. **Depth Map Generation:** Use a deep learning model to compute a depth map.
3. **Point Cloud Creation:** Convert the depth and RGB data into a 3D point cloud.
4. **Post-Processing:** Clean up the point cloud by removing outliers and estimating normals.
5. **Surface Mesh Reconstruction:** Generate a 3D mesh using Poisson surface reconstruction and export it as a `.ply` file.
6. **Visualization:** Visualize the intermediate results (depth map, point cloud) and final 3D mesh using `matplotlib` and `open3d`.

## Applications

This project can be applied in various fields, including:

- 3D modeling and printing
- AR/VR content creation
- Robotics and spatial analysis
- Image-based reconstruction research

## Example

An example image of a dog (`dog.jpg`) is included. The pipeline processes this image to generate a 3D mesh (`dog.ply`), which can be visualized using compatible 3D software or libraries.

---

Feel free to explore the code and adapt it for your own 3D reconstruction tasks!

