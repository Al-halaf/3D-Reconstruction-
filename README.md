3D Reconstruction from Depth Estimation
This project demonstrates a pipeline for 3D reconstruction of objects and scenes using depth estimation from 2D images. By leveraging deep learning models and point cloud processing libraries, the workflow generates 3D meshes ready for visualization and analysis.

Features
Depth Estimation: Uses a pre-trained transformer model (vinvino02/glpn-nyu) for generating depth maps from RGB images.
Custom Point Cloud Creation: Converts depth maps and RGB images into 3D point clouds.
Point Cloud Post-Processing: Includes outlier removal and normal estimation for improved 3D geometry accuracy.
3D Mesh Generation: Produces surface meshes from processed point clouds using Poisson surface reconstruction.
Custom Utility Functions: Includes custom implementations for rotation, normal orientation, and other geometric operations to ensure robustness when standard methods encounter issues.
Requirements
Python
Libraries: torch, transformers, numpy, matplotlib, Pillow, open3d
Workflow
Load an input image and preprocess it for depth estimation.
Use a deep learning model to compute a depth map.
Generate a point cloud from the depth and RGB data.
Perform point cloud cleanup and normal estimation.
Reconstruct a 3D surface mesh and export it as a .ply file.
Visualize the intermediate and final results using matplotlib and open3d.
Applications
3D modeling and printing
AR/VR content creation
Robotics and spatial analysis
Image-based reconstruction research
Example
The project includes an example image of a dog (dog.jpg) and generates a 3D mesh (dog.ply) for visualization and use in 3D applications.
