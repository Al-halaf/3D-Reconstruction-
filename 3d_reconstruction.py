# DEC 21 2024

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import open3d as o3d
import numpy as np


def rotate_mesh(mesh, rotation):
    # Apply the rotation matrix to the mesh vertices
    vertices = np.asarray(mesh.vertices)
    rotated_vertices = np.dot(vertices, rotation.T)  # Apply rotation (transpose matrix)
    mesh.vertices = o3d.utility.Vector3dVector(rotated_vertices)

    return mesh


def rotation_matrix(roll, pitch, yaw):
    # Rotation matrix for X-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Rotation matrix for Y-axis (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Rotation matrix for Z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combine the rotations: R = Rz * Ry * Rx (Note the order of multiplication)
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def orient_normals(pcd, direction_vector=None):
    # Ensure the input point cloud has normals
    if direction_vector is None:
        direction_vector = [0, 0, -1]

    if not pcd.has_normals():
        raise ValueError("Point cloud does not have normals. Use estimate_normals() before calling this function.")

    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Access normals as a numpy array
    normals = np.asarray(pcd.normals)

    # Flip normals to align with the direction
    for i in range(normals.shape[0]):
        # Check if the normal points away from the desired direction
        if np.dot(normals[i], direction_vector) < 0:
            normals[i] = -normals[i]  # Flip the normal

    # Update the normals in the point cloud
    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def depth_to_point_cloud(rgb_image, depth_image, focal_length, cx, cy):
    height, width = depth_image.shape  # Height and width of the depth map

    # Create a meshgrid of pixel coordinates
    i, j = np.meshgrid(np.arange(width), np.arange(height))

    # Compute 3D coordinates (x, y, z)
    z = depth_image
    x = (j - cx) * z / focal_length
    y = -(i - cy) * z / focal_length

    # Stack x, y, z into a 3D array
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Reshape RGB image to match the point cloud dimensions
    colors = rgb_image.reshape(-1, 3) / 255.0  # Normalize to [0, 1]

    # Remove invalid points (where depth == 0)
    valid_indices = (z > 0).flatten()
    points = points[valid_indices]
    colors = colors[valid_indices]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


# Getting the pre-trained model
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# loading the image
image = Image.open("dog.jpg")

# Resizing the image
new_height = 400 if image.height > 400 else image.height
new_height -= (new_height % 32)

new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff

new_size = (new_width, new_height)
image = image.resize(new_size)

# Processing the image using the feature extractor
inputs = feature_extractor(images=image, return_tensors="pt")

# Getting the depth estimation using the model
with torch.no_grad(): 
    outputs = model(**inputs)
    depth = outputs.predicted_depth

# Post-processing the image for visualisation
pad = 16
output = depth.squeeze().cpu().numpy() * 1000
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# Visualise the image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].tick_params(left=False, right=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap="plasma")
ax[1].tick_params(left=False, right=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(10)

# Prepare the depth image for open3d
width, height = image.size
depth_image = (output * 255 / np.max(output)).astype("uint8")
image = np.array(image)

# Create a 3d geometry based on the image
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d,
                                                                convert_rgb_to_intensity=False)

# Create a camera mmodel for the images (standard pinhole camera)
camera = o3d.camera.PinholeCameraIntrinsic()
focal_length = 500  # Example for typical depth cameras
camera.set_intrinsics(width, height, focal_length,  focal_length, width/2, height/2)

# Creating point cloud using open3d
#point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)  # Built-in function
# I tried using the above function, but it didn't work for me, so I made my own function
# If the built-in function does not work for you too, then uncomment the line below
point_cloud = depth_to_point_cloud(image, depth_image, focal_length, width/2, height/2)  # Custom function

# Visualise the geometry
o3d.visualization.draw_geometries([point_cloud])

# Post-processing the 3d point cloud
cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
point_cloud = point_cloud.select_by_index(ind)  # Removing any statistical outlier

point_cloud.estimate_normals()
point_cloud.orient_normals_to_align_with_direction()  # Built-in function
# Again the above function was not working for me, so I created my own custom function
# If it doesn't work for you too, then comment the above line and uncomment the line below
#point_cloud = orient_normals(point_cloud)  # Custom function

o3d.visualization.draw_geometries([point_cloud])

# Surface reconstruction
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=10, n_threads=1)[0]

# Rotate the mesh
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # Built-in function
# And again the built-in function does not work for me
#rotation = rotation_matrix(np.pi, 0, 0)  # Custom function

mesh = mesh.rotate(rotation, center=(0, 0, 0))  # Built-in function
# And again this function also does not work
#mesh = rotate_mesh(mesh, rotation)  # Custom function

# Visualise the mesh
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# Exporting the 3d mesh
o3d.io.write_triangle_mesh("dog.ply", mesh)
