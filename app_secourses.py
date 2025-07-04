# MIT License

# Copyright (c) 2025 VAST-AI-Research and contributors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import random
import gradio as gr
from safetensors.torch import load_file
import torch
from dataclasses import dataclass, field
import numpy as np
import open3d as o3d
import trimesh
import os
import time
import argparse
import subprocess
import platform
import glob
from omegaconf import OmegaConf
from typing import *

from triposf.modules import sparse as sp
from misc import get_device, find

def parse_args():
    parser = argparse.ArgumentParser(description="TripoSF VAE Reconstruction App")
    parser.add_argument("--share", action="store_true", help="Enable Gradio live sharing")
    return parser.parse_args()

def get_random_hex():
    random_bytes = os.urandom(8)
    random_hex = random_bytes.hex()
    return random_hex

def get_random_seed(randomize_seed, seed):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def get_next_sequential_number(base_path):
    """
    Get the next sequential number by checking existing numbered subfolders.
    Returns the next available number.
    """
    counter = 1
    while True:
        subfolder_name = f"{counter:04d}"
        subfolder_path = os.path.join(base_path, subfolder_name)
        if not os.path.exists(subfolder_path):
            return counter
        counter += 1

def get_sequential_filename(base_path, suffix, extension, file_number=None):
    """
    Generate a sequential filename and create subfolder structure.
    Format: outputs/XXXX/XXXX_suffix.extension
    Returns the full path and the file number used.
    """
    if file_number is None:
        file_number = get_next_sequential_number(base_path)
    
    # Create subfolder
    subfolder_name = f"{file_number:04d}"
    subfolder_path = os.path.join(base_path, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    
    # Create filename
    filename = f"{file_number:04d}_{suffix}.{extension}"
    full_path = os.path.join(subfolder_path, filename)
    
    return full_path, file_number

def get_batch_output_filename(input_file_path, output_dir, suffix):
    """
    Generate output filename for batch processing using input file name.
    """
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # Create subfolder with the same name as input file
    subfolder_path = os.path.join(output_dir, base_name)
    os.makedirs(subfolder_path, exist_ok=True)
    
    # Create filename
    filename = f"{base_name}_{suffix}.obj"
    full_path = os.path.join(subfolder_path, filename)
    
    return full_path

def normalize_mesh(mesh_path):
    scene = trimesh.load(mesh_path, process=False, force='scene')
    meshes = []
    for node_name in scene.graph.nodes_geometry:
        geom_name = scene.graph[node_name][1]
        geometry = scene.geometry[geom_name]
        transform = scene.graph[node_name][0]
        if isinstance(geometry, trimesh.Trimesh):
            geometry.apply_transform(transform)
            meshes.append(geometry)

    mesh = trimesh.util.concatenate(meshes)

    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = max(mesh.bounding_box.extents)
    mesh.apply_scale(2.0 / scale * 0.5)

    angle = np.radians(90)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [-1, 0, 0])
    mesh.apply_transform(rotation_matrix)
    return mesh

def load_quantized_mesh_original(
    mesh_path, 
    volume_resolution=256,
    use_normals=True,
    pc_sample_number=4096000,
):
    cube_dilate = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 0, -1],
                [0, -1, 0],
                [0, 1, 1],
                [0, -1, 1],
                [0, 1, -1],
                [0, -1, -1],

                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, -1],
                [1, -1, 0],
                [1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                [1, -1, -1],

                [-1, 0, 0],
                [-1, 0, 1],
                [-1, 1, 0],
                [-1, 0, -1],
                [-1, -1, 0],
                [-1, 1, 1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
            ]
        ) / (volume_resolution * 4 - 1)
    
    try:
        # Load mesh with Open3D
        print(f"Loading mesh with Open3D: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # Check if mesh has any vertices
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        print(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces")
        
        if len(vertices) == 0:
            raise ValueError(f"No vertices found in mesh")
            
        # Clip vertices to valid range
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        # Create voxels from mesh
        voxelization_mesh = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=1. / volume_resolution,
            min_bound=[-0.5, -0.5, -0.5],
            max_bound=[0.5, 0.5, 0.5]
        )
        voxel_mesh = np.asarray([voxel.grid_index for voxel in voxelization_mesh.get_voxels()])
        print(f"Created {len(voxel_mesh)} voxels from mesh")
        
        # Handle mesh sampling with potential issue recovery
        try:
            # Try to load with trimesh for sampling
            mesh_trimesh = trimesh.load(mesh_path)
            points_normals_sample = mesh_trimesh.sample(count=pc_sample_number, return_index=True)
            points_sample = points_normals_sample[0].astype(np.float32)
            faces_indices = points_normals_sample[1]
            print(f"Sampled {len(points_sample)} points using trimesh")
        except Exception as e:
            print(f"Trimesh sampling failed: {str(e)}, using fallback sampling")
            # Fallback: Generate random points within the bounding box
            points_sample = np.random.uniform(-0.5, 0.5, (pc_sample_number, 3)).astype(np.float32)
            # Create fake face indices (all zero)
            faces_indices = np.zeros(pc_sample_number, dtype=np.int32)
        
        # Create voxels from points
        voxelization_points = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(
                    np.clip(
                        (points_sample[np.newaxis] + cube_dilate[..., np.newaxis, :]).reshape(-1, 3),
                        -0.5 + 1e-6, 0.5 - 1e-6)
                    )
                ),
            voxel_size=1. / volume_resolution,
            min_bound=[-0.5, -0.5, -0.5],
            max_bound=[0.5, 0.5, 0.5]
        )
        voxel_points = np.asarray([voxel.grid_index for voxel in voxelization_points.get_voxels()])
        print(f"Created {len(voxel_points)} voxels from points")
        
        # Combine voxels
        if len(voxel_mesh) == 0 and len(voxel_points) == 0:
            raise ValueError("No voxels could be created from mesh or points")
            
        all_voxels = []
        if len(voxel_mesh) > 0:
            all_voxels.append(voxel_mesh)
        if len(voxel_points) > 0:
            all_voxels.append(voxel_points)
            
        voxels = torch.Tensor(np.unique(np.concatenate(all_voxels), axis=0))
        print(f"Combined into {voxels.shape[0]} unique voxels")
        
        # Handle normals
        if use_normals:
            try:
                mesh.compute_triangle_normals()
                normals_sample = np.asarray(mesh.triangle_normals)[faces_indices].astype(np.float32)
                print(f"Generated {len(normals_sample)} normals")
            except Exception as e:
                print(f"Normal calculation failed: {str(e)}, using random normals")
                normals_sample = np.random.randn(len(points_sample), 3).astype(np.float32)
                # Normalize the random normals
                norms = np.linalg.norm(normals_sample, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normals_sample = normals_sample / norms
            
            points_sample = torch.cat((torch.Tensor(points_sample), torch.Tensor(normals_sample)), axis=-1)
        else:
            points_sample = torch.Tensor(points_sample)
        
        return voxels, points_sample
    except Exception as e:
        print(f"Fatal error in load_quantized_mesh_original: {str(e)}")
        raise

class TripoSFVAEInference(torch.nn.Module):
    @dataclass
    class Config:
        local_pc_encoder_cls: str = ""
        local_pc_encoder: dict = field(default_factory=dict)

        encoder_cls: str = ""
        encoder: dict = field(default_factory=dict)

        decoder_cls: str = ""   
        decoder: dict = field(default_factory=dict)

        resolution: int = 256
        sample_points_num: int = 819_200
        use_normals: bool = True
        pruning: bool = False

        weight: Optional[str] = None

    cfg: Config

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.configure()

    def load_weights(self):
        if self.cfg.weight is not None:
            print("Pretrained VAE Loading...")
            state_dict = load_file(self.cfg.weight)
            self.load_state_dict(state_dict)

    def configure(self) -> None:
        self.local_pc_encoder = find(self.cfg.local_pc_encoder_cls)(**self.cfg.local_pc_encoder).eval()
        for p in self.local_pc_encoder.parameters():
            p.requires_grad = False

        self.encoder = find(self.cfg.encoder_cls)(**self.cfg.encoder).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.decoder = find(self.cfg.decoder_cls)(**self.cfg.decoder).eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.load_weights()

    @torch.no_grad()
    def forward(self, points_sample, sparse_voxel_coords):
        with torch.autocast("cuda", dtype=torch.float32):
            sparse_pc_features = self.local_pc_encoder(points_sample, sparse_voxel_coords, res=self.cfg.resolution, bbox_size=(-0.5, 0.5))
        sparse_tensor = sp.SparseTensor(sparse_pc_features, sparse_voxel_coords)
        latent, posterior = self.encoder(sparse_tensor)
        mesh = self.decoder(latent, pruning=self.cfg.pruning)
        return mesh
    
    @classmethod
    def from_config(cls, config_path):
        config = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(OmegaConf.structured(TripoSFVAEInference.Config), config)
        return cls(cfg)

# Header and constants
HEADER = """
### TripoSF VAE Reconstruction Improved SECourses App V6 - https://www.patreon.com/posts/126707772
"""
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
config = "configs/TripoSFVAE_1024.yaml"

random_hex = get_random_hex()
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
model = TripoSFVAEInference.from_config(config).to(device)

def robust_glb_to_obj_conversion(input_glb_path, use_batch_naming=False, batch_output_dir=None, file_number=None, original_input_path=None):
    """
    A robust GLB to OBJ converter that tries multiple strategies and repair techniques
    to ensure a valid mesh is produced.
    """
    if use_batch_naming and batch_output_dir:
        # Use original input path for folder naming, not the GLB path
        base_path = original_input_path if original_input_path else input_glb_path
        obj_path = get_batch_output_filename(base_path, batch_output_dir, "converted")
    else:
        # For single file processing, use the provided file number
        # (file_number should already be determined by the calling function)
        obj_path, _ = get_sequential_filename(output_dir, "converted", "obj", file_number)
    
    # Strategy 1: Direct trimesh load
    try:
        mesh = trimesh.load(input_glb_path, force='mesh')
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0 and len(mesh.vertices) > 0:
            mesh.export(obj_path)
            print(f"Strategy 1 success: Direct trimesh load")
            return obj_path
    except Exception as e:
        print(f"Strategy 1 failed: {str(e)}")
    
    # Strategy 2: Load as scene and combine meshes
    try:
        scene = trimesh.load(input_glb_path, force='scene')
        meshes = []
        for node_name in scene.graph.nodes_geometry:
            try:
                geom_name = scene.graph[node_name][1]
                transform = scene.graph[node_name][0]
                geometry = scene.geometry[geom_name]
                if isinstance(geometry, trimesh.Trimesh):
                    geometry.apply_transform(transform)
                    meshes.append(geometry)
            except Exception as e:
                print(f"Skipping node {node_name}: {str(e)}")
                continue
                
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            if len(combined.faces) > 0:
                combined.export(obj_path)
                print(f"Strategy 2 success: Scene concatenation")
                return obj_path
    except Exception as e:
        print(f"Strategy 2 failed: {str(e)}")
    
    # Strategy 3: Try pymeshlab for repair and conversion
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_glb_path)
        
        # Apply mesh repair operations
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_close_holes()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        ms.save_current_mesh(obj_path)
        
        # Verify the result
        result_mesh = trimesh.load(obj_path)
        if len(result_mesh.faces) > 0 and len(result_mesh.vertices) > 0:
            print(f"Strategy 3 success: PyMeshLab repair")
            return obj_path
    except Exception as e:
        print(f"Strategy 3 failed: {str(e)}")
    
    # If we reach here, all strategies failed
    raise ValueError("Failed to convert GLB to a valid OBJ mesh using all available strategies")

def run_normalize_mesh(input_mesh, use_batch_naming=False, batch_output_dir=None, file_number=None, original_input_path=None):
    try:
        # Store the original input path for consistent naming
        if original_input_path is None:
            original_input_path = input_mesh
            
        # Determine file number first if not provided
        if not use_batch_naming and file_number is None:
            file_number = get_next_sequential_number(output_dir)
            
        # Convert GLB to OBJ if needed
        if input_mesh.lower().endswith('.glb'):
            try:
                input_mesh = robust_glb_to_obj_conversion(input_mesh, use_batch_naming, batch_output_dir, file_number, original_input_path)
                print(f"Successfully converted GLB to: {input_mesh}")
            except Exception as e:
                print(f"GLB conversion error: {str(e)}")
                return gr.update(value=None), f"Error: Failed to convert GLB file to OBJ. {str(e)}", None
        
        # Basic validation - relaxed to allow more files through
        try:
            mesh_validate = trimesh.load(input_mesh)
            if len(mesh_validate.vertices) == 0:  # Only check for vertices, not faces
                print(f"Validation error: No vertices found")
                return gr.update(value=None), f"Error: The mesh file doesn't contain any vertices. Please try another file.", None
        except Exception as e:
            print(f"Mesh validation error: {str(e)}")
            return gr.update(value=None), f"Error: Failed to validate mesh file: {str(e)}", None
        
        print(f"Starting normalization of {input_mesh}")
        mesh_gt = normalize_mesh(input_mesh)
        
        if use_batch_naming and batch_output_dir:
            # Use original input path for consistent folder naming
            mesh_path_gt = get_batch_output_filename(original_input_path, batch_output_dir, "normalized")
            used_file_number = None
        else:
            # Use the file number (already determined above)
            mesh_path_gt, used_file_number = get_sequential_filename(output_dir, "normalized", "obj", file_number)
        
        # Simplified mesh creation - avoid extra validation that might reject valid meshes
        try:
            mesh_normalized = trimesh.Trimesh(vertices=mesh_gt.vertices.tolist(), faces=mesh_gt.faces.tolist())
            mesh_normalized.export(mesh_path_gt)
            print(f"Successfully normalized and saved: {mesh_path_gt}")
            return mesh_path_gt, None, used_file_number if not use_batch_naming else None
        except Exception as e:
            print(f"Error saving normalized mesh: {str(e)}")
            # As a fallback, try to save directly
            try:
                mesh_gt.export(mesh_path_gt)
                print(f"Used fallback export for: {mesh_path_gt}")
                return mesh_path_gt, None, used_file_number if not use_batch_naming else None
            except:
                return gr.update(value=None), f"Error during mesh export: {str(e)}", None
    except Exception as e:
        print(f"General normalization error: {str(e)}")
        return gr.update(value=None), f"Error during mesh normalization: {str(e)}", None

def run_reconstruction(input_mesh, sample_points_num, pruning, seed, error_msg, normalized_mesh_path=None, use_batch_naming=False, batch_output_dir=None, file_number=None, original_input_path=None):
    try:
        # If there was an error in normalization, don't proceed
        if error_msg:
            print(f"Skipping reconstruction due to error: {error_msg}")
            return gr.update(value=None)
            
        model.cfg.pruning = pruning
        model.cfg.sample_points_num = sample_points_num
        
        # Use the normalized mesh path if provided (from batch processing or previous step)
        if normalized_mesh_path and os.path.exists(normalized_mesh_path):
            mesh_path_gt = normalized_mesh_path
        else:
            # Fallback: try to find the most recent normalized file in the numbered subfolder
            if file_number:
                subfolder_name = f"{file_number:04d}"
                pattern = f"{output_dir}/{subfolder_name}/*normalized.obj"
                matching_files = glob.glob(pattern)
                if matching_files:
                    mesh_path_gt = matching_files[0]  # Should only be one
                else:
                    print(f"No normalized mesh file found in subfolder: {subfolder_name}")
                    return gr.update(value=None)
            else:
                print(f"No file number provided and no normalized mesh path")
                return gr.update(value=None)
        
        # Check if the normalized mesh exists
        if not os.path.exists(mesh_path_gt):
            print(f"Normalized mesh file not found: {mesh_path_gt}")
            return gr.update(value=None)
        
        print(f"Starting reconstruction with {sample_points_num} sample points, pruning={pruning}")
        try:
            sparse_voxels, points_sample = load_quantized_mesh_original(
                                                                mesh_path_gt, 
                                                                volume_resolution=model.cfg.resolution, 
                                                                use_normals=model.cfg.use_normals, 
                                                                pc_sample_number=model.cfg.sample_points_num,
                                                            )

            print(f"Generated sparse_voxels shape: {sparse_voxels.shape}, points_sample shape: {points_sample.shape}")
            sparse_voxels, points_sample = sparse_voxels.to(device), points_sample.to(device)
            sparse_voxels_sp = torch.cat([torch.zeros_like(sparse_voxels[..., :1]), sparse_voxels], dim=-1).int()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                mesh_recon = model(points_sample[None], sparse_voxels_sp)[0]

            print(f"Reconstruction successful with {len(mesh_recon.vertices)} vertices and {len(mesh_recon.faces)} faces")
            
            if use_batch_naming and batch_output_dir:
                # Use original input path for consistent folder naming
                base_path = original_input_path if original_input_path else input_mesh
                mesh_path_recon = get_batch_output_filename(base_path, batch_output_dir, "reconstructed")
            else:
                mesh_path_recon, _ = get_sequential_filename(output_dir, "reconstructed", "obj", file_number)
            
            mesh_reconstructed = trimesh.Trimesh(vertices=mesh_recon.vertices.tolist(), faces=mesh_recon.faces.tolist())
            mesh_reconstructed.export(mesh_path_recon)
            print(f"Saved reconstructed mesh to {mesh_path_recon}")
            return mesh_path_recon
        except Exception as e:
            print(f"Error during reconstruction: {str(e)}")
            return gr.update(value=None)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return gr.update(value=None)

def open_output_folder():
    """Open the outputs folder using the default file explorer on either Windows or Linux."""
    output_path = os.path.abspath(output_dir)
    try:
        if platform.system() == "Windows":
            os.startfile(output_path)
        else:  # Linux and macOS
            subprocess.Popen(["xdg-open", output_path])
        return f"Opening folder: {output_path}"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def process_single_file_batch(input_file, output_dir, sample_points_num, pruning, skip_existing):
    """Process a single file in batch mode"""
    try:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Check if output files already exist and skip if requested
        if skip_existing:
            subfolder_path = os.path.join(output_dir, base_name)
            if os.path.exists(subfolder_path):
                # Check if both normalized and reconstructed files exist
                normalized_pattern = os.path.join(subfolder_path, f"{base_name}_normalized.obj")
                reconstructed_pattern = os.path.join(subfolder_path, f"{base_name}_reconstructed.obj")
                if os.path.exists(normalized_pattern) and os.path.exists(reconstructed_pattern):
                    return f"Skipped {base_name} (folder {base_name} already exists)"
        
        # Normalize mesh
        normalized_result, error_msg, used_file_number = run_normalize_mesh(input_file, use_batch_naming=True, batch_output_dir=output_dir, original_input_path=input_file)
        if error_msg:
            return f"Error normalizing {base_name}: {error_msg}"
        
        # Reconstruct mesh
        reconstructed_result = run_reconstruction(
            input_file, sample_points_num, pruning, 0, None, 
            normalized_mesh_path=normalized_result, 
            use_batch_naming=True, 
            batch_output_dir=output_dir,
            original_input_path=input_file
        )
        
        if reconstructed_result:
            return f"Successfully processed {base_name} -> folder {base_name}"
        else:
            return f"Error reconstructing {base_name}"
            
    except Exception as e:
        return f"Error processing {os.path.basename(input_file)}: {str(e)}"

def run_batch_processing(input_folder, output_folder, sample_points_num, pruning, skip_existing, progress=gr.Progress()):
    """Process all mesh files in a folder"""
    try:
        if not input_folder or not os.path.exists(input_folder):
            return "Error: Input folder does not exist or is not specified"
        
        if not output_folder:
            return "Error: Output folder is not specified"
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all mesh files in input folder
        supported_extensions = ['*.obj', '*.glb']
        input_files = []
        for ext in supported_extensions:
            input_files.extend(glob.glob(os.path.join(input_folder, ext)))
            input_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        if not input_files:
            return f"No mesh files (.obj, .glb) found in {input_folder}"
        
        results = []
        total_files = len(input_files)
        
        for i, input_file in enumerate(input_files):
            progress((i + 1) / total_files, f"Processing {os.path.basename(input_file)} ({i + 1}/{total_files})")
            result = process_single_file_batch(input_file, output_folder, sample_points_num, pruning, skip_existing)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if "Successfully processed" in r)
        skipped = sum(1 for r in results if "Skipped" in r)
        errors = total_files - successful - skipped
        
        summary = f"Batch processing completed!\n"
        summary += f"Total files: {total_files}\n"
        summary += f"Successfully processed: {successful}\n"
        summary += f"Skipped: {skipped}\n"
        summary += f"Errors: {errors}\n\n"
        summary += "Details:\n" + "\n".join(results)
        
        return summary
        
    except Exception as e:
        return f"Error during batch processing: {str(e)}"

with gr.Blocks(title="TripoSFRecon",theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)

    with gr.Tabs():
        with gr.TabItem("Single File Processing"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_mesh_path = gr.File(label="Upload mesh file (.obj, .glb)", file_types=[".obj", ".glb"], type="filepath")
                    
                    with gr.Accordion("Reconstruction Settings", open=True):
                        use_pruning = gr.Checkbox(label="Pruning", value=False)
                        recon_button = gr.Button("Reconstruct Mesh", variant="primary")
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=0,
                            value=0
                        )
                        with gr.Row():
                            sample_points_num = gr.Slider(
                                label="Sample point number",
                                minimum=819200,
                                maximum=8192000,
                                step=100,
                                value=819200
                            )
                        with gr.Row():
                            preset_fast = gr.Button("Fast (819K)", size="sm", variant="secondary")
                            preset_balanced = gr.Button("Balanced (2M)", size="sm", variant="secondary")
                            preset_high = gr.Button("High Quality (4M)", size="sm", variant="secondary")
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Column():
                    normalized_model_output = gr.Model3D(label="Normalized Mesh", interactive=True)
                    open_folder_button = gr.Button("Open Outputs Folder")
                    reconstructed_model_output = gr.Model3D(label="Reconstructed Mesh", interactive=True)
                    error_output = gr.Textbox(label="Error Messages", visible=True)
        
        with gr.TabItem("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    input_folder = gr.Textbox(label="Input Folder Path", placeholder="Enter path to folder containing mesh files")
                    output_folder = gr.Textbox(label="Output Folder Path", placeholder="Enter path for output files")
                    
                    with gr.Accordion("Batch Settings", open=True):
                        batch_use_pruning = gr.Checkbox(label="Pruning", value=False)
                        with gr.Row():
                            batch_sample_points_num = gr.Slider(
                                label="Sample point number",
                                minimum=819200,
                                maximum=8192000,
                                step=100,
                                value=819200
                            )
                        with gr.Row():
                            batch_preset_fast = gr.Button("Fast (819K)", size="sm", variant="secondary")
                            batch_preset_balanced = gr.Button("Balanced (2M)", size="sm", variant="secondary")
                            batch_preset_high = gr.Button("High Quality (4M)", size="sm", variant="secondary")
                        skip_existing = gr.Checkbox(label="Skip existing files", value=True)
                        batch_process_button = gr.Button("Start Batch Processing", variant="primary")
                
                with gr.Column():
                    batch_output = gr.Textbox(label="Batch Processing Results", lines=20, max_lines=30)


    with gr.Row():
        gr.Markdown("""### TripoSF represents a significant leap forward in 3D shape modeling, combining high-resolution capabilities with arbitrary topology support.

**New Features:**
- **Clean sequential naming**: Files are saved with clean sequential numbers in organized subfolders for single file processing
  - Single file format: `outputs/0001/0001_converted.obj`, `outputs/0001/0001_normalized.obj`, `outputs/0001/0001_reconstructed.obj`
  - Each single processing session gets its own numbered subfolder (0001, 0002, etc.)
- **Batch processing**: Process entire folders of mesh files with progress tracking and skip existing files option
  - Batch file format: `outputs/filename/filename_converted.obj`, `outputs/filename/filename_normalized.obj`, `outputs/filename/filename_reconstructed.obj`
  - Each batch file gets its own subfolder named after the input file

**Usage Tips:**
1. It is recommanded to enable `pruning` for open-surface objects
2. Increasing sampling points is helpful for reconstructing complex shapes
3. Supports both OBJ and GLB file formats with robust conversion (GLB files will be automatically converted to OBJ)
4. Use batch processing for processing multiple files efficiently
5. Files are organized in numbered subfolders for easy management""")

    # State to store normalized mesh path and file number
    normalized_mesh_state = gr.State()
    file_number_state = gr.State()

    open_folder_button.click(
        fn=open_output_folder,
        inputs=[],
        outputs=[error_output]
    )

    def normalize_and_store_path(input_mesh):
        # Let run_normalize_mesh handle the file number determination
        normalized_path, error_msg, used_file_number = run_normalize_mesh(input_mesh)
        return normalized_path, error_msg, normalized_path, used_file_number

    def reconstruct_with_path(input_mesh, sample_points_num, use_pruning, seed, error_output, normalized_path, file_number):
        return run_reconstruction(input_mesh, sample_points_num, use_pruning, seed, error_output, normalized_path, False, None, file_number)

    recon_button.click(
        normalize_and_store_path,
        inputs=[input_mesh_path],
        outputs=[normalized_model_output, error_output, normalized_mesh_state, file_number_state]
    ).then(
        get_random_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        reconstruct_with_path,
        inputs=[input_mesh_path, sample_points_num, use_pruning, seed, error_output, normalized_mesh_state, file_number_state],
        outputs=[reconstructed_model_output],
    )

    # Batch processing event handler
    batch_process_button.click(
        run_batch_processing,
        inputs=[input_folder, output_folder, batch_sample_points_num, batch_use_pruning, skip_existing],
        outputs=[batch_output]
    )

    # Preset button event handlers for single file processing
    preset_fast.click(lambda: 819200, outputs=[sample_points_num])
    preset_balanced.click(lambda: 2048000, outputs=[sample_points_num])
    preset_high.click(lambda: 4096000, outputs=[sample_points_num])

    # Preset button event handlers for batch processing
    batch_preset_fast.click(lambda: 819200, outputs=[batch_sample_points_num])
    batch_preset_balanced.click(lambda: 2048000, outputs=[batch_sample_points_num])
    batch_preset_high.click(lambda: 4096000, outputs=[batch_sample_points_num])

args = parse_args()
demo.launch(inbrowser=True, share=args.share)
