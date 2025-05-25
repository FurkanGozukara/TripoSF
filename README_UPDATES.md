# TripoSF VAE Reconstruction App - Updates V4

## New Features

### 1. Sequential File Naming System
- **Problem Solved**: Files no longer overwrite each other
- **Implementation**: Files are now saved with sequential numbering format: `XXXX_filename_suffix.obj`
- **Examples**:
  - `0001_mesh_normalized.obj`
  - `0001_mesh_reconstructed.obj`
  - `0002_mesh_normalized.obj`
  - `0002_mesh_reconstructed.obj`

### 2. Batch Processing
- **New Tab**: Added "Batch Processing" tab in the interface
- **Features**:
  - Process entire folders of mesh files (.obj, .glb)
  - Progress tracking with file count
  - Skip existing files option to avoid reprocessing
  - Detailed results summary
  - Preserves original filenames with suffix

### 3. Robust File Handling
- **GLB Support**: Enhanced GLB to OBJ conversion with multiple fallback strategies
- **Error Recovery**: Better error handling and recovery mechanisms
- **File Validation**: Improved mesh validation and repair

## Usage

### Single File Processing
1. Upload a mesh file (.obj or .glb)
2. Configure reconstruction settings
3. Click "Reconstruct Mesh"
4. Files are saved with sequential numbering to prevent overwriting

### Batch Processing
1. Go to "Batch Processing" tab
2. Enter input folder path containing mesh files
3. Enter output folder path for results
4. Configure settings:
   - Enable/disable pruning
   - Set sample point number
   - Choose to skip existing files
5. Click "Start Batch Processing"
6. Monitor progress and results

## File Naming Convention

### Single File Mode
- Normalized: `XXXX_originalname_normalized.obj`
- Reconstructed: `XXXX_originalname_reconstructed.obj`

### Batch Mode
- Normalized: `originalname_normalized.obj`
- Reconstructed: `originalname_reconstructed.obj`

## Technical Improvements

1. **get_next_filename()**: Generates unique sequential filenames
2. **get_batch_output_filename()**: Preserves original names in batch mode
3. **process_single_file_batch()**: Handles individual file processing in batch mode
4. **run_batch_processing()**: Main batch processing function with progress tracking
5. **Enhanced error handling**: Better error messages and recovery

## Benefits

- **No more file overwrites**: Each processing run creates new files
- **Batch efficiency**: Process multiple files automatically
- **Progress tracking**: See real-time progress during batch processing
- **Skip existing**: Avoid reprocessing already completed files
- **Better organization**: Clear naming convention for easy file management 