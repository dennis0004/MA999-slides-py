# MA999 Agent-Based Modelling - Lecturing slides

## Prerequisites

To run this coursework, You will need Python and Jupyter installed, with several key packages.

### 1. Python

You can install Python with Anaconda (https://www.anaconda.com/products/distribution) or download Python from https://www.python.org/downloads/, and then follow installation instructions for your operating system.

Check if Python works by typing `python --version` or `python3 --version` in the terminal.

### 2. Jupyter

Jupyter Notebook or JupyterLab is required to open and run the `.ipynb` files. If you have Anaconda installed, it comes with Jupyter. You can also run Jupyter notebooks directly in IDEs such as VSCode or PyCharm.

### 3. Required Python Libraries

The project relies on the following Python libraries:
- `mesa`: For agent-based modeling.
- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For plotting and visualization.
- `networkx`: For graph creation and manipulation.
- `scipy`: For solving differential equations in the mean-field models.
- `imageio-ffmpeg`: For saving animations.


You can install these packages using `pip` or `conda`. It is recommended to use a separated environment to avoid conflicts with other projects.

```bash
# Create and activate a new environment (optional)
conda create --name ma999 python=3.13
conda activate ma999

# Install the required packages
pip install mesa numpy pandas matplotlib networkx scipy jupyterlab imageio-ffmpeg
```

After installing the packages, you can launch Jupyter by running `jupyter notebook` or `jupyter lab` in your terminal from the project's root directory.

Any issues should be directed to chin-wing.leung@warwick.ac.uk.

# Troubleshooting

## General tips

If you encounter issues with dependencies, ensure your environment is activated and that all packages listed above are installed correctly. You can install them all at once using `pip install -r requirements.txt`.

## Animation Issues (e.g., `matplotlib` animations)

To save animations as `.mp4` files, `matplotlib` requires access to an FFmpeg executable. A straightforward way to handle this is to use the `imageio-ffmpeg` library, which provides a standalone FFmpeg binary.

1.  **Install `imageio-ffmpeg`**: Make sure it is installed in your environment by running `pip install imageio-ffmpeg`.

2.  **Configure Matplotlib**: Before creating your animation, add the following lines to your Python script or Jupyter cell to tell `matplotlib` where to find the FFmpeg executable:

    ```python
    import matplotlib.pyplot as plt
    import imageio_ffmpeg

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    ```

This setup ensures that `matplotlib` can save animations without requiring a system-wide FFmpeg installation, making the project more portable.
