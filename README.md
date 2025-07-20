# ABCD Caduceus
Classifying genomics samples taken from the [ABCD study](https://abcdstudy.org), using the [Caduceus](https://github.com/kuleshov-group/caduceus) model.

## Installation
### Prerequisites
You should have [UV](https://docs.astral.sh/uv) installed on your computer. Package resolution and downloads are generally faster than Conda, although it's theoretically possible to use Conda if you so desire (NOT OFFICIALLY SUPPORTED BY ME THOUGH).

To download all dependencies, run `uv sync --no-build-isolation`

### Important note
The `causal-conv1d` and `mamba` packages are dependencies of the (Caduceus implementation)[https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16] on Huggingface, and are provided in this repo as Git submodules.
Depending on your setup it may or may not be necessary to build and install these packages locally before installing the rest of the project, as the PyPI provided implementations rely on Nvidia CUDA drivers.
If you run AMD graphics or older Nvidia cards with older CUDA drivers, it may be necessary to build locally:
* `cd` into the `causal-conv1d` directory and run `uv pip install . --no-build-isolation`
* `cd` into the `mamba` directory and run `uv pip install . --no-build-isolation`

If you have issues with missing dependencies in the build process, you should run `uv pip install <missing dep>` and re-run the installation command. 
Pytorch is the usual suspect - it's a dependency of the two packages and of the project as a whole but you'd need to install them first before syncing the rest of the project. 
Follow the installation steps to `pip install torch` for your architecture and setup on the [Pytorch](https://pytorch.org) website, making sure to prepend `uv` to keep the installation in your local UV project environment.

## Usage
ABCD dataset should be symlinked to the `/data` directory if on Salk servers, or use `sshfs` on *nix systems to mount locally (it's not tracked by git because that's sensitive information). 
Relevant PLINK genomics are in the `plink` subdirectory, KSADS metrics in `phenos`, NCBI reference genome in `hg38`. Refer to `dataloader.py` for specifics on filenames.

Run `uv run main.py` to get a training run going. Statistics are reported to Weights and Biases (follow [instructions](https://wandb.ai/quickstart) to get set up)
