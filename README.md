# OmniDepth-PyTorch
A PyTorch reimplementation of the Omnidepth paper from Zioulis et al., ECCV 2018:

# Dependencies
 - PyTorch 1.0+
 - numpy (various things)
 - scikit-image (for data loading)
 - OpenEXR (for loading depth files)
 - visdom (for visualizations)

 The easiest way to get set up is to just [set up a conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) using the *omnidepth.yml* file in this repository.

**Note:** The OpenEXR dependency can be problematic to get set up. I've found that if the `pip` version isn't working, installing via `sudo apt install openexr` often solves the problem when using Ubuntu.

# Cloning the network
Once you've installed and activated the conda environment, you should clone the network. You can do that with:

```
git clone https://github.com/meder411/OmniDepth-PyTorch.git
```


# Dataset
To get the OmniDepth dataset, please file a request with the authors [here](http://vcl.iti.gr/360-dataset/).


# Usage
Run `python train_omnidepth.py` to run the training routine. Run `python test_omnidepth.py` to run the evaluation routine. You can edit the parameters in those files.

## Testing
Included is `rectnet.pth`, a PyTorch model converted from the released Caffe model for the top RectNet model from the original paper. If you have set everything up correctly, running `test_omnidepth.py` will run the evaluation script on this model and output the following results:

```
  Avg. Abs. Rel. Error: 0.0641
  Avg. Sq. Rel. Error: 0.0197
  Avg. Lin. RMS Error: 0.2297
  Avg. Log RMS Error: 0.0993
  Inlier D1: 0.9663
  Inlier D2: 0.9951
  Inlier D3: 0.9984
```

## Training

During training, you first need to start the visdom server in order to visualize training. It's best to do this in a [`screen`](https://www.gnu.org/software/screen/manual/screen.html). To start the visdom server just call `visdom`.

The visualizations can be viewed at `localhost:8097`. If running visdom and the network on a server, you will need to tunnel to the server to view it locally. For example:

```
ssh -N -L 8888:localhost:8097 <username>@<server-ip>
```

allows you to view the training visualizations at localhost:8888 on your machine.

**Notable difference with the paper:** PyTorch's weight decay for the Adam solver does not seem to function the same way as Caffe's. Hence, I do not use weight decay in training. Instead, I use a learning rate schedule, but I have not been able to match the Caffe results.


# Credit
If you do use this repository, please make sure to cite the authors' original paper:

```
Zioulis, Nikolaos, et al. "OmniDepth: Dense Depth Estimation for Indoors Spherical Panoramas." 
Proceedings of the European Conference on Computer Vision (ECCV). 2018.
```
