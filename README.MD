# LCR-Net release v2.0 (modified with an improved pose proposals integration)

This project is a modified version of the [LCR-Net release v2.0](https://thoth.inrialpes.fr/src/LCR-Net/).
Please note that this code is released under the same terms and conditions as LCR-Net release v2.0: only for scientific or personal use under the GNU General Public License.

Compared to the original repository, our version includes an improved ppi version (in the file `lcr_net_ppi_improved.py` that extends the original pose proposals integration (PPI) in order to handle not only bodies (with 13 or 14 joints) but also hands (with 21 joints) and faces (with 84 joints). This required to slighly change the output of the LCR-Net network. Results for LCR-Net / LCR-Net++ should be the same.

Similar to the orignal repository, code for testing the models is only provided, not for training.
The code has been tested on Linux with Cuda 9.x and Cudnn 7.x. 
No support will be provided for installation or for other OS/configuration.

### Preparation

Please follow the instructions from the original repository:

* Download and install Detectron.pytorch: https://github.com/roytseng-tw/Detectron.pytorch (we do not provide support for its installation)
* Create a symbolic link in this folder to Detectron.pytorch

### Usage 

To use the code on an image, use the same command as the original repository:

python demo.py <modelname> <imagename> <gpuid>

* <modelname>: name of model to use, see [LCR-Net++ website](https://thoth.inrialpes.fr/src/LCR-Net/) for the list of available models)
* <imagename>: name of the image to test
* <gpuid>: gpu id to use (-1 for cpu)

### Citation

If you use this code, please cite the following papers:

```
@inproceedings{rogez2017lcr,
  title={{LCR-Net: Localization-Classification-Regression for Human Pose}},
  author={Rogez, Gregory and Weinzaepfel, Philippe and Schmid, Cordelia},
  booktitle={CVPR},
  year={2017}
}
```

```
@article{rogez2019lcr,
  TITLE = {{LCR-Net++: Multi-person 2D and 3D Pose Detection in Natural Images}},
  AUTHOR = {Rogez, Gr\'egory and Weinzaepfel, Philippe and Schmid, Cordelia},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2019},
  publisher={IEEE}
}
```
