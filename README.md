# WARNING: THIS INSTRUCTIONS ARE STILL BEING WRITTEN. STAY TUNED.

# 4D Generic Video Object Proposals
# Track, then Decide: Category-Agnostic Vision-based Multi-Object Tracking

This repository contains code for the work as described in

**Track, then Decide: Category-Agnostic Vision-based Multi-Object Tracking. ICRA 2018. (https://arxiv.org/pdf/1712.07920.pdf)**

By [Aljosa Osep](https://www.vision.rwth-aachen.de/person/13/), [Wolfgang Mehner](https://www.vision.rwth-aachen.de/person/7/), [Paul Voigtlaender](https://www.vision.rwth-aachen.de/person/197/), [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/), Computer Vision Group, RWTH Aachen University

and 

**4D Generic Video Object Proposals** (Under review, coming to arxiv soon)

By [Aljosa Osep](https://www.vision.rwth-aachen.de/person/13/), [Paul Voigtlaender](https://www.vision.rwth-aachen.de/person/197/), [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/), Mark Weber, [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/), Computer Vision Group, RWTH Aachen University

## Demo  Video
* [CAMOT video (older version)](https://youtu.be/HYXzHuD4AKI)
* 4DGVT videos coming soon!
## Prerequisite

In order to run the code, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):
* cmake (tested with 3.9.6, earlier versions should work too)
* GCC 5.4.0
* Libs:
  * Eigen (3.x)
  * Boost (1.55 or later)
  * OpenCV (3.0.0 + OpenCV contrib)
  * PCL (1.8.0)

## Install

### Data
Note: any other paths will do too, you will just need to adapt for that in the `%ROOT$/script/exec_tracker.sh`

0. Download [KITTI tracking dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and place it to `/home/${USER}/data/kitti_tracking`
0. Download [precomputed segmentations](https://drive.google.com/open?id=1AmDVzanSeHvmgJ4nh36jByOH-qIsib_2) we provide for KITTI tracking dataset, unzip to `/home/${USER}/data/kitti_tracking/preproc`
0. Clone this repo to `/home/${USER}/projects`

### Compiling the code using CMake
0.  `mkdir build`
0.  `cmake ..`
0.  `make all`

### Running the tracker
0.  Enter `%PROJ_DIR%/script/`
0.  Execute `exec_tracker.sh`

### Tracker settings
* `SEGM_INPUTS` - Specify which pre-computed segmentations to use -- Mask Proposal R-CNN (`mprcnn_coco`; recommended), Sharpmask (`sharpmask_coco`), Mask R-CNN fine-tuned on KITTI (`mrcnn_tuned`) 
* `INF_MODEL` - Specify which model should be used for inference - `4DGVT` (recommended) or `CAMOT`.
* `INPUT_CONF_THRESH` - Detection/proposal score threshold. In case it is set to `0.8` or more, you will be only forwarding confident detections.
* `MAX_NUM_PROPOSALS` - Max. proposals fed to track generator per frame. More proposals -> slower, higher recall. Not recommended to be set above 500.

## Remarks

* Running CAMOT vs. 4DGVT
    * TODO

* Inputs to the tracker
    * You can use our [precomputed segmentations](https://drive.google.com/open?id=1AmDVzanSeHvmgJ4nh36jByOH-qIsib_2) for KITTI
    * Provide your own using (export per-frame segmentations to json, pass jsons to the tracker):
        * Sharpmask [repo](https://github.com/facebookresearch/deepmask)
        * Our Mask Proposal R-CNN (MP R-CNN) [repo](https://github.com/aljosaosep/mprcnn)
        * You can also use MaskX R-CNN, trained on 3K+ classes on Visual Genome dataset [project page + code](http://ronghanghu.com/seg_every_thing/)

* External libraries
    * The tracker ships the following external modules:
        * **libelas** - disparity estimation (http://www.cvlibs.net/software/libelas/)
        * **libviso2** - egomotion estimation (http://www.cvlibs.net/software/libviso/)
        * **nlohman::json** - json parser (https://github.com/nlohmann/json)
        * **maskApi** - COCO mask API for C (https://github.com/cocodataset/cocoapi)

* Additional remarks about CAMOT
    * TODO

* Run the tracker in `release` mode (otherwise it will be slow).

If you have any issues or questions about the code, please contact me https://www.vision.rwth-aachen.de/person/13/

## Citing

If you find the tracker useful in your research, please consider citing:

    @article{Osep18ICRA,
      author = {O\v{s}ep, Aljo\v{s}a and Mehner, Wolfgang and Voigtlaender, Paul and Leibe, Bastian},
      title = {Track, then Decide: Category-Agnostic Vision-based Multi-Object Tracking},
      journal = {ICRA},
      year = {2018}
    }

    @article{Osep19arxiv,
      author = {O\v{s}ep, Aljo\v{s}a and Voigtlaender, Paul and Luiten, Jonathon and Weber, Mark and Leibe, Bastian},
      title = {4D Generic Video Object Proposals},
      journal = {arXiv preprint arXiv:TBA},
      year = {2019}
    }

## Potential Issues
* In case you want to use self-compiled libs, you may need to specify these paths (eg. edit cmake cache or use `ccmake`): `PCL_DIR`, `OpenCV_DIR`, `BOOST_ROOT`
* `CMake Error Unable to find the requested Boost libraries. Unable to find the Boost header files.  Please set BOOST_ROOT to the root directory containing Boost or BOOST_INCLUDEDIR to the directory containing Boost's headers.` For certain combinations of boost and cmake versions, it may happen cmake will not find all dependencies. Typically this will happen when using newer boost and older cmake; try using most recent cmake to avoid this issue.

## License

GNU General Public License (http://www.gnu.org/licenses/gpl.html)

Copyright (c) 2017 Aljosa Osep
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
