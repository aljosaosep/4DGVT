##### WARNING: THIS INSTRUCTIONS ARE STILL BEING WRITTEN. STAY TUNED.

# 4D Generic Video Object Proposals

![Alt text](img/4dgvt.png?raw=true "Our method.")
![Alt text](img/3dviz.png?raw=true "Our method.")

This repository contains code, experimental data and **oxford-unknown** dataset for the work as described in

**4D Generic Video Object Proposals (https://arxiv.org/pdf/1901.09260.pdf)**

By [Aljosa Osep](https://www.vision.rwth-aachen.de/person/13/), [Paul Voigtlaender](https://www.vision.rwth-aachen.de/person/197/), Mark Weber, [Jonathon Luiten](https://www.vision.rwth-aachen.de/person/216/), [Bastian Leibe](https://www.vision.rwth-aachen.de/person/1/), Computer Vision Group, RWTH Aachen University

## TODO
- [x] Upload sequences surrounding labeled frames of Oxford Dataset
- [x] Upload result files
- [ ] Make sure that the uploaded verson of the tracker and configs reproduce the paper results
- [ ] Detailed instructions 

## Demo  Videos
* [CAMOT video (older version)](https://youtu.be/HYXzHuD4AKI)
* [CAMOT with Mask R-CNN](https://drive.google.com/open?id=1DlWWBcBTqBSPXY2c8UxdszruQcvNk8wn)
* [4DGVT video](https://drive.google.com/file/d/1gT1JqUJcN-pTm3cCKklBqv1Xf_eqAe4_/view?usp=sharing)


## Oxford-unknown dataset

For the labeling process, we manually selected 150 images of the [Oxford RobotCar dataset](https://robotcar-dataset.robots.ox.ac.uk/). The subset of images we labeled is available [here](https://drive.google.com/file/d/1WYwQD-FKj3xcgzEN7NTwqnJu7OS-UHxR/view?usp=sharing). 

Image sequences are available (temporal neighborhood of the annotated frames) [here](https://drive.google.com/file/d/1BfY92M8sQxCf4RUTjSS6VbPsad1X6FGN/view?usp=sharing).

Additional data for sequences (precomputed proposals) are available [here](https://drive.google.com/file/d/1gVQVIPOyM4ubi7gflI15wWdQoZfaL4Hf/view?usp=sharing).

We labeled 1,494 bounding boxes (1,081 *known*, 413 *unknown*) covering the visible portions of objects (non-amodal) by clicking the extremal points.

*Known* labeled classes (those that overlap with the [COCO](http://cocodataset.org/#home) classes) are *car, person, bike* and *bus*. In addition, we labeled several object classes that are not present in the COCO dataset, labeled as *unknown* objects. Most notable object classes in the unknown set are the following: *signage, pole, stone road sign, traffic cone, street sign, rubbish bin, transformer, post box, booth* and *stroller*. 

| **Category**   |  Car          | Person | Bike  | Bus   | Unknown | All   | 
| -------------- |:-------------:|:------:|:------:|:------:|:--------:|:------:|
| **#instances** | 599           | 354    | 78    | 50    | 413     | 1494  | 
| **Portion**    | 40.1%         | 23.1%  | 5.2%  | 3.3%  | 27.6%   | 100%  | 

### Baselines
We evaluated the performance of several methods on both known and unknown splits, see our paper for the details. All results will be available for download soon.

### Labels
Please find labels in `$REPO/eval/oxford_labels`. Labels are stored using JSON format. To evaluate recall, use the script `$REPO/eval/eval_single_image_proposals.py`. To see how to use this script, take a look at `$REPO/eval/run_evaluation.sh`.

Further instructions and descripton of the label format will be avalible soon. Until then, we recommend to step through `eval_single_image_proposals.py` script to understand the format.

### Prerequisite

* Proposal evaluation:
  * Python 3.x (for running `eval_single_image_proposals.py`)
  * pycocotools
  * matplotlib
  * numpy
* Tracking evaluation (in addition):
  * Python 2.7 (required by tracking evaluation legacy scripts)
  * munkres
  * tabulate

## Video Object Proposal Generator (Tracker)
### Prerequisite
In order to run the video object proposal generator code, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):
* cmake (tested with 3.9.6, earlier versions should work too)
* GCC 5.4.0
* Libs:
  * Eigen (3.x)
  * Boost (1.55 or later)
  * OpenCV (tested with 3.x, 4.x)
  * PCL (tested on 1.8.0 and 1.9.x) (note: requires FLANN and VTK for the 3D visualizer)


### Data
Note: any other paths will do too, you will need to adapt for that in the `$REPO/script/exec_tracker.sh`

0. Download [KITTI tracking dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and place it to `/home/${USER}/data/kitti_tracking`
0. Download [precomputed segmentations](https://drive.google.com/open?id=1AmDVzanSeHvmgJ4nh36jByOH-qIsib_2) we provide for KITTI tracking dataset, unzip to `/home/${USER}/data/kitti_tracking/preproc`
0. Clone this repo to `/home/${USER}/projects`

### Compiling the code
0.  `mkdir build && cd build`
0.  `cmake ..`
0.  `make all`

### Running the tracker
0.  Enter `$REPO/script/`
0.  Execute `exec_tracker.sh`

### Tracker execution script settings
* `SEGM_INPUTS` - Specify which pre-computed segmentations to use -- Mask Proposal R-CNN (`mprcnn_coco`; recommended), Sharpmask (`sharpmask_coco`), Mask R-CNN fine-tuned on KITTI (`mrcnn_tuned`) 
* `INF_MODEL` - Specify which model should be used for inference - `4DGVT` (recommended) or `CAMOT`.
* `INPUT_CONF_THRESH` - Detection/proposal score threshold. In case it is set to `0.8` or more, you will be only forwarding confident detections.
* `MAX_NUM_PROPOSALS` - Max. proposals fed to track generator per frame. More proposals -> slower, higher recall. Not recommended to be set above 500.

#### Remarks

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

* Run the tracker in `release` mode (otherwise, it will be slow).

If you have any issues or questions about this repository, please contact me at aljosa (dot) osep (at) tum.de

## Citing

If you find this repository useful in your research, please cite:

    @inproceedings{Osep18ICRA,
      author = {O\v{s}ep, Aljo\v{s}a and Mehner, Wolfgang and Voigtlaender, Paul and Leibe, Bastian},
      title = {Track, then Decide: Category-Agnostic Vision-based Multi-Object Tracking},
      booktitle = {ICRA},
      year = {2018}
    }

    @inproceedings{Osep19ICRA,
      author = {O\v{s}ep, Aljo\v{s}a and Voigtlaender, Paul and Weber, Mark and Luiten, Jonathon and Leibe, Bastian},
      title = {4D Generic Video Object Proposals},
      booktitle = ICRA,
      year = {2020}
    }

When using oxford-unknown labels, please cite the original dataset:

    @article{Maddern17IJRR,  
      Author = {Will Maddern and Geoff Pascoe and Chris Linegar and Paul Newman},  
      Title = {{1 Year, 1000km: The Oxford RobotCar Dataset}},  
      Journal = {The International Journal of Robotics Research (IJRR)},  
      Volume = {36},  
      Number = {1},  
      Pages = {3-15},  
      Year = {2017} 
    }
    

## Potential Issues
* In case you want to use self-compiled libs, you may need to specify these paths (e.g., edit CMake cache or use `ccmake`): `PCL_DIR`, `OpenCV_DIR`, `BOOST_ROOT`
* `CMake Error Unable to find the requested Boost libraries. Unable to find the Boost header files.  Please set BOOST_ROOT to the root directory containing Boost or BOOST_INCLUDEDIR to the directory containing Boost's headers.` For certain combinations of boost and CMake versions, it may happen CMake will not find all dependencies. Typically this will happen when using newer boost and older CMake; try using the most recent CMake to avoid this issue.
* I had issues compiling PCL with VTK 9.x, recommending to use VTK 8.x. 

## License

GNU General Public License (http://www.gnu.org/licenses/gpl.html)

Copyright (c) 2017 Aljosa Osep
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
