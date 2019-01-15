/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep (osep -at- vision.rwth-aachen.de)

rwth_mot framework is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

rwth_mot framework is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
rwth_mot framework; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#ifndef GOT_PARAMETERS_CAMOT_H_H
#define GOT_PARAMETERS_CAMOT_H_H


#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace CAMOTApp {

    void InitParameters(po::options_description &options) {

        options.add_options()

                ///  --- Object Proposals preproc. ---

                ("proposals_do_geom_filtering", po::value<bool>()->default_value(true),
                 "Apply geom. filter (3D) on proposals?")
                ("proposals_geometric_filter_lateral", po::value<double>()->default_value(50.0),
                 "Geometric filter: lateral")
                ("proposals_geometric_filter_far", po::value<double>()->default_value(60.0), "Geometric filter: far")
                ("proposals_geometric_filter_near", po::value<double>()->default_value(0.0), "Geometric filter: near")
                ("proposals_geometric_min_distance_to_plane", po::value<double>()->default_value(0.1),
                 "Geometric filter: min. distance to plane")
                ("proposals_geometric_max_distance_to_plane", po::value<double>()->default_value(5.0),
                 "Geometric filter: max. distance to plane")

                ///  --- Tracking Etc ---
                ("tracking_model_accepted_frames_without_inliers", po::value<int>()->default_value(5),
                 "Max. number of frames for track extrapolation")
                ("tracking_temporal_window_size", po::value<int>()->default_value(5),
                 "Temporal window size")
                ("tracking_non_max_supp_threshold", po::value<double>()->default_value(0.8),
                 "Track non-max-supp threshold")
                ("tracklets_min_inliers_to_init_tracklet", po::value<int>()->default_value(3),
                 "Min. inliers needed to start a tracklet")
                ("tracking_selective_hypothesis_initialization", po::value<bool>()->default_value(true),
                 "Start new hypotheses only from unassigned observations")
                ("tracking_suppression_front", po::value<int>()->default_value(4),
                 "Track suppression parameter")

                ("tracking_use_filtered_velocity_for_mask_warping", po::value<bool>()->default_value(true),
                 "Use filtered velocities for filtering?")
                ("tracking_distance_based_segment_filter", po::value<bool>()->default_value(true),
                 "Remove far-away points from the 3D segments?")
                ("tracking_fill_holes_predicted_masks", po::value<bool>()->default_value(true),
                 "Fill holes in predicted masks via dilation/erosion?")

                ///  --- CRF ---
                // General model
                ("inf_e1", po::value<double>()->default_value(5.0), "CRF -- unary")
                ("inf_e2", po::value<double>()->default_value(20.0), "CRF -- pairwise")
                ("inf_window_size", po::value<int>()->default_value(40), "CRF inf. window size")

                // CAMOT unary fnc
                ("inf_w1", po::value<double>()->default_value(0.8), "CAMOT unary w_1")
                ("inf_w2", po::value<double>()->default_value(0.2), "CAMOT unary w_2")
                ("inf_tau", po::value<int>()->default_value(60), "Temporal decay parameter")

                // 4D-GVT unary fnc
                ("inf_alpha", po::value<double>()->default_value(0.5), "4DGVT w_1")
                ("inf_beta", po::value<double>()->default_value(0.5), "4DGVT w_2")

                // Unary selector
                ("unary_fnc", po::value<std::string>()->default_value("4DGVT"), "Which unary function? [4DGVT | CAMOT]")

                ///  --- Tracking Area ---
                ("tracking_exit_zones_rear_distance", po::value<double>()->default_value(1.3), "Tracking range - min")
                ("tracking_exit_zones_far_distance", po::value<double>()->default_value(80.0), "Tracking range - max")
                ("tracking_exit_zones_lateral_distance", po::value<double>()->default_value(0.3),
                 "Tracking range - lateral")

                ///  --- Data Association and Gaiting ---
                ("data_association_min_association_score", po::value<double>()->default_value(0.02),
                 "Data association: minimal assoc. score (combined).")
                ("gaiting_motion_model_threshold", po::value<double>()->default_value(5.0), "Gaiting: motion model")
                ("gaiting_IOU_threshold", po::value<double>()->default_value(0.5), "Gaiting: mask overlap [0...1].")
                ("gaiting_IOU_rect_threshold", po::value<double>()->default_value(0.3),
                 "Gaiting: bounding-box-2D overlap [0...1].")
                ("gaiting_size_2D", po::value<double>()->default_value(0.6), "Gaiting: bounding-box-2D size.")

                ///  --- Etc ---
                ("closing_op_kernel_size", po::value<int>()->default_value(4),
                 "Kernel size for dilation/ersion (segment prediction)")
                ("run_inference", po::value<bool>()->default_value(true), "Run inference at the end?")
                ("do_track_nms", po::value<bool>()->default_value(true), "Perform track suppression?")

                /// --- Proposals ---
                ("proposals_max_number_of_proposals", po::value<int>()->default_value(300),
                 "Max. num. of proposals (per-frame).")
                ("proposals_confidence_thresh", po::value<double>()->default_value(0.0),
                 "Proposals: confidence threshold.")

                ("label_mapping", po::value<std::string>()->default_value("coco"), "Labels mapping: coco, kitti")
                ("debug_mode", po::value<bool>()->default_value(false), "Debug mode on/off.")

                ("flow_type", po::value<std::string>()->default_value("sparse"), "Flow type [dense|sparse]")
                ("compute_visual_odometry", po::value<bool>()->default_value(true), "Compute VO online?")
                ("process_proposals_and_tracker", po::value<bool>()->default_value(true),
                 "Process proposals and tracker?")

                ///  --- Kalman Filter ---
                ("dt", po::value<double>()->default_value(0.1),
                 "Time difference between two frames, captured by the sensor (sec.)")
                ("use_flow", po::value<bool>()->default_value(true), "Use scene-flow as velocity observation?");
    }
}

#endif //GOT_PARAMETERS_CAMOT_H_H
