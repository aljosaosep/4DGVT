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

// std
#include <iostream>
#include <memory>
#include <cassert>

// opencv
#include <opencv2/highgui/highgui.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>

// tracking
#include <tracking/multi_object_tracker_base.h>
#include <tracking/visualization.h>

// scene segm.
#include <scene_segmentation/utils_segmentation.h>
#include <scene_segmentation/segmentation_visualization.h>
#include <scene_segmentation/json_prop_tools.h>
#include <scene_segmentation/parameters_gop3D.h>
#include <scene_segmentation/scene_segmentation.h>
#include <src/CAMOT/QPBO_fnc.h>
#include <tracking/qpbo.h>

// utils
#include "utils_io.h"
#include "utils_visualization.h"
#include "utils_pointcloud.h"
#include "ground_model.h"
#include "datasets_dirty_utils.h"
#include "utils_bounding_box.h"
#include "utils_common.h"

// CAMOT
#include "CAMOT/CAMOT_tracker.h"
#include "CAMOT/hypo_export.h"
#include "CAMOT/parameters_CAMOT.h"
#include "CAMOT/observation_processing_utils.h"
#include "CAMOT/inference.h"
#include "CAMOT/proposal_proc.h"

#include "CAMOT/batch_proc.h"


#define MAX_PATH_LEN 500

// For convenience.
namespace po = boost::program_options;
namespace CAMOT = GOT::tracking::CAMOT_tracker;
namespace object_tracking = GOT::tracking;
typedef pcl::PointCloud<pcl::PointXYZRGBA> pcloud;

namespace CAMOTApp {

    std::string config_parameters_file;
    std::string calib_path;

    // -------------------------------------------------------------------------------
    // +++ Command Args Parser +++
    // -------------------------------------------------------------------------------
    bool ParseCommandArguments(const int argc, const char **argv, po::variables_map &config_variables_map) {
        po::options_description cmdline_options;
        try {
            std::string config_file;

            // Declare a group of options that will be allowed only on command line
            po::options_description generic_options("Command line options:");
            generic_options.add_options()
                    ("help", "Produce help message")
                    ("config", po::value<std::string>(&config_file), "Config file path.")
                    ("config_parameters", po::value<std::string>(&config_parameters_file), "Config file path (parameters only!).")
                    ("debug_level", po::value<int>()->default_value(0), "Debug level")
                    ("dataset", po::value<std::string>()->default_value("kitti"), "Dataset (default: KITTI)")
                    ("eval_dir_kitti", po::value<std::string>(), "Output path: KITTI format evaluation.")
                    ("export_tracks", po::value<bool>()->default_value(false), "Export global track data?")
                    ("export_tracks_filename", po::value<std::string>(), "Export global track data?")                    ;

            // Declare a group of options that will be  allowed both on command line and in config file
            po::options_description config_options("Config options");
            config_options.add_options()
                    // Input
                    ("left_image_path", po::value<std::string>(), "Image (left) path")
                    ("right_image_path", po::value<std::string>(), "Image (right) path")
                    ("semantic_map_path", po::value<std::string>(), "Path to a semantic map.")
                    ("velodyne_path", po::value<std::string>(), "LiDAR data path (optional; only when doing LIDAR tracking)")
                    ("left_disparity_path", po::value<std::string>(), "Disparity (left) path")
                    ("flow_map_path", po::value<std::string>(), "Path to flow dir. (optional)")
                    ("calib_path", po::value<std::string>(&calib_path), "Camera calibration path (currently supported: kitti)")
                    ("ground_plane_path", po::value<std::string>(), "Ground-plane params.")

                    ("segmentations_json_file", po::value<std::string>(), "JSON file with segmentation masks.")
                    ("object_proposals_path", po::value<std::string>(), "Object proposals path (KITTI format)")

                    ("output_dir", po::value<std::string>(), "Output path")
                    ("eval_output_dir", po::value<std::string>(), "Output path: evaluation.")

                    ("start_frame", po::value<int>()->default_value(0), "Starting frame")
                    ("end_frame", po::value<int>()->default_value(10000), "Last frame")

                    ("cache_proposals", po::value<bool>()->default_value(true), "Cache proposals?")
                    ("run_tracker", po::value<bool>()->default_value(true), "Run tracker?")
                    ("subsequence", po::value<std::string>(), "Sub-sequence name")
                    ;

            po::options_description parameter_options("Parameters:");
            CAMOTApp::InitParameters(parameter_options);
            po::options_description parameters_proposals("Parameters-proposals:");
            po::options_description parameter_options_coco("Parameters-COCO:");
            GOP3D::InitDefaultParameters(parameter_options_coco);

            cmdline_options.add(generic_options);
            cmdline_options.add(config_options);
            cmdline_options.add(parameter_options);
            cmdline_options.add(parameters_proposals);
            cmdline_options.add(parameter_options_coco);

            store(po::command_line_parser(argc, argv).options(cmdline_options).run(), config_variables_map);
            notify(config_variables_map);

            if (config_variables_map.count("help")) {
                std::cout << cmdline_options << endl;
                return false;
            }

            // "generic" config
            if (config_variables_map.count("config")) {
                std::ifstream ifs(config_file.c_str());
                if (!ifs.is_open()) {
                    std::cout << "Can not Open config file: " << config_file << "\n";
                    return false;
                } else {
                    store(parse_config_file(ifs, cmdline_options), config_variables_map);
                    notify(config_variables_map);
                }
            }

            if (config_variables_map.count("config_parameters")) {
                // "parameter" config
                std::ifstream ifs_param(config_parameters_file.c_str());
                if (!ifs_param.is_open()) {
                    std::cout << "Can not Open parameter config file: " << config_parameters_file << "\n";
                    return false;
                } else {
                    store(parse_config_file(ifs_param, cmdline_options), config_variables_map);
                    notify(config_variables_map);
                }
            }
        }
        catch (std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << cmdline_options << std::endl;
            return false;
        }

        return true;
    }
}

/*
  -------------
  Debug Levels:
  -------------
  0 - Outputs basically nothing, except relevant error messages.
  1 - Console output, logging.
  2 - Quantitative evaluation.
  3 - Most relevant visual results (per-frame, eg. segmentation, tracking results, ...).
  4 - Point clouds (per-frame), less relevant visual results.
  5 - Additional possibly relevant frame output (segmentation 3D data, integrated models, ...).
  >=6 - All possible/necessary debug stuff. Should make everything really really really slow.
  */

using namespace CAMOTApp;

void RunSequence(const po::variables_map& variables_map) {

    const auto debug_level = variables_map.at("debug_level").as<int>();
    const auto start_frame = variables_map.at("start_frame").as<int>();
    const auto end_frame = variables_map.at("end_frame").as<int>();

    // -------------------------------------------------------------------------------
    // +++ OUTPUT DIRS +++
    // -------------------------------------------------------------------------------
    const auto output_dir = variables_map.at("output_dir").as<std::string>();
    if (debug_level > 0) printf("[Creating output dirs in:%s] \r\n", output_dir.c_str());
    const auto subsequence_name = variables_map.at("subsequence").as<std::string>();
    std::string output_dir_visual_results = output_dir + "/" + subsequence_name + "/visual_results";
    if (!SUN::utils::IO::MakeDir(output_dir_visual_results.c_str())) {
        throw std::runtime_error("Error, can not create output dirs.");
    }

    // -------------------------------------------------------------------------------
    // +++ GENERATE TRACKS +++
    // -------------------------------------------------------------------------------
    const clock_t tracklet_gen_begin_time = clock();
    auto tracklets_result = GOT::tracking::CAMOT_tracker::GenerateTracks(start_frame, end_frame, variables_map, output_dir_visual_results, debug_level);

    auto &exported_tracklets = tracklets_result.tracklets_;
    auto &cams = tracklets_result.cameras_;
    auto num_frames_actually_processed = tracklets_result.num_frames_processed_;

    auto cam0 = *(cams.begin());
    auto label_mapper = SUN::utils::GetCategoryLabelMap(variables_map.at("label_mapping").as<std::string>());

    // -------------------------------------------------------------------------------
    // +++ EXPORT PER-FRAME TRACK INFO FOR EVALUATION +++
    // -------------------------------------------------------------------------------
    auto f_export_per_frame_track_info_for_eval =
            [&variables_map, num_frames_actually_processed, start_frame, debug_level, subsequence_name]
                    (const std::vector<GOT::tracking::Hypothesis> &tracks_to_exp, const std::string &out_dir_name="json_per_frame") {
                if (variables_map.count("eval_output_dir") > 0) {
                    auto eval_out_json = variables_map.at("eval_output_dir").as<std::string>() + "/" + out_dir_name + "/" + subsequence_name;
                    if (debug_level>0) printf("[ Exporting eval data to: %s ] \r\n", eval_out_json.c_str());
                    SUN::utils::IO::MakeDir(eval_out_json.c_str());

                    for (int i = 0; i < num_frames_actually_processed; i++) {
                        int curr_frame = i + start_frame;
                        boost::filesystem::path p(variables_map.at("left_image_path").as<std::string>());
                        char fname_buff[100];
                        snprintf(fname_buff, 100, p.stem().c_str(), curr_frame);
                        char buff[500];
                        snprintf(buff, 500, "%s/%s.json", eval_out_json.c_str(), fname_buff);
                        GOT::tracking::CAMOT_tracker::SerializeHyposPerFrameJson(buff, curr_frame, tracks_to_exp);
                    }

                    if (debug_level>0) printf("[ Done exporting. ] \r\n");
                }
            };

    if (debug_level>0) printf("[ Exporting all hypos -> json ... ] \r\n");
    f_export_per_frame_track_info_for_eval(exported_tracklets, "tracklets_per_frame");

    // -------------------------------------------------------------------------------
    // +++ COMPUTE MAP + MERGE +++
    // -------------------------------------------------------------------------------
    const auto unary_fnc_str = variables_map.at("unary_fnc").as<std::string>();
    auto unary_fnc = GOT::tracking::CAMOT_tracker::GetUnaryFnc(unary_fnc_str);
    auto pwise_fnc = GOT::tracking::CAMOT_tracker::PairwiseFncIoU;

    const clock_t inference_begin_time = clock();
    if (variables_map.at("run_inference").as<bool>()) {

        // Compute MAP
        auto inference_results = GOT::tracking::CAMOT_tracker::RunMAP(
                start_frame, start_frame + num_frames_actually_processed, exported_tracklets, variables_map,
                unary_fnc, pwise_fnc
        );

        // Based on MAP result, loop over all frames, generate visual results
        // Export track results to files
        for (int i = 0; i < num_frames_actually_processed; i++) {
            const int curr_frame = i + start_frame;

            if (debug_level > 0) {
                cout << "-----------------------------------------" << endl;
                cout << "| CRF postproc, frame: " << curr_frame << endl;
                cout << "-----------------------------------------" << endl;
            }

            // Get left image and clone it -- needed for visualizations
            char left_image_path_buff[MAX_PATH_LEN];
            snprintf(left_image_path_buff, MAX_PATH_LEN, variables_map["left_image_path"].as<std::string>().c_str(),
                     curr_frame);
            cv::Mat viz_boxes = cv::imread(left_image_path_buff, cv::IMREAD_COLOR);
            auto viz_masks = viz_boxes.clone();

            auto selected_this_frame = inference_results.at(curr_frame); // Inds
            std::vector<GOT::tracking::Hypothesis> selected_this_frame_hypos_vec; // Hypos

            for (int sel_idx : selected_this_frame) {
                const auto hypo = exported_tracklets.at(sel_idx);
                if (hypo.cache().Exists(curr_frame)) {
                    // Export labels
                    if (!hypo.IsHypoTerminatedInFrame(curr_frame)) {
                        selected_this_frame_hypos_vec.push_back(hypo);

                        // Viz
                        if (debug_level >= 0) {
                            GOT::tracking::draw_hypos::DrawHypothesis2dForFrame(curr_frame, hypo, cams.at(curr_frame),
                                                                                viz_boxes);
                            GOT::tracking::draw_hypos::DrawHypothesisMaskForFrame(curr_frame, hypo, cams.at(curr_frame),
                                                                                  viz_masks);
                        }
                    }
                }
            }

            // Export selected -> json
            if (debug_level>0) printf("[ Exporting selected hypos -> json ... ] \r\n");
            f_export_per_frame_track_info_for_eval(selected_this_frame_hypos_vec, "selected_tracks_frame");

            // Export hypos -> qualitative
            if (debug_level >= 3) {
                char output_path_buff[500];
                snprintf(output_path_buff, 500, "%s/hypos_selected_%06d.png", output_dir_visual_results.c_str(),
                         curr_frame);
                cv::imwrite(output_path_buff, viz_boxes);
                snprintf(output_path_buff, 500, "%s/hypos_selected_masks_%06d.png", output_dir_visual_results.c_str(),
                         curr_frame);
                cv::imwrite(output_path_buff, viz_masks);
            }
        }
    }

    if (debug_level > 0) {
        printf("[ EXECUTION STATS - INFERENCE ]\r\n");
        printf("[ Processing time (inference for %d frames): %.3f s ]\r\n",
               num_frames_actually_processed, float(clock() - inference_begin_time) / CLOCKS_PER_SEC);
        printf("[ Processing time per-frame %.3f s ]\r\n",
               float((clock() - inference_begin_time) / num_frames_actually_processed) / CLOCKS_PER_SEC);
    }
}

int main(const int argc, const char** argv) {
    std::cout << "Hello from new and awesome generic tracker!" << std::endl;

    // -------------------------------------------------------------------------------
    // +++ Parse cmd args +++
    // -------------------------------------------------------------------------------
    po::variables_map variables_map;
    if (!CAMOTApp::ParseCommandArguments(argc, argv, variables_map)) {
        printf("Error parsing command args/configs, exiting.\r\n");
        return -1;
    }

    RunSequence(variables_map);

    std::cout << "Finished, yay!" << std::endl;
    return 0;
}
