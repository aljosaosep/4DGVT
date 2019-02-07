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
#include <stdexcept>

// opencv
#include <opencv2/highgui/highgui.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

// tracking
#include <tracking/visualization.h>
#include <tracking/multi_object_tracker_base.h>

// scene segm.
#include <scene_segmentation/utils_segmentation.h>
#include <scene_segmentation/segmentation_visualization.h>
#include <scene_segmentation/json_prop_tools.h>
#include <scene_segmentation/scene_segmentation.h>
#include <src/CAMOT/QPBO_fnc.h>
#include <tracking/qpbo.h>

// utils
#include "utils_io.h"
#include "utils_visualization.h"
#include "utils_observations.h"
#include "utils_pointcloud.h"
#include "ground_model.h"
#include "datasets_dirty_utils.h"
#include "utils_bounding_box.h"
#include "utils_common.h"
#include "utils_flow.h"

// CAMOT
#include "CAMOT/CAMOT_tracker.h"
#include "CAMOT/data_association.h"
#include "CAMOT/hypo_export.h"
#include "CAMOT/observation_processing_utils.h"
#include "CAMOT/inference.h"
#include "CAMOT/proposal_proc.h"

#include "batch_proc.h"

// For convenience.
namespace po = boost::program_options;
namespace CAMOT = GOT::tracking::CAMOT_tracker;
namespace object_tracking = GOT::tracking;
typedef pcl::PointCloud<pcl::PointXYZRGBA> pcloud;

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {

            /*!
             * @brief Try loading cached proposals, if they don't exist, compute them using specified 'proposal_gen_fnc'.
             * @return Boolean (failure/success)
             */
            bool GetObjectProposals(int frame, const std::string &proposals_path,
                                    const po::variables_map &options_map,
                                    std::function<GOT::segmentation::ObjectProposal::Vector(
                                            const po::variables_map &)> proposal_gen_fnc,
                                    GOT::segmentation::ObjectProposal::Vector &proposals_out) {
                char proposal_path_buff[500];
                snprintf(proposal_path_buff, 500, proposals_path.c_str(), frame);

                if (options_map.at("cache_proposals").as<bool>()) {
                    std::cout << "[ Caching proposals ... ]" << std::endl;
                    auto success_loading_proposals = GOT::segmentation::utils::DeserializeJson(proposal_path_buff,
                                                                                               proposals_out);

                    if (!success_loading_proposals) {
                        printf("Could not load proposals, computing (note: processing will slow-down!) ...\r\n");
                        proposals_out = proposal_gen_fnc(options_map);
                    }
                }

                return true;
            }

            std::map<int, std::vector<int> >
            RunMAP(int start_frame, int end_frame, const std::vector<GOT::tracking::Hypothesis> &tracklets,
                   const po::variables_map &variables_map,
                   f_unary unary_fnc,
                   f_pairwise pairwise_fnc,
                   int debug_level) {

                if (debug_level > 0) printf("[ Run CRF ... ] \r\n");
                const int num_frames = end_frame - start_frame;
                std::map<int, std::vector<int> > inference_results;
                const clock_t crf_begin_time = clock();
                for (int curr_frame = start_frame; curr_frame <= end_frame; curr_frame++) {
                    if (debug_level > 0) printf(". ");
                    std::vector<int> selected_this_frame = GOT::tracking::CAMOT_tracker::InferStateForFrame(
                            curr_frame,
                            tracklets,
                            unary_fnc, pairwise_fnc,
                            variables_map
                    );
                    inference_results.insert(std::make_pair(curr_frame, selected_this_frame));
                }
                if (debug_level > 0) printf("\r\n");
                if (debug_level > 0) printf("[ Inference done. ] \r\n");
                if (debug_level > 0) {
                    printf("\r\n");
                    printf("[ +++ CRF +++ ]\r\n");
                    printf("[ Processing time (tracklet-gen for %d frames): %.3f s ]\r\n", num_frames,
                           float(clock() - crf_begin_time) / CLOCKS_PER_SEC);
                    printf("[ Processing time per-frame %.3f s ]\r\n",
                           float((clock() - crf_begin_time) / num_frames) / CLOCKS_PER_SEC);
                    printf("\r\n");
                }

                return inference_results;
            }


            TrackletsResult GenerateTracks(int start_frame, int end_frame,
                                           const po::variables_map &variables_map, const std::string &output_dir,
                                           int debug_level) {

                // Const globals
                const int num_frames = end_frame - start_frame;
                const auto dataset_str = variables_map.at("dataset").as<std::string>();
                const auto subsequence_name = variables_map.at("subsequence").as<std::string>();
                const auto run_tracker = variables_map.at("run_tracker").as<bool>();
                const auto max_num_proposals = variables_map.at("proposals_max_number_of_proposals").as<int>();
                const auto use_flow = variables_map.at("use_flow").as<bool>();

                // Other globals
                Eigen::Matrix4d g_egomotion = Eigen::Matrix4d::Identity();
                std::map<int, SUN::utils::Camera, std::less<int>,
                         Eigen::aligned_allocator<std::pair<const int, SUN::utils::Camera> > > g_left_cameras_all; // Need for post-CRF viz.
                GOT::segmentation::ObjectProposal::Vector object_proposals;
                std::vector<std::vector<double>> all_poses; // For export of poses of tracked objects

                // Track stats
                int total_num_hypos = 0;
                int total_num_observations = 0;

                // Makes sure the relevant values are correct.
                assert(num_frames > 0);
                assert(debug_level >= 0);

                // -------------------------------------------------------------------------------
                // +++ Init the data loader +++
                // -------------------------------------------------------------------------------
                std::cout << "Init dataset assistant ..." << std::endl;
                SUN::utils::dirty::DatasetAssitantDirty dataset_assistant(variables_map);

                // -------------------------------------------------------------------------------
                // +++ Init matcher(s) +++
                // -------------------------------------------------------------------------------
                std::shared_ptr<libviso2::VisualOdometryStereo> vo_module = nullptr;
                libviso2::Matrix pose = libviso2::Matrix::eye(4);
                Eigen::Matrix<double, 4, 4> currPose = Eigen::MatrixXd::Identity(4, 4); // Current pose
                auto *matcher = SUN::utils::scene_flow::InitMatcher();

                // -------------------------------------------------------------------------------
                // +++ Per-frame containers +++
                // -------------------------------------------------------------------------------
                cv::Mat left_image;
                pcloud::Ptr left_point_cloud;
                Eigen::Matrix4d vo_matrix;
                SUN::utils::Camera prev_camera;

                // -------------------------------------------------------------------------------
                // +++ Resource Manager +++
                // -------------------------------------------------------------------------------
                if (debug_level > 0) printf("[Init resource manager ...]\r\n");
                assert(variables_map.count("tracking_temporal_window_size"));
                const int temporal_window_size = variables_map.at("tracking_temporal_window_size").as<int>();
                std::shared_ptr<GOT::tracking::DataQueue> tracking_resources(
                        new GOT::tracking::DataQueue(temporal_window_size)
                );

                // -------------------------------------------------------------------------------
                // +++ Init tracker object +++
                // -------------------------------------------------------------------------------
                /// Create the object-tracker object
                std::unique_ptr<CAMOT::CAMOTTracker> object_tracker(
                        new CAMOT::CAMOTTracker(variables_map)
                );

                /// And tracking visualizer
                GOT::tracking::Visualizer tracking_visualizer;

                /// Bind either mask-based assoc. scoring func. or bbox-based. The one you like better.
                auto data_assoc_fnc = std::bind(
                        CAMOT::data_association::data_association_motion_mask,
                        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        variables_map
                );

                object_tracker->set_data_association_fnc(data_assoc_fnc);
                object_tracker->set_verbose(debug_level > 2);

                // -------------------------------------------------------------------------------
                // +++ MAIN_LOOP +++
                // -------------------------------------------------------------------------------
                int num_frames_actually_processed = 0;
                const clock_t tracklet_gen_begin_time = clock();
                for (int current_frame = start_frame; current_frame <= end_frame; current_frame++) {
                    const clock_t current_frame_begin_time = clock();

                    if (debug_level > 0) {
                        printf("---------------------------\r\n");
                        printf("| PROC FRAME %03d/%03d    |\r\n", current_frame, end_frame);
                        printf("---------------------------\r\n");

                    }

                    // -------------------------------------------------------------------------------
                    // +++ Load data +++
                    // -------------------------------------------------------------------------------
                    if (debug_level > 0) printf("[ Loading data ... ] \r\n");
                    if (!dataset_assistant.LoadData(current_frame, dataset_str)) {
                        if (num_frames_actually_processed > 5) {
                            printf("WARNING: could not load data for frame %d, "
                                           "will write-out the results and terminate. \r\n", current_frame);
                            break;
                        } else {
                            SUN::utils::scene_flow::FreeMatcher(matcher);
                            throw std::runtime_error("Dataset assistant failed to load the data.");
                        }
                    }

                    // Sanity checks
                    if (!(dataset_assistant.got_left_image_ && dataset_assistant.got_left_camera_
                          && dataset_assistant.got_right_camera_)) {
                        SUN::utils::scene_flow::FreeMatcher(matcher);
                        throw std::runtime_error("One of the required resources not successfully loaded.");
                    }

                    left_image = dataset_assistant.left_image_.clone();
                    left_point_cloud.reset(new pcloud);
                    SUN::utils::Camera &left_camera = dataset_assistant.left_camera_;
                    SUN::utils::Camera &right_camera = dataset_assistant.right_camera_;
                    GOT::tracking::Observation::Vector observations_to_pass_to_tracker;

                    // -------------------------------------------------------------------------------
                    // +++ Generic proposal generators +++
                    // -------------------------------------------------------------------------------
                    std::string json_path;
                    if (variables_map.count("segmentations_json_file")) {
                        char json_path_buff[500];
                        snprintf(json_path_buff, 500,
                                 variables_map["segmentations_json_file"].as<std::string>().c_str(), current_frame);
                        json_path = std::string(json_path_buff);
                    }

                    // @Dan: std::bind always copy or move its arguments, i.e. it cannot pass by reference.
                    // Is passing by value desired behavior here?
                    // See https://stackoverflow.com/questions/26187192/how-to-bind-function-to-an-object-by-reference
                    // std::function do not respect custom alignment of captured variables.
                    // If the input variable needs to be aligned (left_camera, right_camera), they have to be captured by reference.
                    // See https://stackoverflow.com/questions/44318653/segmentation-fault-in-capturing-aligned-variables-in-lambdas
                    // In general, I would get rid of this std::function usage, as it is not necessary.
                    SUN::utils::Camera left_camera_copy = left_camera;
                    SUN::utils::Camera right_camera_copy = right_camera;
                    auto proposal_gen_fnc = std::bind(GOT::segmentation::proposal_generation::ProposalsFromJson,
                                                      current_frame,
                                                      json_path.c_str(),
                                                      std::ref(left_camera_copy),
                                                      std::ref(right_camera_copy),
                                                      left_point_cloud,
                                                      std::placeholders::_1,
                                                      1000);

                    pcl::copyPointCloud(*dataset_assistant.left_point_cloud_, *left_point_cloud);

                    // -------------------------------------------------------------------------------
                    // +++ Estimate egomotion +++
                    // -------------------------------------------------------------------------------
                    Eigen::Matrix4d ego_estimate = Eigen::Matrix4d::Identity();
                    bool ego_success = true;
                    if (variables_map.at("compute_visual_odometry").as<bool>() && run_tracker) {
                        if (dataset_assistant.got_right_image_) {
                            if (debug_level > 0) printf("[ Estimating odometry ...] \r\n");
                            SUN::utils::scene_flow::InitVO(vo_module, left_camera.f_u(),
                                                           left_camera.c_u(), left_camera.c_v(),
                                                           dataset_assistant.stereo_baseline_);
                            ego_estimate = SUN::utils::scene_flow::EstimateEgomotion(*vo_module,
                                                                                     dataset_assistant.left_image_,
                                                                                     dataset_assistant.right_image_);

                            // Accumulated transformation
                            g_egomotion = g_egomotion * ego_estimate.inverse();

                            // Update left_camera, right_camera using estimated pose transform
                            left_camera.ApplyPoseTransform(g_egomotion);
                            right_camera.ApplyPoseTransform(g_egomotion);
                        } else {
                            SUN::utils::scene_flow::FreeMatcher(matcher);
                            throw std::runtime_error("Egomotion estimation failed!");
                        }
                    }


                    // -------------------------------------------------------------------------------
                    // +++ Estimate ground-plane +++
                    // -------------------------------------------------------------------------------
                    std::shared_ptr<SUN::utils::PlanarGroundModel> planar_ground_model(
                            new SUN::utils::PlanarGroundModel);
                    planar_ground_model->FitModel(left_point_cloud, 1.4);
                    left_camera.set_ground_model(planar_ground_model);
                    right_camera.set_ground_model(planar_ground_model);

                    // Update 'global' cam maps (need for post-CRF tracking viz)
                    auto left_copy = left_camera;
                    g_left_cameras_all.insert(std::make_pair(current_frame, left_copy));

                    // -------------------------------------------------------------------------------
                    // +++ Compute Sparse Scene Flow +++
                    // -------------------------------------------------------------------------------
                    assert(variables_map.count("dt") > 0.0);
                    std::vector<SUN::utils::scene_flow::VelocityInfo> sparse_flow_info;
                    cv::Mat sparse_flow_map;
                    bool first_frame = current_frame <= start_frame;
                    bool use_sparse_flow = variables_map.at("flow_type").as<std::string>() == "sparse";
                    if (use_sparse_flow && run_tracker) {
                        if (dataset_assistant.got_right_image_) {
                            const clock_t sceneflow_start = clock();
                            if (debug_level > 0) printf("[ Computing sparse scene flow ... ] \r\n");
                            auto matches = SUN::utils::scene_flow::GetMatches(
                                    matcher,
                                    left_image,
                                    dataset_assistant.right_image_,
                                    left_camera,
                                    first_frame
                            );

                            if (!first_frame) {
                                auto flow_result = SUN::utils::scene_flow::GetSceneFlow(matches,
                                                                                        ego_estimate,
                                                                                        left_camera,
                                                                                        static_cast<float>(dataset_assistant.stereo_baseline_),
                                                                                        variables_map.at(
                                                                                                "dt").as<double>()
                                );
                                sparse_flow_info = std::get<1>(flow_result);
                                sparse_flow_map = std::get<0>(flow_result);
                            }

                        } else {
                            SUN::utils::scene_flow::FreeMatcher(matcher);
                            throw std::runtime_error(
                                    "You need left and right camera image to compute sparse scene flow!");
                        }
                    }

                    GOT::segmentation::ObjectProposal::Vector loaded_proposals_for_viz_only;

                    // -------------------------------------------------------------------------------
                    // +++ Observation Processing + Tracking / Tracklet Generation +++
                    // -------------------------------------------------------------------------------
                    if (variables_map.at("process_proposals_and_tracker").as<bool>()) {
                        const clock_t proposals_start = clock();


                        // -------------------------------------------------------------------------------
                        // +++ OBJECT PROPOSAL PROCESSING +++
                        // -------------------------------------------------------------------------------

                        const clock_t proposals_begin_time = clock();

                        /// Fetch object proposals
                        if (debug_level > 0) printf("[ Fetching proposals ... ] \r\n");
                        bool got_prop_path = variables_map.count("object_proposals_path") > 0;
                        const auto &prop_path = variables_map.at("object_proposals_path").as<std::string>();
                        if (!(got_prop_path &&
                              GetObjectProposals(current_frame, prop_path,
                                                 variables_map, proposal_gen_fnc, object_proposals))) {
                            SUN::utils::scene_flow::FreeMatcher(matcher);
                            throw std::runtime_error("Failed to fetch object proposals!");
                        }

                        loaded_proposals_for_viz_only = object_proposals;

                        auto num_loaded_props = object_proposals.size();
                        if (debug_level > 0)
                            printf("[ Got %d proposals. ] \r\n", static_cast<int>(object_proposals.size()));

                        /// Filter-out < 100px
                        if (debug_level > 0) printf("[ Proposal filtering ...] \r\n");
                        GOT::segmentation::ObjectProposal::Vector obj_prop_area_filt;
                        for (const auto &p : object_proposals) {
                            Eigen::Vector4d prop_bbox = p.bounding_box_2d();
                            if (prop_bbox[2] * prop_bbox[3] < 100) {
                                continue;
                            }

                            obj_prop_area_filt.push_back(p);
                        }

                        object_proposals = obj_prop_area_filt;

                        /// Confidence-based filtering
                        double proposals_confidence_thresh = variables_map.at(
                                "proposals_confidence_thresh").as<double>();
                        object_proposals = GOT::tracking::CAMOT_tracker::proposal_utils::ProposalsConfidenceFilter(
                                object_proposals,
                                proposals_confidence_thresh);

                        if (debug_level > 0) {
                            printf("[ Keeping %d proposals after conf. filtering. ] \r\n",
                                   static_cast<int>(object_proposals.size()));
                        }

                        /// Perform geometric filtering
                        if (variables_map.at("proposals_do_geom_filtering").as<bool>()) {
                            if (debug_level > 0) printf("[ Geometric filtering ... ] \r\n");
                            object_proposals = GOT::tracking::CAMOT_tracker::proposal_utils::GeometricFiltering(
                                    left_camera,
                                    object_proposals,
                                    variables_map
                            );
                        }

                        /// Make sure proposals are sorted
                        std::sort(object_proposals.begin(), object_proposals.end(),
                                  [](const GOT::segmentation::ObjectProposal &a,
                                     const GOT::segmentation::ObjectProposal &b) { return a.score() > b.score(); });

                        /// Keep K-best scoring
                        GOT::segmentation::ObjectProposal::Vector prop_tmp;
                        prop_tmp.insert(
                                prop_tmp.begin(),
                                object_proposals.begin(),
                                object_proposals.begin() +
                                std::min(static_cast<int>(object_proposals.size()), max_num_proposals)
                        );
                        object_proposals = prop_tmp;


                        if (debug_level > 0)
                            printf("[ %d survived the geom. filtering.] \r\n",
                                   static_cast<int>(object_proposals.size()));
                        if (debug_level > 0)
                            printf("[ Processing proposals I. %.3f s ]\r\n",
                                   float((clock() - proposals_start)) / CLOCKS_PER_SEC);

                        /// Cache proposals (note: this is highly hacky!)
                        if (variables_map.at("cache_proposals").as<bool>()) {
                            if (num_loaded_props != object_proposals.size()) {
                                char proposal_path_buff[500];
                                snprintf(proposal_path_buff, 500, prop_path.c_str(), current_frame);
                                std::cout << "[ cache_proposals: Caching proposals to: " << proposal_path_buff << " ]"
                                          << std::endl;

                                boost::filesystem::path prop_path(proposal_path_buff);
                                boost::filesystem::path prop_dir = prop_path.parent_path();

                                if (!SUN::utils::IO::MakeDir(prop_dir.c_str())) {
                                    SUN::utils::scene_flow::FreeMatcher(matcher);
                                    throw std::runtime_error("Could not create dir.");
                                }

                                //std::cout << "Serializing obj proposals to: " << proposal_path_buff << std::endl;
                                if (!GOT::segmentation::utils::SerializeJson(proposal_path_buff, object_proposals)) {
                                    SUN::utils::scene_flow::FreeMatcher(matcher);
                                    throw std::runtime_error("Could not save obj proposals.");
                                }
                            }
                        }

                        printf("[ Processing time proposals: %.3f s ]\r\n",
                               float(clock() - proposals_begin_time) / CLOCKS_PER_SEC);

                        // -------------------------------------------------------------------------------
                        // +++ OBSERVATION PROCESSING +++
                        // -------------------------------------------------------------------------------

                        // Turn proposals into 'observations' that can be passed to the tracker
                        if (debug_level >= 1) printf("[ Running: ProcessObservations ...] \r\n");
                        observations_to_pass_to_tracker = GOT::tracking::CAMOT_tracker::obs_proc::ProcessObservations(
                                left_point_cloud,
                                object_proposals,
                                left_camera
                        );

                        /// Don't need proposal inds anymore -> release memory
                        for (auto &prop : object_proposals) {
                            prop.free();
                        }

                        /// Compute obs. velocities using velocity/flow maps
                        cv::Mat velocity_map_to_use = sparse_flow_map;
                        if (use_flow) {
                            observations_to_pass_to_tracker = GOT::tracking::CAMOT_tracker::obs_proc::ComputeObservationVelocity(
                                    observations_to_pass_to_tracker, velocity_map_to_use,
                                    variables_map.at("dt").as<double>()
                            );
                        }

                        // -------------------------------------------------------------------------------
                        // +++ Make a 'tracking step' +++
                        // -------------------------------------------------------------------------------

                        /// Run the tracker
                        if (run_tracker) {
                            if (debug_level > 0) {
                                printf("[ Running the tracker (passing %d observations)... ] \r\n",
                                                        static_cast<int>(observations_to_pass_to_tracker.size()));
                            }

                            // Add current-frame detections to the resource manager
                            tracking_resources->AddNewMeasurements(current_frame, left_point_cloud, left_camera);
                            tracking_resources->AddNewObservations(observations_to_pass_to_tracker);
                            tracking_resources->AddEgoEstimate(ego_estimate);

                            const clock_t tracker_start = clock();
                            object_tracker->ProcessFrame(tracking_resources, current_frame);
                            if (debug_level > 0) {
                                printf("[ Tracker step %.3f s ]\r\n",
                                       float((clock() - tracker_start)) / CLOCKS_PER_SEC);
                            }
                        }
                    }

                    // At this point, we are done with tracklet generation.

                    // -------------------------------------------------------------------------------
                    // +++ Update visualizations +++
                    // -------------------------------------------------------------------------------
                    auto f_draw_hypos_and_save = [&tracking_visualizer, &left_image, output_dir, current_frame, &left_camera]
                            (const std::string &name,
                             const std::vector<GOT::tracking::Hypothesis> &hypos,
                             GOT::tracking::DrawHypoFnc f_draw_hypo,
                             float scale_fct = 1.0) {

                        auto im_draw = left_image.clone();
                        char output_path_buff[500];
                        tracking_visualizer.DrawHypotheses(hypos, left_camera, im_draw, f_draw_hypo);
                        cv::resize(im_draw, im_draw, cv::Size(0, 0), scale_fct, scale_fct);
                        snprintf(output_path_buff, 500, "%s/%s_%06d.png",
                                 output_dir.c_str(),
                                 name.c_str(),
                                 current_frame);
                        cv::imwrite(output_path_buff, im_draw);
                    };


                    const auto &hypos_conf = object_tracker->GetConfidentHypotheses();

                    /// Viz: save only essentials (hypos 2D, down-scaled)
                    if (debug_level >= 2) {
                        printf("[ Exporting qualitative results ... ] \r\n");
                        f_draw_hypos_and_save(
                                "hypos_all", hypos_conf, object_tracking::draw_hypos::DrawHypothesis2d, 1.0);
                        f_draw_hypos_and_save("hypos_all_2d_masks", hypos_conf,
                                              object_tracking::draw_hypos::DrawHypothesisMask, 1.0);
                        f_draw_hypos_and_save("hypos_all_3D", hypos_conf, object_tracking::draw_hypos::DrawHypothesis3d,
                                              1.0);
                    }

                    // -------------------------------------------------------------------------------
                    // +++ Update stats and visualizations +++
                    // -------------------------------------------------------------------------------


                    /// Update stats
                    total_num_hypos += static_cast<int>(hypos_conf.size());
                    total_num_observations += static_cast<int>(observations_to_pass_to_tracker.size());

                    auto label_mapper = SUN::utils::GetCategoryLabelMap(variables_map.at("label_mapping").as<std::string>());

                    prev_camera = left_camera;
                    num_frames_actually_processed++;
                    printf("***** Processing time current frame: %.3f s *****\r\n",
                           float(clock() - current_frame_begin_time) / CLOCKS_PER_SEC);
                }

                /// Release matcher memory
                SUN::utils::scene_flow::FreeMatcher(matcher);
                matcher = nullptr;
                vo_module.reset();

                // -------------------------------------------------------------------------------
                // +++ END OF MAIN_LOOP +++
                // -------------------------------------------------------------------------------

                /// Important: add all still-active (non-term.) tracklets to the 'exported' set
                object_tracker->AppendActiveTrackletsToExported();
                auto exported_tracklets = object_tracker->exported_tracklets();
                object_tracker->ComputeUnariesGlobal(exported_tracklets);
                object_tracker.release();

                /// Print tracking stats
                printf("\r\n=========================\r\n\r\n");
                printf("[ TRACKLET GENERATION - DONE ]\r\n");
                printf("[ Num. total tracklets: %d ]\r\n", static_cast<int>(exported_tracklets.size()));
                printf("[ Processing time (tracklet-gen for %d frames): %.3f s ]\r\n",
                       num_frames,
                       float(clock() - tracklet_gen_begin_time) / CLOCKS_PER_SEC);

                printf("[ Processing time per-frame %.3f s ]\r\n",
                       float((clock() - tracklet_gen_begin_time) / num_frames) / CLOCKS_PER_SEC);

                printf("[ Num active tracklets / frame: %d ]\r\n", total_num_hypos / num_frames);
                printf("[ Num. observations / frame: %d ]\r\n", total_num_observations / num_frames);

                return TrackletsResult(exported_tracklets, g_left_cameras_all, num_frames_actually_processed);
            }
        }
    }
}
