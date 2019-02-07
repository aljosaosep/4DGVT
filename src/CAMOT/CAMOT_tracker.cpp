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

// sys
#include <thread>

// opencv
#include <opencv2/highgui/highgui.hpp>

// eigen
#include <Eigen/Geometry>

// std
#include <unordered_map>

// pcl
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

// tracking
#include <tracking/qpbo.h>
#include <pcl/io/io.h>
#include <numeric>

// utils
#include "utils_bounding_box.h"
#include "ground_model.h"
#include "CAMOT_tracker.h"
#include "kalman_filter_const_velocity_const_elevation.h"
#include "utils_common.h"
#include "QPBO_fnc.h"
#include "inference.h"

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {

            // -------------------------------------------------------------------------------
            // +++ 'MAIN' PROCESSING FUNC. +++
            // -------------------------------------------------------------------------------

            /*
             * @brief: Hypothesis Non-Maxima-Suppression
             * @Warning: Assumes hypos are scored! Internally, uses Hypothesis::score()
             */
            std::vector<Hypothesis>
            HypothesisNonMaximaSuppression(const std::vector<Hypothesis> &input_hypos, double iou_threshold) {

                std::vector<Hypothesis> active_set = input_hypos;
                std::vector<Hypothesis> set_suppressed;

                while (active_set.size() > 0) {

                    // Get best-scoring element
                    auto it_to_max_element = std::max_element(active_set.begin(), active_set.end(),
                                                              [](const Hypothesis &a, const Hypothesis &b) {
                                                                  return a.score() < b.score();
                                                              });
                    const Hypothesis &best_scoring_hypothesis = *it_to_max_element; // This is the best-scoring element!

                    std::set<int> overlapping_set_indices; // Here we store indices of 'overlapping' objects
                    for (int i = 0; i < active_set.size(); i++) {
                        const auto &element_to_test = active_set.at(i);

                        // Test how much hypo len. differs
                        size_t len_longer = 0;
                        size_t len_smaller = 0;
                        if (best_scoring_hypothesis.cache().timestamps().size() >
                            element_to_test.cache().timestamps().size()) {
                            len_longer = best_scoring_hypothesis.cache().timestamps().size();
                            len_smaller = element_to_test.cache().timestamps().size();
                        } else {
                            len_smaller = best_scoring_hypothesis.cache().timestamps().size();
                            len_longer = element_to_test.cache().timestamps().size();
                        }

                        double len_ratio = static_cast<double>(len_smaller) / static_cast<double>(len_longer);

                        if (len_ratio < 0.7)
                            continue;

                        const double sum_iou = ComputePhysicalOverlap_IOU_mask(best_scoring_hypothesis,
                                                                               element_to_test);
                        auto longer_hypo = std::max(best_scoring_hypothesis.inliers().size(),
                                                    element_to_test.inliers().size());
                        const double normalized_iou = sum_iou / static_cast<double>(longer_hypo);

                        if (normalized_iou > iou_threshold) {
                            overlapping_set_indices.insert(i);
                        }
                    }

                    // Add 'best' hypo to the suppressed set
                    set_suppressed.push_back(best_scoring_hypothesis);

                    // Filter-out overlapping dudes
                    std::vector<Hypothesis> tmp_set;

                    for (int i = 0; i < active_set.size(); i++) {
                        if (overlapping_set_indices.count(i) == 0 && overlapping_set_indices.size() > 0) {
                            tmp_set.push_back(active_set.at(i));
                        }
                    }
                    active_set = tmp_set;
                }
                return set_suppressed;
            }

            std::vector<Hypothesis>
            NewHyposSuppression(const std::vector<Hypothesis> &hypos, const std::vector<Hypothesis> &new_hypos,
                                double iou_threshold) {
                std::vector<Hypothesis> set_suppressed;

                for (const auto &new_hypo:new_hypos) {
                    bool got_dupli = false;
                    for (const auto &hypo:hypos) {
                        bool is_dupli = DuplicateTest(new_hypo, hypo, iou_threshold);
                        if (is_dupli)
                            got_dupli = true;
                    }

                    if (!got_dupli) set_suppressed.push_back(new_hypo);
                }

                return set_suppressed;
            }

            void CAMOTTracker::ProcessFrame(DataQueue::ConstPtr detections, int current_frame) {
                /// Access camera from past frame to current
                bool lookup_success = false;
                auto camera = detections->GetCamera(current_frame, lookup_success);
                const double max_hole_size_fct = 0.2;

                // -------------------------------------------------------------------------------
                // +++ HYPOTHESIS HANDLING +++
                // -------------------------------------------------------------------------------

                /// Extend existing hypotheses
                std::vector<Hypothesis> extended_hypos;
                if (hypotheses_.size() > 0)
                    extended_hypos = ExtendHypotheses(detections, current_frame);

                if (this->verbose_) {
                    printf(" -> Extended %d hypotheses.\n", extended_hypos.size());
                }

                /// Start new hypotheses
                auto new_hypos = StartNewHypotheses(detections, current_frame);

                if (this->verbose_) {
                    printf(" -> Created %d hypotheses.\n", new_hypos.size());
                }

                /// Check new hypos for 'exact' duplicates
                auto new_hypos_supp = NewHyposSuppression(extended_hypos, new_hypos, /*0.7*/0.9);
                new_hypos = new_hypos_supp;

                /// {active_hypos_set} = {extended_set} U {new_set}
                this->hypotheses_.clear();
                this->hypotheses_.insert(hypotheses_.end(), extended_hypos.begin(), extended_hypos.end());
                this->hypotheses_.insert(hypotheses_.end(), new_hypos.begin(), new_hypos.end());

                // -------------------------------------------------------------------------------
                // +++ TERMINATION CONDITIONS CHECK +++
                // -------------------------------------------------------------------------------

                /// Terminate hypos that go out of the camera viewing frustum.
                this->CheckExitZones(camera, current_frame);

                /// Terminate hypos with 'large holes'
                for (auto &hypo:hypotheses_) {
                    if (!hypo.terminated().IsTerminated()) {
                        const int last_detection_frame = hypo.inliers().back().timestamp_;
                        int hole_size = std::abs(current_frame - last_detection_frame);

                        auto max_hole_size = parameter_map_.at(
                                "tracking_model_accepted_frames_without_inliers").as<int>();
                        int max_hole_size_this_hypo = std::min(max_hole_size, std::max(1, (int) std::round(
                                hypo.inliers().size() * max_hole_size_fct)));

                        if (hole_size > /*max_hole_size*/max_hole_size_this_hypo) {
                            hypo.set_terminated(GOT::tracking::TerminationInfo(true, current_frame));
                        }
                    }
                }

                // -------------------------------------------------------------------------------
                // +++ HYPOTHESIS TERMINATION +++
                // -------------------------------------------------------------------------------
                // Remove ALL terminated from the active hypothesis set.
                // Those hypotheses (tracklets), that are considered 'valid learning instances', push to the exported tracklets set,
                std::vector<Hypothesis> filtered_hypotheses;
                for (auto &hypo:hypotheses_) {
                    if (hypo.terminated().IsTerminated()) {
                        GOT::tracking::ClearCachedPredictions(hypo.cache());
                        exported_tracklets_.push_back(hypo);
                    } else
                        filtered_hypotheses.push_back(hypo);
                }

                hypotheses_ = filtered_hypotheses;
                this->ComputeUnariesGlobal(hypotheses_);

                // -------------------------------------------------------------------------------
                // +++ ONLINE NON-MAX-SUPP +++
                // -------------------------------------------------------------------------------
                if (this->parameter_map_.at("do_track_nms").as<bool>()) {
                    int non_max_supp_window = this->parameter_map_.at("tracking_suppression_front").as<int>();

                    /// Split hypos into 'active' and 'passive' set
                    // 'passive' set is the one with no inlier in the active temporal window.
                    std::vector<Hypothesis> active;
                    std::vector<Hypothesis> passive;

                    for (const auto &hypo:hypotheses_) {
                        int temporal_threshold = std::max(0, current_frame - non_max_supp_window);
                        if (hypo.creation_timestamp() < temporal_threshold)
                            passive.push_back(hypo);
                        else
                            active.push_back(hypo);
                    }

                    printf("Hypo suppression (passive: %d, active: %d) ... \r\n", (int) passive.size(), (int) active.size());

                    /// Call non-max-supp
                    const double nma_thresh = this->parameter_map_.at("tracking_non_max_supp_threshold").as<double>();
                    auto supressed_passive = HypothesisNonMaximaSuppression(passive, nma_thresh);
                    printf("Online suppression; passive: %d, suppressed: %d\r\n", (int) passive.size(),
                           (int) supressed_passive.size());

                    /// Merge sets
                    this->hypotheses_.clear();
                    this->hypotheses_.insert(hypotheses_.end(), supressed_passive.begin(), supressed_passive.end());
                    this->hypotheses_.insert(hypotheses_.end(), active.begin(), active.end());
                } else {
                    std::cout << "CAMOT: Warning: NOT doing track suppression!" << std::endl;
                }
            }

            // -------------------------------------------------------------------------------
            // +++ INITIALIZE HYPO ENTRIES +++
            // -------------------------------------------------------------------------------
            auto get_label_mapper = [](const std::string &label_mapper_str) -> std::map<int, std::string> {
                std::map<int, std::string> ret_map;
                if (label_mapper_str == "coco") {
                    ret_map = SUN::shared_types::category_maps::coco_map;
                } else if (label_mapper_str == "kitti") {
                    ret_map = SUN::shared_types::category_maps::kitti_map;
                } else {
                    printf("Error, invalid label_map str specified!\r\n");
                    assert (false);
                    return ret_map;
                }

                return ret_map;
            };

            auto ExtendBboxToGround = [](const Eigen::Vector4d &bbox_in, const Eigen::Vector3d &pos,
                                         const SUN::utils::Camera &camera) -> Eigen::Vector4d {
                Eigen::Vector4d front_point_projected;
                front_point_projected.head<3>() = pos;
                front_point_projected[3] = 1.0;
                front_point_projected.head<3>() = camera.ground_model()->ProjectPointToGround(
                        front_point_projected.head<3>());
                Eigen::Vector3i proj_footpoint_front = camera.CameraToImage(front_point_projected);
                Eigen::Vector4d bounding_box_2d = bbox_in;
                bounding_box_2d[3] = std::max(bounding_box_2d[3],
                                              std::abs(proj_footpoint_front[1] - bounding_box_2d[1]));
                return bounding_box_2d;
            };

            void CAMOTTracker::HypothesisInit(DataQueue::ConstPtr detections,
                                              bool is_forward_update, int inlier_index, int current_frame,
                                              Hypothesis &hypo) {

                const std::string &label_mapping_str = this->parameter_map_.at("label_mapping").as<std::string>();
                std::map<int, std::string> label_map = get_label_mapper(label_mapping_str);

                /// Get inlier and camera for the current frame
                Observation observation;
                SUN::utils::Camera camera;
                bool got_inlier = detections->GetInlierObservation(current_frame, inlier_index, observation);
                bool got_cam = detections->GetCamera(current_frame, camera);
                auto scene_cloud = detections->GetPointCloud(current_frame);

                if (!(got_inlier && got_cam)) {
                    printf("FATAL_ERROR:HypothesisInit: Can't reach resources!\n");
                    assert(false);
                    return;
                }

                /// Add inlier info and hypo pose+timestamp pair to the hypo.
                Eigen::Vector4d measurement_pos_in_world = camera.CameraToWorld(observation.footpoint());
                hypo.AddEntry(measurement_pos_in_world, current_frame);
                hypo.AddInlier(HypothesisInlier(current_frame, inlier_index, observation.score(), 1.0));

                /// Access relevant KF states
                const auto ptr_to_kf = std::static_pointer_cast<GOT::tracking::ConstantVelocitySize3DKalmanFilter>(
                        hypo.kalman_filter());
                const Eigen::Vector3d &posterior_pose_xyz_on_ground = ptr_to_kf->GetFullPoseGroundPlane();
                Eigen::Vector3d posterior_size_3d = hypo.kalman_filter_const()->GetSize3d();
                Eigen::Vector2d posterior_velocity = hypo.kalman_filter_const()->GetVelocityGroundPlane();

                /// Filtered pose -> camera-space
                Eigen::Vector4d plane_cam = std::static_pointer_cast<SUN::utils::PlanarGroundModel>(
                        camera.ground_model())->plane_params();
                const double x = posterior_pose_xyz_on_ground[0];
                const double y = posterior_pose_xyz_on_ground[1];
                const double z = posterior_pose_xyz_on_ground[2];
                Eigen::Vector4d hypo_position_camera_space = camera.WorldToCamera(Eigen::Vector4d(x, y, z, 1.0));
                double height_estim = posterior_size_3d[1];
                Eigen::Vector3d bbox_3d_center_in_camera_space =
                        hypo_position_camera_space.head<3>() + Eigen::Vector3d(0.0, 1.0, 0.0) * (-height_estim / 2.0);

                /// Filtered 3D bounding-box
                Eigen::VectorXd filtered_bbox_3d;
                filtered_bbox_3d.setZero(6);
                filtered_bbox_3d << bbox_3d_center_in_camera_space[0],
                        bbox_3d_center_in_camera_space[1],
                        bbox_3d_center_in_camera_space[2],
                        posterior_size_3d[0], posterior_size_3d[1], posterior_size_3d[2];

                /// Camera-space pose
                hypo.cache().Update(current_frame).pose_cam() = hypo_position_camera_space;
                hypo.cache().Update(current_frame).box3() = filtered_bbox_3d;

                Eigen::Vector4d bbox_to_add;

                // Segmentation-derived bbox
                Eigen::Vector4i mask_bbox;
                observation.compressed_mask().GetBoundingBox(mask_bbox[0], mask_bbox[1], mask_bbox[2], mask_bbox[3]);


                // Original bbox from the observation
                Eigen::Vector4d obs_bbox = observation.bounding_box_2d();

                /// Filtered bounding-box 2D
                hypo.cache().Update(current_frame).box2() = obs_bbox;
                hypo.cache().Update(current_frame).box_ext() = ExtendBboxToGround(obs_bbox,
                                                                                  observation.footpoint().head<3>(),
                                                                                  camera);

                /// Masks
                hypo.cache().Update(current_frame).mask() = observation.compressed_mask();
                hypo.cache().Update(current_frame).predicted_mask() = observation.compressed_mask();

                /// Color histogram
                hypo.set_color_histogram(observation.color_histogram());

                /// Segment
                pcl::PointCloud<pcl::PointXYZRGBA> seg_cloud;
                pcl::copyPointCloud(*scene_cloud, observation.pointcloud_indices(), seg_cloud);

                hypo.cache().Update(current_frame).predicted_segment() = seg_cloud;

                /// Segment id
                hypo.cache().Update(current_frame).segm_id() = observation.segm_id();

                /// Init category posterior
                hypo.category_probability_distribution() = observation.category_posterior();
            }

            // -------------------------------------------------------------------------------
            // +++ UPDATE HYPO ENTRIES +++
            // -------------------------------------------------------------------------------

            auto RemoveStupidPoints = [](const Eigen::Vector3d &median_pos, const Eigen::Matrix3d &cov_mat,
                                         const pcl::PointCloud<pcl::PointXYZRGBA> &pc_in) -> pcl::PointCloud<pcl::PointXYZRGBA> {
                pcl::PointCloud<pcl::PointXYZRGBA> cloud_out;

                // Idea:
                // Foreach point, compute mahalanobis distance to the median pose
                // If > T, reject the point

                double T = 10.0;
                Eigen::Matrix2d cov_mat_2 = Eigen::Matrix2d::Identity();
                cov_mat_2(0, 0) = cov_mat(0, 0);
                cov_mat_2(1, 0) = cov_mat(2, 0);
                cov_mat_2(0, 1) = cov_mat(0, 2);
                cov_mat_2(1, 1) = cov_mat(2, 2);
                Eigen::Matrix2d cov_mat_2_inv = cov_mat_2.inverse();
                for (const auto pt : pc_in.points) {
                    Eigen::Vector2d pos(pt.x, pt.z);
                    // Compute Mahalanobis dist to the median
                    Eigen::Vector2d diff = Eigen::Vector2d(median_pos[0], median_pos[2]) - pos;
                    double mh_dist = std::sqrt(diff.transpose() * cov_mat_2_inv * diff);
                    if (mh_dist < T) {
                        cloud_out.points.push_back(pt);
                    }
                }

                cloud_out.height = 1;
                cloud_out.width = cloud_out.size();
                return cloud_out;
            };

            /**  @function Erosion  */
            void Erosion(const cv::Mat &src, cv::Mat &dst, int erosion_size) {
                cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                            cv::Point(erosion_size, erosion_size));
                cv::erode(src, dst, element);
            }

            /** @function Dilation */
            void Dilation(const cv::Mat &src, cv::Mat &dst, int dilation_size) {
                cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                            cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                            cv::Point(dilation_size, dilation_size));
                cv::dilate(src, dst, element);
            }


            std::tuple<pcl::PointCloud<pcl::PointXYZRGBA>, SUN::shared_types::CompressedMask, bool>
            CAMOTTracker::GetPredictedSegment(int frame, bool forward, DataQueue::ConstPtr detections,
                                              const Hypothesis &hypo) {
                const double dt = this->parameter_map_.at("dt").as<double>();
                const bool use_filtered_velocity = this->parameter_map_.at(
                        "tracking_use_filtered_velocity_for_mask_warping").as<bool>();
                const bool use_flow = this->parameter_map_.at("use_flow").as<bool>();
                const bool segment_dist_filter = this->parameter_map_.at(
                        "tracking_distance_based_segment_filter").as<bool>();
                const bool fill_holes_in_predicted_masks = this->parameter_map_.at(
                        "tracking_fill_holes_predicted_masks").as<bool>();

                pcl::PointCloud<pcl::PointXYZRGBA> predicted_segment;

                // Get egomotion estimate
                bool fetch_ego_success;
                Eigen::Matrix4d ego;
                detections->GetEgoEstimate(frame, ego, fetch_ego_success);

                // Idea:
                // If we have inlier -> take inlier segment
                // If no -> take last segment

                const auto &last_inlier = hypo.inliers().back();

                /// Get velocity estimate
                // --------- Get filtered velocity estimate ----------
                Eigen::Vector3d velocity_estim;
                velocity_estim.setZero();

                // Access filtered velocity, project to camera-space
                SUN::utils::Camera camera;
                bool got_cam = detections->GetCamera(frame, camera);
                assert(got_cam);

                const Eigen::Vector2d &vel_gp = hypo.kalman_filter_const()->GetVelocityGroundPlane();
                Eigen::Vector3d vel_cam_space(vel_gp[0], 0.0, vel_gp[1]);
                vel_cam_space = camera.R().transpose() * vel_cam_space;

                if (!forward) vel_cam_space *= -1.0; // ! not a bug !

                velocity_estim[0] = vel_cam_space[0];
                velocity_estim[2] = vel_cam_space[2];
                // ---------------------------------------------------

                /// Fetch the segment that will be 'warped' into the current frame
                if (std::abs(frame - last_inlier.timestamp_) <= 1) { // We have inlier?

                    // We have inlier, use the 'fresh' inlier segment
                    const auto last_pointcloud = detections->GetPointCloud(last_inlier.timestamp_);
                    GOT::tracking::Observation last_inlier_obs;
                    detections->GetInlierObservation(last_inlier.timestamp_, last_inlier.index_, last_inlier_obs);
                    pcl::copyPointCloud(*last_pointcloud, last_inlier_obs.pointcloud_indices(), predicted_segment);

                    // Remove potential bleeding-far-away points
                    if (segment_dist_filter) {
                        predicted_segment = RemoveStupidPoints(last_inlier_obs.footpoint().head<3>(),
                                                               last_inlier_obs.covariance3d(), predicted_segment);
                    }


                    // WARNING! Filtered velocity estimate gets overriden here!
                    // Alternatively, use 'last-inlier-flow' for velocity prediction
                    if (!use_filtered_velocity && use_flow) {
                        // Over-ride filtered velocity with last-observation velocity
                        // printf ("WARNING: over-ridding filtered velocity with observed velocity!\r\n");
                        velocity_estim = last_inlier_obs.velocity();
                    }

                } else {
                    // We don't have inlier, just take the last 'predicted segment'
                    predicted_segment = hypo.cache().back().predicted_segment();
                }

                /// Take into the account the 'time arrow' (backward or forward association?)
                if (!forward) {

                    // ! note !
                    // this will 'revert' velocity dir in case you are using observed velocity
                    // otherwise, it will UNDO FLIP ABOVE [if (!forward) vel_cam_space *= -1.0; // ! not a bug !]
                    // because filtering is performed in the 'inverse dir' anyway
                    // this should be coded properly ...

                    velocity_estim *= -1; // Reverse velocity estimation
                    Eigen::Matrix4d ego_inv = ego.inverse(); // Reverse ego
                    ego = ego_inv;
                }

                /// Apply egomotion-compenstation
                pcl::transformPointCloud(predicted_segment, predicted_segment, ego);

                /// Apply velocity-compensation
                for (auto &p:predicted_segment.points) {
                    p.x += velocity_estim[0] * dt;
                    p.z += velocity_estim[2] * dt;
                }

                /// NEW: turn inds -> mask
                // Get mask
                // Dilate mask
                // Erode mask
                // Turn mask back to inds
                // Draw 'mask'
                cv::Mat mask_image(camera.height(), camera.width(), CV_8UC1);
                mask_image *= 0;
                for (const auto &pt:predicted_segment.points) {
                    Eigen::Vector4d p;
                    p.head<3>() = pt.getVector3fMap().cast<double>();
                    p[3] = 1.0;
                    Eigen::Vector3i proj_p = camera.CameraToImage(p);
                    int x = proj_p[0], y = proj_p[1];
                    //const float depth = pt.z;
                    if (x >= 0 && y >= 0 && x < camera.width() && y < camera.height()) {
                        // RGB
                        mask_image.at<uchar>(y, x) = 255;
                    }
                }

                if (fill_holes_in_predicted_masks) {
                    int ksize = this->parameter_map_.at("closing_op_kernel_size").as<int>();
                    Dilation(mask_image, mask_image, ksize);
                    Erosion(mask_image, mask_image, ksize);
                }

                std::vector<int> smoothed_mask_inds;
                for (int y = 0; y < mask_image.rows; y++) {
                    for (int x = 0; x < mask_image.cols; x++) {
                        if (mask_image.at<uchar>(y, x) == 255) {
                            int ind;
                            SUN::utils::RavelIndex(x, y, mask_image.cols, &ind);
                            smoothed_mask_inds.push_back(ind);
                        }
                    }
                }

                const int MIN_THRESH = 100;
                return std::make_tuple(predicted_segment,
                                       SUN::shared_types::CompressedMask(smoothed_mask_inds, camera.width(),
                                                                         camera.height()),
                                       smoothed_mask_inds.size() > MIN_THRESH);
            }

            bool CAMOTTracker::HypoAddPredictedSegment(int frame, bool forward, DataQueue::ConstPtr detections,
                                                       Hypothesis &hypo) {
                auto ret = this->GetPredictedSegment(frame, forward, detections,
                                                     hypo); // Returns 3D pcl (segment), mask, something
                const auto &pcl = std::get<0>(ret);
                hypo.cache().Update(frame).predicted_segment() = pcl;
                hypo.cache().Update(frame).predicted_mask() = std::get<1>(ret);

                return true;
            }

            auto avg_two_vecsf = [](const std::vector<float> &v1, const std::vector<float> &v2) -> std::vector<float> {

                if (v1.size() != v2.size()) {
                    throw std::runtime_error("avg_two_vecs error, v1.size() != v2.size()!");
                }

                std::vector<float> res(v1.size());
                for (int i = 0; i < v1.size(); i++) {
                    res.at(i) = (v1.at(i) + v2.at(i)) / 2.0f;
                }

                return res;
            };

            void CAMOTTracker::HypothesisUpdate(DataQueue::ConstPtr detections,
                                                bool is_forward_update,
                                                std::tuple<int, double, std::vector<double>> data_assoc_context,
                                                int current_frame,
                                                Hypothesis &hypo) {

                const std::string &label_mapping_str = this->parameter_map_.at("label_mapping").as<std::string>();
                std::map<int, std::string> label_map = get_label_mapper(label_mapping_str);

                /// Get inlier and camera for the current frame
                SUN::utils::Camera camera;
                bool got_cam = detections->GetCamera(current_frame, camera);
                //auto scene_cloud = detections->GetPointCloud(current_frame);

                if (!got_cam) {
                    printf("FATAL_ERROR:HypothesisUpdate:Can't reach resources!\n");
                    assert(false);
                    return;
                }

                /// Access relevant KF states
                // Note, that in case there was no inlier, KF state posterior is actually prediction ...
                const auto ptr_to_kf = std::static_pointer_cast<GOT::tracking::ConstantVelocitySize3DKalmanFilter>(
                        hypo.kalman_filter());
                const Eigen::Vector3d &posterior_pose_xyz_on_ground = ptr_to_kf->GetFullPoseGroundPlane();
                const Eigen::Vector3d &posterior_size_3d = hypo.kalman_filter_const()->GetSize3d();
                const Eigen::Vector2d posterior_velocity = hypo.kalman_filter_const()->GetVelocityGroundPlane();
                const double x = posterior_pose_xyz_on_ground[0];
                const double y = posterior_pose_xyz_on_ground[1];
                const double z = posterior_pose_xyz_on_ground[2];

                /// Compute camera-coords. of the object pos. (current view)
                Eigen::Vector4d hypo_position_camera_space = camera.WorldToCamera(Eigen::Vector4d(x, y, z, 1.0));

                /// Add current (filtered) pose (world-space)
                hypo.AddEntry(Eigen::Vector4d(x, y, z, 1.0), current_frame);

                /// Compute filtered bounding-box (3D)
                const double object_width = posterior_size_3d[0];
                const double object_height = posterior_size_3d[1];
                const double object_length = posterior_size_3d[2];
                Eigen::Vector3d bbox_3d_center_in_camera_space =
                        hypo_position_camera_space.head<3>() + Eigen::Vector3d(0.0, 1.0, 0.0) * (-object_height / 2.0);

                /// Append filtered bounding-box (3D)
                Eigen::VectorXd object_bbox_3d;
                object_bbox_3d.setZero(6);
                object_bbox_3d << bbox_3d_center_in_camera_space[0],
                        bbox_3d_center_in_camera_space[1],
                        bbox_3d_center_in_camera_space[2],
                        object_width, object_height, object_length;

                hypo.cache().Update(current_frame).box3() = object_bbox_3d;

                /// Append filtered pose (camera-space)
                hypo.cache().Update(current_frame).pose_cam() = hypo_position_camera_space;
                auto mask_prior = hypo.cache().back().predicted_mask();

                /// Updates that require an associated inlier
                int inlier_index = std::get<0>(data_assoc_context);
                double data_assoc_score = std::get<1>(data_assoc_context);
                if (inlier_index >= 0) {
                    Observation observation;
                    bool got_inlier = detections->GetInlierObservation(current_frame, inlier_index, observation);

                    if (!got_inlier) {
                        printf("FATAL_ERROR:HypothesisUpdate:Can't reach resources!\n");
                        assert(false);
                        return;
                    }

                    auto inl = HypothesisInlier(current_frame, inlier_index, observation.score(), data_assoc_score);
                    inl.assoc_data_ = std::get<2>(data_assoc_context);
                    hypo.AddInlier(inl);
                    hypo.set_color_histogram(0.4 * hypo.color_histogram() + 0.6 * observation.color_histogram());

                    auto mask_posterior = observation.compressed_mask();
                    hypo.cache().Update(current_frame).mask() = mask_posterior;
                    Eigen::Vector4d bbox_to_add;

                    // Segmentation-derived bbox
                    Eigen::Vector4i mask_bbox;
                    observation.compressed_mask().GetBoundingBox(mask_bbox[0], mask_bbox[1], mask_bbox[2],
                                                                 mask_bbox[3]);

                    // Original bbox from the observation
                    Eigen::Vector4d obs_bbox = observation.bounding_box_2d();

                    hypo.cache().Update(current_frame).box2() = obs_bbox;
                    hypo.cache().Update(current_frame).box_ext() = ExtendBboxToGround(obs_bbox,
                                                                                      observation.footpoint().head<3>(),
                                                                                      camera);
                    /// Segment id
                    hypo.cache().Update(current_frame).segm_id() = observation.segm_id();

                    /// Simple avg
                    hypo.category_probability_distribution() = avg_two_vecsf(observation.category_posterior(),
                                                                             hypo.category_probability_distribution());

                    float sum = std::accumulate(hypo.category_probability_distribution().begin(),
                                                hypo.category_probability_distribution().end(),
                                                static_cast<float>(0.0));
                    assert(sum < 1.1); // Simple validity check

                } else {
                    hypo.cache().Update(current_frame).mask() = mask_prior;

                    /// ! new ! bbox from mask
                    Eigen::Vector4i mask_bbox;
                    mask_prior.GetBoundingBox(mask_bbox[0], mask_bbox[1], mask_bbox[2], mask_bbox[3]);
                    Eigen::Vector4d bbox_to_add = mask_bbox.cast<double>();


                    hypo.cache().Update(current_frame).box2() = bbox_to_add;
                    hypo.cache().Update(current_frame).box_ext() = ExtendBboxToGround(bbox_to_add,
                                                                                      hypo.cache().back().pose_cam().head<3>(),
                                                                                      camera);
                    /// Update 'category' with uniform -> decay
                    auto post_size = hypo.category_probability_distribution().size();
                    auto uniform_likelihood = std::vector<float>(post_size, static_cast<float>(1.0 / post_size));

                    // Simple avg
                    hypo.category_probability_distribution() = avg_two_vecsf(uniform_likelihood,
                                                                             hypo.category_probability_distribution());
                }
            }

            int CAMOTTracker::AdvanceHypo(DataQueue::ConstPtr detections, int reference_frame, bool is_forward,
                                          Hypothesis &ref_hypo, bool allow_association) {

                bool detections_lookup_success = false;
                bool camera_lookup_success = false;

                // Access data, needed to make an association
                const auto &ref_camera = detections->GetCamera(reference_frame, camera_lookup_success);
                //auto past_state_point_cloud = detections->GetPointCloud(reference_frame);
                const Observation::Vector &ref_state_observations = detections->GetObservations(reference_frame,
                                                                                                detections_lookup_success);

                int max_element_index = -1;

                std::vector<int> used_inds;
                if (detections_lookup_success && camera_lookup_success) {

                    /// Hypo prediction/transition
                    this->dynamics_model_handler_->ApplyTransition(ref_camera, Eigen::VectorXd::Zero(8), ref_hypo);
                    bool predict_in_image = HypoAddPredictedSegment(reference_frame, is_forward, detections, ref_hypo);

                    // Is the predicted object still in the image?
                    if (!predict_in_image /*&& is_forward*/) {
                        ref_hypo.set_terminated(GOT::tracking::TerminationInfo(true, reference_frame));
                        return -2;
                    }

                    /// Data association
                    std::vector<double> assoc_score_table;
                    std::vector<std::vector<double> > assoc_extra;
                    if (allow_association) {
                        auto res = this->data_association_scores_fnc_(detections, ref_hypo, reference_frame);
                        assoc_score_table = std::get<0>(res);
                        assoc_extra = std::get<1>(res);
                    }

                    /// Update hypo state using the new evidence
                    bool got_valid_inlier = false;
                    if (assoc_score_table.size() > 0) {
                        auto max_element_it = std::max_element(assoc_score_table.begin(), assoc_score_table.end());
                        double max_element_assoc_score = *max_element_it;

                        // See if we have a matching detection (this is sort of hacked data association)
                        const double min_assoc_score = parameter_map_.at(
                                "data_association_min_association_score").as<double>();
                        if (max_element_assoc_score > min_assoc_score) {

                            got_valid_inlier = true;
                            max_element_index = static_cast<int>(std::distance(assoc_score_table.begin(),
                                                                               max_element_it));
                            const auto &observation = ref_state_observations.at(max_element_index);

                            /// Perform update
                            this->dynamics_model_handler_->ApplyCorrection(ref_camera, observation, is_forward,
                                                                           ref_hypo);

                            /// Update state of the hypo using the inlier
                            this->HypothesisUpdate(detections, is_forward,
                                                   std::make_tuple(max_element_index, max_element_assoc_score,
                                                                   assoc_extra.at(max_element_index)),
                                                   reference_frame,
                                                   ref_hypo
                            );
                        }
                    }

                    /// Extrapolate hypo state (in case there is no new evidence of course)
                    if (!got_valid_inlier) {
                        this->HypothesisUpdate(detections, is_forward,
                                               std::make_tuple(-1, 0.0, std::vector<double>()),
                                               reference_frame,
                                               ref_hypo
                        );
                    }
                }

                return max_element_index;
            }


            // -------------------------------------------------------------------------------
            // +++ CREATE NEW HYPOTHESES +++
            // -------------------------------------------------------------------------------
            std::vector<Hypothesis>
            CAMOTTracker::StartNewHypotheses(DataQueue::ConstPtr detections, int current_frame) {
                /// Parameters
                auto max_hole_size_init = parameter_map_.at("tracking_model_accepted_frames_without_inliers").as<int>();
                const double max_hole_size_fct = 0.2;


                auto temporal_window_size = parameter_map_.at("tracking_temporal_window_size").as<int>();
                auto min_obs_init_hypo = parameter_map_.at("tracklets_min_inliers_to_init_tracklet").as<int>();
                //auto range_min = parameter_map_.at("tracking_exit_zones_rear_distance").as<double>();
                auto range_max = parameter_map_.at("tracking_exit_zones_far_distance").as<double>();
                bool selective_start_optimization = parameter_map_.at(
                        "tracking_selective_hypothesis_initialization").as<bool>();

                std::vector<Hypothesis> new_hypotheses;

                /// Get resources
                bool observations_lookup_success = false;
                const auto &observations_current_frame = detections->GetObservations(current_frame,
                                                                                     observations_lookup_success);
                bool camera_success = false;
                const auto &camera = detections->GetCamera(current_frame, camera_success);

                if (!(camera_success && observations_lookup_success))
                    return new_hypotheses; // Return empty hypothesis set

                const int num_observations = observations_current_frame.size();

                // -------------------------------------------------------------------------------
                // +++ LOOP OBSERVATIONS +++
                // -------------------------------------------------------------------------------
                for (int i = 0; i < num_observations; i++) {
                    const auto &observation_current_frame = observations_current_frame.at(i);

                    /// Only start hypos that are in the close camera range.
                    if (observation_current_frame.footpoint()[2] > (range_max + 10))
                        continue;
                    if (observation_current_frame.footpoint()[2] < 1.0) //range_min)
                        continue;

                    int accept_new_hypothesis = true;

                    // Only start new hypo if this detection was not used for extending others.
                    bool obs_not_used_for_ext = this->detection_indices_used_for_extensions_.count(i) <= 0;
                    if (((obs_not_used_for_ext == true) && (selective_start_optimization == true)) ||
                        (selective_start_optimization == false)) {

                        /// Init new hypothesis from this observation.
                        Hypothesis new_hypo;
                        dynamics_model_handler_->InitializeState(camera, observation_current_frame, /*true*/ false,
                                                                 new_hypo);
                        this->HypothesisInit(detections, false, i, current_frame, new_hypo);

                        // -------------------------------------------------------------------------------
                        // +++ FIND EVIDENCE FOR CURRENT OBS. IN PAST FRAMES +++
                        // -------------------------------------------------------------------------------
                        const int num_past_frames_for_hypo_construction = std::min(current_frame, temporal_window_size);
                        for (int j = 1; j <= (num_past_frames_for_hypo_construction - 1); j++) {
                            const int past_frame_currently_being_processed = current_frame - j;

                            /// EarlyReject: Require certain number of supporting detections in first n-frames.
                            if (j <= min_obs_init_hypo) {
                                if ((new_hypo.inliers().size() != j)) {
                                    accept_new_hypothesis = false;
                                    break;
                                }
                            }

                            /// Stop extending hypo, if we haven't been able to make association for too long.
                            if (new_hypo.cache().timestamps().size() > 0) {
                                // We have seen detection already. See how big the "hole" is.
                                int last_inlier_frame = new_hypo.inliers().back().timestamp_;
                                int hole_size = std::abs(last_inlier_frame - past_frame_currently_being_processed);

                                int max_hole_size_this_hypo = std::min(max_hole_size_init, std::max(1, (int) std::round(
                                        new_hypo.inliers().size() * max_hole_size_fct)));

                                if (hole_size > /*max_hole_size_init*/max_hole_size_this_hypo) {
                                    break;
                                }
                            }

                            int assoc_idx = this->AdvanceHypo(detections, past_frame_currently_being_processed, false,
                                                              new_hypo, accept_new_hypothesis);

                            // NEW: IF CANT MAKE ASSOC -> BREAK
                            if (assoc_idx == -2)
                                break;

                        }

                        // -------------------------------------------------------------------------------
                        // +++ END-OF FIND EVIDENCE FOR CURRENT OBS. IN PAST FRAMES +++
                        // -------------------------------------------------------------------------------
                        const int min_inlier_to_init = parameter_map_.at(
                                "tracklets_min_inliers_to_init_tracklet").as<int>();
                        if (accept_new_hypothesis && new_hypo.inliers().size() >= min_inlier_to_init) {
                            new_hypo.set_id(this->last_hypo_id_++);
                            new_hypo.set_creation_timestamp(current_frame);
                            new_hypotheses.push_back(new_hypo);
                        }
                    }
                }

                // -------------------------------------------------------------------------------
                // +++ END-OF LOOP OBSERVATIONS +++
                // -------------------------------------------------------------------------------

                // -------------------------------------------------------------------------------
                // +++ RE-INITIALIZE HYPOTHESES +++
                // -------------------------------------------------------------------------------
                //! Post-processing:
                // 1. Flip-around the inliers.
                // 2. Drop kalman filter; initialize a new one.
                // 2. Start from first to last, use already-determined inliers.

                std::vector<Hypothesis> rev_hypos;
                for (Hypothesis &hypo : new_hypotheses) {

                    std::vector<HypothesisInlier> inliers = hypo.inliers();
                    std::vector<HypothesisInlier> inliers_reverse = hypo.inliers();

                    assert (inliers.size() >= min_obs_init_hypo);

                    if (inliers.size() >= min_obs_init_hypo) {

                        // Inliers
                        std::reverse(inliers.begin(), inliers.end());

                        /// Reset all entries
                        hypo.set_inliers(std::vector<HypothesisInlier>());
                        hypo.cache().Reset();

                        const int first_frame = inliers.front().timestamp_;
                        const int first_frame_inlier_index = inliers.front().index_;

                        /// Obtain observations, camera
                        bool first_frame_camera_lookup_success = false, first_frame_detections_lookup_success = false;
                        const auto &camera_first_frame = detections->GetCamera(first_frame,
                                                                               first_frame_camera_lookup_success);
                        const Observation::Vector &observations_2d_first_frame = detections->GetObservations(
                                first_frame, first_frame_detections_lookup_success);

                        if (first_frame_camera_lookup_success && first_frame_detections_lookup_success) {
                            const auto first_frame_obs = observations_2d_first_frame.at(first_frame_inlier_index);

                            // Re-start the hypothesis
                            hypo.kalman_filter().reset();
                            dynamics_model_handler_->InitializeState(camera_first_frame, first_frame_obs, true, hypo);
                            HypothesisInit(detections, true, first_frame_inlier_index, first_frame, hypo);

                            /// Walk from past frame back to the current frame
                            const int next_frame =
                                    inliers.front().timestamp_ + 1; // This is correct! Don't fix it (again)!
                            const int last_inlier_frame = inliers.back().timestamp_;
                            int current_inlier_index = 1;
                            for (int candidate_frame = next_frame;
                                 candidate_frame <= last_inlier_frame; candidate_frame++) {

                                bool current_frame_camera_lookup_success = false, current_frame_detections_lookup_success = false;

                                /// Access next inlier in the inlier list
                                const int inlier_frame = inliers.at(current_inlier_index).timestamp_;
                                const int inlier_index = inliers.at(current_inlier_index).index_;

                                const double inlier_data_assoc_score = inliers_reverse.at(
                                        current_inlier_index).association_score_;
                                const auto inlier_data_assoc_score_extra = inliers_reverse.at(
                                        current_inlier_index).assoc_data_;

                                /// Access all cameras&observations
                                const auto &camera_current_frame = detections->GetCamera(candidate_frame,
                                                                                         current_frame_camera_lookup_success);
                                assert(current_frame_camera_lookup_success);
                                const Observation::Vector &observations_current_frame = detections->GetObservations(
                                        candidate_frame, current_frame_detections_lookup_success);

                                /// Perform FORWARD egomotion compensation
                                Eigen::VectorXd u = Eigen::VectorXd::Zero(8); // NOPE

                                /// Perform FORWARD transition
                                this->dynamics_model_handler_->ApplyTransition(camera_current_frame, u, hypo);
                                HypoAddPredictedSegment(candidate_frame, true, detections, hypo);

                                if (candidate_frame ==
                                    inlier_frame) { // Check, if the next inlier in the list comes from current frame
                                    /// YES: got inlier in this frame

                                    /// Access inlier and perform FORWARD update
                                    const auto &obs = observations_current_frame.at(inlier_index);
                                    this->dynamics_model_handler_->ApplyCorrection(camera_current_frame, obs, true,
                                                                                   hypo);

                                    /// Update hypo entries
                                    HypothesisUpdate(detections, true,
                                                     std::make_tuple(inlier_index, inlier_data_assoc_score,
                                                                     inlier_data_assoc_score_extra), candidate_frame,
                                                     hypo);

                                    /// Inc. inlier 'pointer'
                                    current_inlier_index++;
                                } else {
                                    /// GOT _NO_ INLIER IN CURRENT FRAME, call HypothesisUpdate to perform extrapolation
                                    HypothesisUpdate(detections, true, std::make_tuple(-1, 0.0, std::vector<double>()),
                                                     candidate_frame, hypo);
                                }
                            }
                        } else {
                            std::cout << "StartNewHpotheses::ERROR: can't reach resources!" << std::endl;
                        }

                        rev_hypos.push_back(hypo);
                    }
                }
                // -------------------------------------------------------------------------------
                // +++ END-OF RE-INITIALIZE HYPOTHESES +++
                // -------------------------------------------------------------------------------
                return rev_hypos;
            }


            // -------------------------------------------------------------------------------
            // +++ UPDATE EXISTING HYPOTHESES +++
            // -------------------------------------------------------------------------------
            std::vector<Hypothesis> CAMOTTracker::ExtendHypotheses(DataQueue::ConstPtr detections, int current_frame) {
                detection_indices_used_for_extensions_.clear();
                const double max_hole_size_fct = 0.2;

                /// Get camera & observations from the resource manager
                bool observations_lookup_success = false;
                const Observation::Vector &observations = detections->GetObservations(current_frame,
                                                                                      observations_lookup_success);
                bool camera_lookup_success = false;
                const auto &camera = detections->GetCamera(current_frame, camera_lookup_success);
                assert(camera_lookup_success);
                int max_hole_size = parameter_map_.at("tracking_model_accepted_frames_without_inliers").as<int>();
                std::vector<Hypothesis> new_hypothesis_set;
                if (observations_lookup_success && camera_lookup_success) {

                    // -------------------------------------------------------------------------------
                    // +++ LOOP OVER HYPOTHESES +++
                    // -------------------------------------------------------------------------------
                    for (int j = 0; j < this->hypotheses_.size(); j++) {
                        auto new_hypo = hypotheses_.at(j);
                        bool allow_association = true;

                        // Stop extending hypo if last inlier is 'too old'
                        if (new_hypo.cache().timestamps().size() > 0) {
                            // We have seen detection already. See how big the "hole" is.
                            int last_inlier_frame = new_hypo.inliers().back().timestamp_;
                            int hole_size = std::abs(current_frame - last_inlier_frame);
                            int max_hole_size_this_hypo = std::min(max_hole_size, std::max(1, (int) std::round(
                                    new_hypo.inliers().size() * max_hole_size_fct)));
                            if (hole_size > /*max_hole_size*/max_hole_size_this_hypo) {
                                allow_association = false;
                            }
                        }

                        // Do not perform data assoc. if hypo was terminated
                        if (new_hypo.terminated().IsTerminated())
                            allow_association = false;

                        int ext_inlier_idx = this->AdvanceHypo(detections, current_frame, true, new_hypo,
                                                               allow_association);

                        if (ext_inlier_idx == -2)
                            new_hypo.set_terminated(GOT::tracking::TerminationInfo(true, current_frame));

                        if (ext_inlier_idx >= 0)
                            detection_indices_used_for_extensions_.insert(ext_inlier_idx);

                        new_hypothesis_set.push_back(new_hypo);
                    }
                }

                return new_hypothesis_set;
            }


            std::vector<GOT::tracking::Hypothesis> CAMOTTracker::GetConfidentHypotheses() const {
                std::vector<GOT::tracking::Hypothesis> confident_hypos;
                for (const auto &hypo:hypotheses_) {
                    confident_hypos.push_back(hypo);
                }
                return confident_hypos;
            }

            const std::vector<GOT::tracking::Hypothesis> &CAMOTTracker::exported_tracklets() const {
                return exported_tracklets_;
            }

            void CAMOTTracker::AppendActiveTrackletsToExported() {
                int num_appended = 0;
                for (const auto &hypo:this->hypotheses_) {
                    exported_tracklets_.push_back(hypo);
                    num_appended++;
                }

                printf("Num. appended active tracklets: %d\r\n", num_appended);
            }

            void CAMOTTracker::ComputeUnariesGlobal(std::vector<Hypothesis> &hypos) {

                // Pick scoring fnc
                const auto unary_fnc_str = parameter_map_.at("unary_fnc").as<std::string>();
                auto unary_fnc = GOT::tracking::CAMOT_tracker::GetUnaryFnc(unary_fnc_str);

                // Apply to each hypo
                for (auto &hypo:hypos) {
                    double score = unary_fnc(hypo, hypo.cache().timestamps().back(), this->parameter_map_);
                    hypo.set_score(score);
                }
            }
        }

    }
}
