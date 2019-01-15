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

#include "data_association.h"

// eigen
#include <Eigen/Core>

// tracking
#include <tracking/visualization.h>

// utils
#include "ground_model.h"
#include "utils_bounding_box.h"
#include "utils_common.h"

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {
            namespace data_association {

                /** ========================================================================
                 *      Data Association (for tracker!)
                 * =======================================================================*/
                auto project_covariance_2D = [](const Eigen::Matrix3d &pose_covariance_3d) -> Eigen::Matrix2d {
                    Eigen::Matrix2d pose_covariance_2d = Eigen::Matrix2d::Identity();
                    pose_covariance_2d(0, 0) = pose_covariance_3d(0, 0);
                    pose_covariance_2d(0, 1) = pose_covariance_3d(0, 2);
                    pose_covariance_2d(1, 0) = pose_covariance_3d(2, 0);
                    pose_covariance_2d(1, 1) = pose_covariance_3d(2, 2);
                    return pose_covariance_2d;
                };

                auto compute_motion_model_info = [](const object_tracking::Observation &observation,
                                                    const object_tracking::Hypothesis &hypo,
                                                    const SUN::utils::Camera &camera) -> std::tuple<double, double> {

                    // Access the filtered state of the hypothesis (world space)
                    const Eigen::Vector2d &hypo_pos_on_ground = hypo.kalman_filter_const()->GetPoseGroundPlane();

                    // Access obs. data, transform: camera space -> world space -> ground plane
                    Eigen::Vector4d obs_pos_world = camera.CameraToWorld(observation.footpoint()); // Cam -> World space
                    Eigen::Vector2d obs_pos_on_ground = Eigen::Vector2d(obs_pos_world[0],
                                                                        obs_pos_world[2]); // Take x, z coords
                    Eigen::Matrix3d obs_cov_3d_world =
                            camera.R() * observation.covariance3d() * camera.R().transpose(); // Cam -> World space
                    Eigen::Matrix2d obs_cov_2d_on_ground = project_covariance_2D(obs_cov_3d_world);

                    // Add a Gaussian prior (numerical reasons ...)
                    Eigen::Matrix2d gaussian_prior;
                    gaussian_prior << 0.2, 0.0, 0.0, 0.2;
                    obs_cov_2d_on_ground += gaussian_prior;

                    bool invertible;
                    double determinant = 0.0;
                    const Eigen::Vector2d pose_diff = obs_pos_on_ground - hypo_pos_on_ground;
                    Eigen::Matrix2d cov_2d_inverse;
                    obs_cov_2d_on_ground.computeInverseAndDetWithCheck(cov_2d_inverse, determinant, invertible);

                    assert(invertible);
                    if (!invertible) {
                        std::cout
                                << "compute_motion_model_info::ERROR: Cov. matrix is not invertible! Can't do association!"
                                << std::endl;
                        std::cout << "The matrix you were trying to invert:" << std::endl;
                        std::cout << obs_cov_2d_on_ground << std::endl;
                        return std::make_tuple<double, double>(std::numeric_limits<double>::quiet_NaN(),
                                                               std::numeric_limits<double>::quiet_NaN());
                    }

                    const double mahalanobis_dist_squared = pose_diff.transpose() * cov_2d_inverse * pose_diff;
                    double denom = std::sqrt(39.47841 * std::abs(determinant));
                    assert(denom > 0.0);
                    double motion_model_term = (1.0 / denom) * std::exp(-0.5 * mahalanobis_dist_squared);
                    return std::make_tuple(mahalanobis_dist_squared, motion_model_term);
                };

                auto compute_rectangle_IOU = [](int frame,
                                                const object_tracking::Observation &observation,
                                                const object_tracking::Hypothesis &hypo,
                                                const SUN::utils::Camera &camera) -> double {

                    // Get 2D bounding box prediction
                    const auto &predicted_mask = hypo.cache().at_frame(frame).predicted_mask();
                    Eigen::Vector4i hypo_bbox_2d;
                    predicted_mask.GetBoundingBox(hypo_bbox_2d[0], hypo_bbox_2d[1], hypo_bbox_2d[2], hypo_bbox_2d[3]);

                    // Access obs. data
                    const Eigen::Vector4d &obs_bbox_2d = observation.bounding_box_2d();

                    // Compute&return IOU_2D
                    return SUN::utils::bbox::IntersectionOverUnion2d(hypo_bbox_2d.cast<double>(), obs_bbox_2d);
                };


                auto gaiting_fnc_size_2D = [](const object_tracking::Observation &observation,
                                              const object_tracking::Hypothesis &hypo, int frame_of_assoc) -> double {
                    Eigen::Vector4d hypo_bbox_2d = hypo.cache().predecessor_frame(
                            frame_of_assoc).box2(); // Hypo box, from prev. frame
                    const Eigen::Vector4d &obs_bbox_2d = observation.bounding_box_2d(); // Observation

                    // 2D-size gaiting
                    auto h = hypo_bbox_2d[3];
                    auto h_det = obs_bbox_2d[3];
                    auto w = hypo_bbox_2d[2];
                    auto w_det = obs_bbox_2d[2];
                    double score = 1.0 - (std::fabs(h - h_det) / (2 * (h + h_det))) -
                                   (std::fabs(w - w_det) / (2 * (w + w_det)));

                    return score;
                };


                std::tuple<std::vector<double>, std::vector<std::vector<double> > > data_association_motion_mask(
                        GOT::tracking::DataQueue::ConstPtr detections,
                        const GOT::tracking::Hypothesis &hypo, int frame_of_association,
                        const po::variables_map &parameters) {

                    SUN::utils::Camera camera;
                    bool got_camera = detections->GetCamera(frame_of_association, camera);
                    bool got_observations;
                    const auto &observations = detections->GetObservations(frame_of_association, got_observations);

                    assert(got_camera);
                    assert(got_observations);

                    if (observations.size() == 0) { // Nothing to associate
                        return std::make_tuple(std::vector<double>(), std::vector<std::vector<double>>());
                    }

                    // Access the filtered state of the hypothesis
                    const Eigen::Vector2d &hypo_pos_on_ground = hypo.kalman_filter_const()->GetPoseGroundPlane();

                    // Init the association table
                    std::vector<double> data_assoc_scores_table(observations.size(), 0.0);
                    std::vector<std::vector<double>> data_assoc_extra(observations.size(), std::vector<double>());

                    auto gaiting_motion_model_thresh = parameters.at("gaiting_motion_model_threshold").as<double>();
                    auto gaiting_iou_thresh = parameters.at("gaiting_IOU_threshold").as<double>();
                    auto gaiting_size_2d_thresh = parameters.at("gaiting_size_2D").as<double>();
                    auto gaiting_iou_rect_thresh = parameters.at("gaiting_IOU_rect_threshold").as<double>();

                    for (int i = 0; i < observations.size(); i++) {
                        const auto &observation = observations.at(i);

                        auto motion_model_result = compute_motion_model_info(observation, hypo,
                                                                             camera); // Returns a tuple.
                        auto gaiting_size_2D_score = gaiting_fnc_size_2D(observation, hypo, frame_of_association);
                        auto projection_model_term = compute_rectangle_IOU(frame_of_association, observation, hypo,
                                                                           camera); // Returns double (IOU, in range [0, 1]).

                        // First item in the tuple is squared Mahalanobis dist tuple(mh_dist_squared, motion_model_weight)
                        if ((std::sqrt(std::get<0>(motion_model_result)) < gaiting_motion_model_thresh) &&
                            (projection_model_term > gaiting_iou_rect_thresh) &&
                            (gaiting_size_2D_score > gaiting_size_2d_thresh)) {
                            const auto &predicted_mask = hypo.cache().at_frame(frame_of_association).predicted_mask();
                            double mask_IoU = predicted_mask.IoU(observation.compressed_mask());
                            double motion_model_term = std::get<1>(
                                    motion_model_result); // Second item in tuple is the weight

                            if (mask_IoU > gaiting_iou_thresh) {
                                // Data assoc. prob.
                                data_assoc_scores_table.at(i) = mask_IoU * motion_model_term; // Joint prob.

                                // Export extra info
                                data_assoc_extra.at(i).push_back(mask_IoU);
                                data_assoc_extra.at(i).push_back(motion_model_term);
                                data_assoc_extra.at(i).push_back(
                                        std::get<0>(motion_model_result)); // Mahalanobis squared
                            }
                        }
                    }
                    return std::make_tuple(data_assoc_scores_table, data_assoc_extra);
                }

            }
        }
    }
}