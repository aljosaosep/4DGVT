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

#include "observation_processing_utils.h"

#include <tracking/hypothesis.h>
#include <src/sun_utils/utils_bounding_box.h>
#include <src/sun_utils/utils_common.h>
#include "sun_utils/utils_kitti.h"
#include "sun_utils/utils_observations.h"
#include "sun_utils/ground_model.h"

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {
            namespace obs_proc {

                Observation::Vector
                ComputeObservationVelocity(const Observation::Vector &observations,
                                           const cv::Mat &velocity_map, double dt) {
                    Observation::Vector obs_new;
                    for (const auto &obs:observations) {
                        // Compute velocity
                        Eigen::Vector3d obs_velocity = SUN::utils::observations::ComputeVelocity(velocity_map,
                                                                                                 obs.pointcloud_indices(),
                                                                                                 dt, 5);
                        auto obs_copy = obs;
                        obs_copy.set_velocity(obs_velocity);
                        obs_new.push_back(obs_copy);
                    }

                    return obs_new;
                }

                Observation::Vector
                ProcessObservations(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                    const GOT::segmentation::ObjectProposal::Vector &proposals,
                                    const SUN::utils::Camera &camera) {
                    Observation::Vector observations;

                    for (const auto &proposal : proposals) {

                        GOT::tracking::Observation obs;

                        const Eigen::Vector4d &prop_bbox = proposal.bounding_box_2d();

                        // Bounding-boxes
                        obs.set_bounding_box_2d(proposal.bounding_box_2d());
                        obs.set_bounding_box_3d(proposal.bounding_box_3d());

                        // Make sure 3D-pose is proj. to the ground
                        auto pos_proj_to_ground = proposal.pos3d();
                        pos_proj_to_ground.head<3>() = camera.ground_model()->ProjectPointToGround(
                                proposal.pos3d().head<3>());
                        obs.set_footpoint(pos_proj_to_ground);
                        obs.set_covariance3d(proposal.pose_covariance_matrix());

                        // Image-space indices
                        obs.set_pointcloud_indices(proposal.pointcloud_indices(), camera.width(), camera.height());

                        // '3D-proposal' info
                        obs.set_proposal_3d_score(proposal.score());
                        obs.set_proposal_3d_avalible(true);
                        obs.set_segm_id(proposal.segm_id());

                        // Proposal category can be decoded fom the 'second posterior'
                        const auto &second_post = proposal.second_posterior();
                        auto result = std::max_element(second_post.begin(), second_post.end());
                        int argmax_el = std::distance(second_post.begin(), result);

                        obs.set_detection_category(argmax_el);
                        obs.set_category_posterior(second_post);
                        obs.set_score(proposal.score());

                        observations.push_back(obs);
                    }

                    return observations;
                }
            }
        }
    }
}