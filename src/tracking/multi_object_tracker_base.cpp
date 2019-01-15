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

#include <tracking/multi_object_tracker_base.h>

// utils
#include "utils_bounding_box.h"
#include "ground_model.h"
#include "camera.h"

namespace GOT {
    namespace tracking {

        MultiObjectTracker3DBase::MultiObjectTracker3DBase(const po::variables_map &params) {
            this->parameter_map_ = params;
            last_hypo_id_ = 0;
        }

        void MultiObjectTracker3DBase::CheckExitZones(const SUN::utils::Camera &camera, int current_frame) {

            assert(parameter_map_.count("tracking_exit_zones_lateral_distance"));
            assert(parameter_map_.count("tracking_exit_zones_rear_distance"));
            assert(parameter_map_.count("tracking_exit_zones_far_distance"));

            double dist_lat = parameter_map_.at("tracking_exit_zones_lateral_distance").as<double>();
            double dist_rear = parameter_map_.at("tracking_exit_zones_rear_distance").as<double>();
            double dist_far = parameter_map_.at("tracking_exit_zones_far_distance").as<double>();

            const double multi=0.8;
            SUN::utils::Ray ray_top_left, ray_top_right, ray_bottom_left, ray_bottom_right;

            // Cast rays from camera origin through bottom-left, bottom-right, nearly-bottom-left, nearly-bottom-right pixels.
            ray_top_left = camera.GetRayCameraSpace(1, camera.height());
            ray_top_right = camera.GetRayCameraSpace(camera.width()-1, camera.height());
            ray_bottom_left = camera.GetRayCameraSpace(1, static_cast<int>(camera.height()*multi));
            ray_bottom_right = camera.GetRayCameraSpace(camera.width()-1, static_cast<int>(camera.height()*multi));

            // Intersect those rays with a ground plane. This will give us points, that define left and right 'border' of the viewing frustum
            Eigen::Vector3d proj_top_left = camera.ground_model()->IntersectRayToGround(ray_top_left.origin, ray_top_left.direction);
            Eigen::Vector3d proj_top_right = camera.ground_model()->IntersectRayToGround(ray_top_right.origin, ray_top_right.direction);
            Eigen::Vector3d proj_bottom_left = camera.ground_model()->IntersectRayToGround(ray_bottom_left.origin, ray_bottom_left.direction);
            Eigen::Vector3d proj_bottom_right = camera.ground_model()->IntersectRayToGround(ray_bottom_right.origin, ray_bottom_right.direction);

            // Compute vectors, defined by left and right points.
            Eigen::Vector3d left_vec = proj_bottom_left - proj_top_left;
            Eigen::Vector3d right_vec = proj_bottom_right - proj_top_right;

            /*
                  Camera frustum: left plane, right plane, bottom plane
                  Planes are defined by plane normals and 'd' ([a b c d] parametrization)

                                       \->    <-/
                                        \      /
                                         \  | /
                                          ----
             */

            // Camera frustum 'left' and 'right' planes
            Eigen::Vector3d normal_left = camera.ground_model()->Normal(proj_bottom_left);
            Eigen::Vector3d normal_right = camera.ground_model()->Normal(proj_bottom_right);

            Eigen::Vector3d normal_left_plane = left_vec.cross(normal_left).normalized();
            Eigen::Vector3d normal_right_plane = right_vec.cross(normal_right).normalized() * -1.0;
            double d_left_plane = -1.0*normal_left_plane.dot(proj_top_left);
            double d_right_plane = -1.0*normal_right_plane.dot(proj_top_right);

            Eigen::Vector3d normal_principal_plane = Eigen::Vector3d(0.0, 0.0, 1.0);
            double d_principal_plane = 0.0;


            int end_tolerance = 0; //1; // How many points need to be behind the exit zone(s)?

            // For each hypothesis, check if it is out of camera frustum bounds!
            for (auto &hypo:hypotheses_) {
                if (hypo.cache().size() < 3)
                    continue;

                if (hypo.terminated().IsTerminated())
                    continue;

                // Don't terminate newly created hypos
                if (current_frame-hypo.creation_timestamp() < 10)
                    continue;

                // Count how many trajectory points falls out of bounds to these variables.
                int left_plane_count = 0;
                int right_plane_count = 0;
                int bottom_plane_count = 0;
                //int far_plane_count = 0;
                int far_count = 0;

                // Check last three traj. points
                for(auto j=hypo.cache().size()-1; j>(hypo.cache().size()-3); j--) {
                    auto hypo_point = hypo.cache().at_idx(j).pose();
                    hypo_point = camera.WorldToCamera(hypo_point);

                    // Dennis magic ...
                    double left_plane_check = hypo_point[0]*normal_left_plane[0] + hypo_point[1]*normal_left_plane[1] + hypo_point[2]*normal_left_plane[2] + d_left_plane;
                    double right_plane_check = hypo_point[0]*normal_right_plane[0] + hypo_point[1]*normal_right_plane[1] + hypo_point[2]*normal_right_plane[2] + d_right_plane;
                    double bottom_plane_check = hypo_point[0]*normal_principal_plane[0] + hypo_point[1]*normal_principal_plane[1] + hypo_point[2]*normal_principal_plane[2] + d_principal_plane;

                    if (left_plane_check < dist_lat)
                        left_plane_count ++;
                    if (right_plane_check < dist_lat)
                        right_plane_count ++;
                    if (bottom_plane_check < dist_rear)
                        bottom_plane_count ++;

                    if (hypo_point[2] > dist_far)
                        far_count ++;
                }

                if(left_plane_count > end_tolerance) {
                    if (verbose_) std::cout << "Hypo "<< hypo.id() << " entered LEFT EXIT ZONE." << std::endl;
                    hypo.set_terminated(GOT::tracking::TerminationInfo(true, current_frame));
                }
                else if(right_plane_count > end_tolerance) {
                    if (verbose_) std::cout << "Hypo "<< hypo.id() << " entered RIGHT EXIT ZONE." << std::endl;
                    hypo.set_terminated(GOT::tracking::TerminationInfo(true, current_frame));
                }
                else if (bottom_plane_count > end_tolerance /*|| (far_plane_count > 2)*/) {
                    if (verbose_) std::cout << "Hypo "<< hypo.id() << " entered BACK EXIT ZONE." << std::endl;
                    hypo.set_terminated(GOT::tracking::TerminationInfo(true, current_frame));
                }

                if (far_count > end_tolerance) {
                    if (verbose_) std::cout << "Hypo "<< hypo.id() << " went past FAR EXIT ZONE." << std::endl;
                    hypo.set_terminated(GOT::tracking::TerminationInfo(true, current_frame));
                }
            }
        }

        const HypothesesVector& MultiObjectTracker3DBase::hypotheses() const {
            return this->hypotheses_;
        }

        void MultiObjectTracker3DBase::set_verbose(bool verbose) {
            verbose_ = verbose;
        }

        void MultiObjectTracker3DBase::set_data_association_fnc(const MultiObjectTracker3DBase::DataAssocFnc &fnc) {
            this->data_association_scores_fnc_ = fnc;
        }
    }
}
