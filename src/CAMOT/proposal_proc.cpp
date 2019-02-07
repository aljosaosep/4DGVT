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

#include "proposal_proc.h"

// segmentation
#include <scene_segmentation/utils_segmentation.h>

// utils
#include "ground_model.h"
#include "camera.h"


namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {
            namespace proposal_utils {

                GOT::segmentation::ObjectProposal::Vector
                GeometricFiltering(const SUN::utils::Camera &ref_camera,
                                   const GOT::segmentation::ObjectProposal::Vector &proposals_in,
                                   const po::variables_map &variables_map) {
                    GOT::segmentation::ObjectProposal::Vector proposals_tmp;
                    for (const auto &prop:proposals_in) {

                        const double rear = variables_map.at("proposals_geometric_filter_near").as<double>();
                        const double far = variables_map.at("proposals_geometric_filter_far").as<double>();
                        const double lateral = variables_map.at("proposals_geometric_filter_lateral").as<double>();
                        const double min_plane = variables_map.at(
                                "proposals_geometric_min_distance_to_plane").as<double>();
                        const double max_plane = variables_map.at(
                                "proposals_geometric_max_distance_to_plane").as<double>();

                        // Based on distance-to-plane
                        double dist_to_plane = ref_camera.ground_model()->DistanceToGround(prop.pos3d().head<3>());
                        const bool height_cond = (dist_to_plane > min_plane && dist_to_plane < max_plane);

                        // Based on left, right, near, far ranges
                        Eigen::Vector3d p_ground = ref_camera.ground_model()->ProjectPointToGround(
                                prop.pos3d().head<3>());
                        const double X = p_ground[0];
                        const double Z = p_ground[2];
                        const bool range_cond = ((std::fabs(X) < lateral) && (Z > rear) && (Z < far));

                        if (height_cond && range_cond) {
                            proposals_tmp.push_back(prop);
                        }
                    }

                    return proposals_tmp;
                }

                GOT::segmentation::ObjectProposal::Vector
                ProposalsConfidenceFilter(const GOT::segmentation::ObjectProposal::Vector &obj_proposals_in,
                                          double thresh) {
                    GOT::segmentation::ObjectProposal::Vector obj_proposals_out;
                    obj_proposals_out.reserve(obj_proposals_in.size());
                    for (const auto &prop:obj_proposals_in) {
                        if (prop.score() > thresh)
                            obj_proposals_out.push_back(prop);
                    }

                    return obj_proposals_out;
                }
            }
        }
    }
}
