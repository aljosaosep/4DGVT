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

#include "scene_segmentation/scene_filtering.h"

// pcl
#include <pcl/io/io.h>

// boost
#include <boost/program_options.hpp>

// utils
#include "utils_filtering.h"
#include "camera.h"
#include "ground_model.h"

namespace GOT {
    namespace segmentation {
        namespace scene_filters {

            void GroundPlaneFilter(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_filter,
                                   const SUN::utils::Camera &camera,
                                   const boost::program_options::variables_map &parameter_map) {

                printf("Running GroundPlaneFilter ...\r\n");

                double plane_min_dist = parameter_map.at("filtering_min_distance_to_plane").as<double>();
                double plane_max_dist = parameter_map.at("filtering_max_distance_to_plane").as<double>();

                /// Prefilter the point-cloud
                // Remove ground-plane points, far-away points
                SUN::utils::filter::FilterPointCloudBasedOnDistanceToGroundPlane(cloud_to_filter, camera.ground_model(),
                                                                                 plane_min_dist, plane_max_dist, false);
            }

            void SemanticAndGroundPlaneFilter(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_filter,
                                              const SUN::utils::Camera &camera,
                                              const boost::program_options::variables_map &parameter_map,
                                              const cv::Mat &semantic_map) {

                printf("Running SemanticAndGroundPlaneFilter ...\r\n");

                // Height-based filtering params
                double plane_min_dist = parameter_map.at("filtering_min_distance_to_plane").as<double>();
                double plane_max_dist = parameter_map.at("filtering_max_distance_to_plane").as<double>();

                // Semantic filtering params, currently hard-coded
                // These are specific to the semantic maps utilized
                // TODO: make me more flexible!
                std::vector<cv::Vec3b> labels = {cv::Vec3b(0, 0, 255)/*road*/,
                                                 cv::Vec3b(153, 153, 0)/*grass*/};

                for (const auto &label:labels) {
                    SUN::utils::filter::FilterPointCloudBasedOnSemanticMapRemoveCategory(cloud_to_filter,
                                                                                         semantic_map,
                                                                                         label, false);
                }

                // Remove ground-plane points, far-away points
                SUN::utils::filter::FilterPointCloudBasedOnDistanceToGroundPlane(cloud_to_filter, camera.ground_model(),
                                                                                 plane_min_dist, plane_max_dist, false);
            }

            void SemanticFilterJakobSemseg(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_filter,
                                           const SUN::utils::Camera &camera,
                                           const boost::program_options::variables_map &parameter_map,
                                           const cv::Mat &semantic_map) {

                printf("Running SemanticFilterJakobSemseg ...\r\n");

                // Height-based filtering params
                double plane_min_dist = parameter_map.at("filtering_min_distance_to_plane").as<double>();
                double plane_max_dist = parameter_map.at("filtering_max_distance_to_plane").as<double>();

                // Semantic filtering params, currently hard-coded
                // These are specific to the semantic maps utilized
                // TODO: make me more flexible!
                std::vector<cv::Vec3b> labels = {cv::Vec3b(128, 64, 128)/*road*/,
                                                 cv::Vec3b(244, 35, 232)/*ped*/,
                                                 cv::Vec3b(70, 70, 70),/*building*/
                                                 cv::Vec3b(152, 251, 152)/*grass/dirt*/
                };

                for (const auto &label:labels) {
                    SUN::utils::filter::FilterPointCloudBasedOnSemanticMapRemoveCategory(cloud_to_filter,
                                                                                         semantic_map,
                                                                                         label, false);
                }

                // Remove ground-plane points, far-away points
                SUN::utils::filter::FilterPointCloudBasedOnDistanceToGroundPlane(cloud_to_filter, camera.ground_model(),
                                                                                 plane_min_dist, plane_max_dist, false);
            }

        }

    }
}