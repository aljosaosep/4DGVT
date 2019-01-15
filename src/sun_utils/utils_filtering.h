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

#ifndef GOT_UTILS_FILTERING_H
#define GOT_UTILS_FILTERING_H

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

namespace SUN { namespace utils { class GroundModel; }}

// Forward declarations
namespace SUN { namespace utils { class Camera; }}

namespace SUN {
    namespace utils {
        namespace filter {

            /**
             * @brief Computes the median and keeps only the iner-quartile of the points closest to the median.
             * @param[in] input_cloud
             * @param[in] percentage
             * @return Filtered point-cloud.
             */
            std::vector<int> FilterKeepInnerqQuartile(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                      const std::vector<int> inds);

            /**
               * @brief Filters points based on distance from ground plane. WARNING: Requires gp-rectified point cloud!
               * @param[in] cloud_to_be_cleaned
               * @param[in] minDistance
               * @param[in] maxDistance
               * @param[in] only_color_outlier_points
               * @return Nothing. Filtering is done directly on input cloud.
               */
            void
            FilterPointCloudBasedOnDistanceToGroundPlane(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned,
                                                         std::shared_ptr<SUN::utils::GroundModel> ground_model,
                                                         const double minDistance = -0.15,
                                                         const double maxDistance = 3.0,
                                                         bool only_color_outlier_points = false);

            /**
               * @brief Filters all points, whose semantic label differs from the specified one (by semantic_label_to_keep).
               * WARNING: cv::Vec3b assumes BGR order, rather than RGB.
               * @param[in] cloud_to_be_cleaned
               * @param[in] semantic_map
               * @param[in] semantic_label_to_keep
               * @param[in] only_color_outlier_points
               * @return Nothing. Filtering is done directly on input cloud.
               */
            void FilterPointCloudBasedOnSemanticMap(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned,
                                                    const cv::Mat &semantic_map, const cv::Vec3b semantic_label_to_keep,
                                                    bool only_color_outlier_points = false);

            void FilterPointCloudBasedOnSemanticMapRemoveCategory(
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned,
                    const cv::Mat &semantic_map, const cv::Vec3b semantic_label_to_remove,
                    bool only_color_outlier_points);
        }
    }
}


#endif //GOT_UTILS_FILTERING_H
