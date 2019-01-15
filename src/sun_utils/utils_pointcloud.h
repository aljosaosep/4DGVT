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


#ifndef SUN_UTILS_POINTCLOUD_H
#define SUN_UTILS_POINTCLOUD_H

// std
#include <cstdint>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>

namespace SUN { namespace utils { class Camera; }}

namespace SUN {
    namespace utils {
        namespace pointcloud {

            /**
             * @brief Converts disparity-map to pointcloud with specified pose.
             * Assumes pointCloud points to allocated memory!
             */
            void ConvertDisparityMapToPointCloud(
                    const cv::Mat &disparity_map,
                    const cv::Mat &color_image,
                    float c_u,
                    float c_v,
                    float focal_len,
                    float baseline,
                    const Eigen::Matrix4d &pose,
                    bool withNaN,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud,
                    double far_plane = 60.0);

            /**
             * @brief Takes raw 360 LiDAR scan,
             * returns portion that intersects with camera frustum (organized RGBD point cloud)
             */
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr
            RawLiDARCloudToImageAlignedAndOrganized(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr raw_lidar_cloud,
                                                    const Eigen::Matrix4d &T_lidar_to_cam,
                                                    const cv::Mat &image, const SUN::utils::Camera &camera);
        }
    }
}

#endif
