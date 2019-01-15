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

#ifndef GOT_UTILS_OBSERVATIONS_H
#define GOT_UTILS_OBSERVATIONS_H

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

// eigen
#include <Eigen/Core>

// opencv
#include <opencv2/core/core.hpp>

// utils
#include "camera.h"

namespace SUN {
    namespace utils {
        namespace observations {

            /**
             * @brief Computes velocity from the velocity measurements, corresponding to the 3D segment
             * @param[in] velocity_map Assumes velocity encoded in the channels (cv::Mat is of type float).
             * Important: NaN values means no velocity measurements!
             * @param[in] indices Segmentation mask
             * @param[in] dt Delta-time between two consecutive frames
             * @param[out] Velocity estimate for the given object
             */
            Eigen::Vector3d ComputeVelocity(const cv::Mat &velocity_map,
                                            const std::vector<int> &indices, double dt, int min_samples = 10);

            /**
             * @brief Computes 3D-point pos. covariance.
             * @param[in] pose_3d Estimated 3D position of the object
             * @param[in] indices Object segmentation mask
             * @param[in] P_left Projection matrix of the left camera
             * @param[in] P_right Projection matrix of the right camera
             * @param[in] min_num_points Min. points needed to estimate the covariance
             * @param[out] covariance_matrix_3d Estimated covariance matrix (ret. by ref.)
             */
            bool ComputePoseCovariance(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                       const Eigen::Vector4d &pose_3d,
                                       const std::vector<int> &indices,
                                       const Eigen::Matrix<double, 3, 4> &P_left,
                                       const Eigen::Matrix<double, 3, 4> &P_right,
                                       Eigen::Matrix3d &covariance_matrix_3d, int min_num_points = 10);

            /**
             * @brief Computes median 3D pose given the segmentation mask (indices)
             * @param[in] point_cloud The point-cloud (scene measurement).
             * @param[in] indices The segmentation indices (linear image-coordinates)
             * @param[out] median 3D point
             */
            bool ComputeDetectionPoseUsingStereo(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                 const std::vector<int> &indices, Eigen::Vector4d &median);
        }
    }
}

#endif //GOT_UTILS_OBSERVATIONS_H
