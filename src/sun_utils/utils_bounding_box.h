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

#ifndef GOT_UTILS_BOUNDING_BOX_H
#define GOT_UTILS_BOUNDING_BOX_H

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

// Forward declarations
namespace SUN { namespace utils { class Camera; }}

namespace SUN {
    namespace utils {
        namespace bbox {

            /**
             * @brief IntersectionOverUnion2d
             * @param[in] rect1
             * @param[in] rect2
             * @return Intersection over union, scalar
             */
            double IntersectionOverUnion2d(const Eigen::Vector4d &rect1, const Eigen::Vector4d &rect2);

            Eigen::Vector4d Intersection2d(const Eigen::Vector4d &rect1, const Eigen::Vector4d &rect2);

            Eigen::Vector4d BoundingBox2d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                          const std::vector<int> &indices, double percentage);

            Eigen::Vector4d EnlargeBoundingBox2d(const Eigen::Vector4d &bounding_box, double scale_x, double scale_y);

            Eigen::VectorXd BoundingBox3d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                          const std::vector<int> &indices, double percentage);

            /**
             * @brief Does-oriented-bounding-box-contain-a-point test (OBB)
             * @param[in] x
             * @param[in] y
             * @param[in] z
             * @param[in] bounding_box_3d
             * @return boolean indicating whether or not point is inside of the OBB
             */
            bool IsPointInOBB3d(const double x, const double y, const double z, const Eigen::VectorXd &bounding_box_3d,
                                double angleY);
        }
    }
}


#endif //GOT_UTILS_BOUNDING_BOX_H
