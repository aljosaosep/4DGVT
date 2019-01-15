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

#ifndef GOT_SCENE_FILTERING_H
#define GOT_SCENE_FILTERING_H

// pcl
#include <pcl/common/common.h>


// fwd. declr.
namespace SUN { namespace utils { class Camera; }}
namespace boost { namespace program_options { class variables_map; }}
namespace cv { class Mat; }

namespace GOT {
    namespace segmentation {
        namespace scene_filters {
            void GroundPlaneFilter(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud,
                                   const SUN::utils::Camera &camera,
                                   const boost::program_options::variables_map &parameter_map);

            void SemanticAndGroundPlaneFilter(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_filter,
                                              const SUN::utils::Camera &camera,
                                              const boost::program_options::variables_map &parameter_map,
                                              const cv::Mat &semantic_map);

            void SemanticFilterJakobSemseg(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_filter,
                                           const SUN::utils::Camera &camera,
                                           const boost::program_options::variables_map &parameter_map,
                                           const cv::Mat &semantic_map);
        }
    }
}


#endif //GOT_SCENE_FILTERING_H
