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

#ifndef GOT_COCO_H
#define GOT_COCO_H

namespace GOT { namespace segmentation { class ObjectProposal; } }
namespace SUN { namespace utils { class Camera; } }

// pcl
#include <pcl/common/common_headers.h>

// boost
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace GOT {
    namespace segmentation {

        namespace proposal_generation {
            std::vector<GOT::segmentation::ObjectProposal> ProposalsFromJson(int current_frame,
                                                                             const std::string &json_file,
                                                                             const SUN::utils::Camera &left_camera,
                                                                             const SUN::utils::Camera &right_camera,
                                                                             pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                                             const po::variables_map &parameter_map,
                                                                             int max_num_proposals = 1000);
        }
    }
}


#endif //GOT_COCO_H
