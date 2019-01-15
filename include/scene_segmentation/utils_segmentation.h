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

#ifndef GOT_SCENE_SEGMENTATION_UTILS
#define GOT_SCENE_SEGMENTATION_UTILS

// OpenCV
#include <opencv2/core/core.hpp>

// boost
#include <boost/program_options.hpp>

// external
#include "external/json.hpp"

// Segmentation
#include <scene_segmentation/object_proposal.h>
#include <scene_segmentation/ground_histogram.h>

// Utils
#include "sun_utils/utils_kitti.h"

namespace po = boost::program_options;

namespace GOT {
    namespace segmentation {

        namespace utils {

            /*
             * @brief: Cloud + inds => proposal; assumed organized pcl
             */
            bool PointsToProposalRepresentation(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                                const std::vector<int> &object_indices,
                                                Eigen::Vector4d &bbox2d,
                                                Eigen::VectorXd &bbox3d,
                                                int min_cluster_size = 100);

            /*
             * @brief: Cloud + inds => proposal; assumed NON-organized pcl
             */
            bool PointsToProposalRepresentation(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr segment_cloud,
                                                const SUN::utils::Camera &camera,
                                                Eigen::Vector4d &bbox2d,
                                                Eigen::VectorXd &bbox3d,
                                                int min_cluster_size = 100);

            /*
             * @brief: Multi-scale sort-of proposal supression (as desc. in Osep etal., ICRA'16)
             */
            std::vector<ObjectProposal>
            MultiScaleSuppression(const std::vector<std::vector<ObjectProposal>> &proposals_per_scale,
                                  double IOU_threshold);

            /*
             * @brief: Traditional NMS
             */
            std::vector<GOT::segmentation::ObjectProposal>
            NonMaximaSuppression(const std::vector<GOT::segmentation::ObjectProposal> &proposals_in,
                                 double iou_threshold);


            bool ExtractBasicInfoFromJson(const nlohmann::json &json_prop,
                                          double &score,
                                          int &category_id,
                                          int &second_category_id,
                                          std::string &mask_counts,
                                          int &mask_im_w,
                                          int &mask_im_h,
                                          std::vector<float> &second_posterior);

            /*
              * @brief: Export proposals -> json
              */
            bool SerializeJson(const char *filename, const std::vector<ObjectProposal> &proposals);

            /*
              * @brief: Read proposals from a json file
              */
            bool DeserializeJson(const char *filename, std::vector<ObjectProposal> &proposals,
                                 int max_proposals_to_proc = 500);
        }
    }
}

#endif
