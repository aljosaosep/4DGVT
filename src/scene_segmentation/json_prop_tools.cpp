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

// std
#include <stdexcept>

// segmentation
#include <scene_segmentation/object_proposal.h>
#include <scene_segmentation/utils_segmentation.h>

// utils
#include "sun_utils/shared_types.h"
#include "sun_utils/utils_common.h"
#include "sun_utils/camera.h"
#include "sun_utils/ground_model.h"
#include "sun_utils/utils_observations.h"

namespace po = boost::program_options;

namespace GOT {
    namespace segmentation {
        namespace proposal_generation {

            /*
             * @brief Given binary segm. mask, returns a vector of linear inds, corresponding to 1-element set.
             */
            std::vector<int> SegmentationMaskToIndices(byte *mask, int width, int height) {

                assert(mask!= nullptr);
                std::vector<int> inds;
                for (int y=0; y<height; y++) {
                    for (int x=0; x<width; x++) {
                        byte val = mask[height*x + y];
                        if (val>0) {
                            int ind;
                            SUN::utils::RavelIndex(x, y, width, &ind);
                            inds.push_back(ind);
                        }
                    }
                }

                return inds;
            }

            std::vector<GOT::segmentation::ObjectProposal> ProposalsFromJson(int current_frame,
                                                                             const std::string &json_file,
                                                                             const SUN::utils::Camera &left_camera,
                                                                             const SUN::utils::Camera &right_camera,
                                                                             pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                                             const po::variables_map &parameter_map,
                                                                             int max_num_proposals) {

                std::ifstream i(json_file);
                if (!i.is_open()) {
                    throw std::runtime_error("Error, json file not found: " + json_file);
                }

                nlohmann::json json_proposals;
                i >> json_proposals;

                std::vector<GOT::segmentation::ObjectProposal> object_proposals;
                int object_no = 0;

                // -------------------------------------------------------------------------------
                // +++ FOR EVERY OBJECT (MASK) IN THE JSON ARRAY +++
                // -------------------------------------------------------------------------------
                //for (auto obj:value) { // Foreach 'object' in the array
                for (nlohmann::json &el : json_proposals) { // Foreach 'object' in the array
                    if (object_no>max_num_proposals) break;

                    /// Parse the JSON object -> score, RLE mask, size of the image
                    double score;
                    std::string counts;
                    int mw, mh;
                    int category_id;
                    int second_category_id;
                    std::vector<float> second_posterior;
                    GOT::segmentation::utils::ExtractBasicInfoFromJson(el, score, category_id, second_category_id,
                            counts, mw, mh, second_posterior);

                    char *compressed_str = &counts[0u];

                    /// Uncompress the mask data
                    RLE *R = new RLE;
                    siz w = mw, h = mh;
                    rleFrString(R, compressed_str, (siz) h, (siz) w);
                    const siz imsize = R->w * R->h;
                    auto M = new byte[imsize];
                    rleDecode(R, M, 1);

                    /// Turn binary segm. mask to indices
                    std::vector<int> inds = SegmentationMaskToIndices(M, static_cast<int>(w), static_cast<int>(h));

                    /// Turn RLE repr. directly to a 2D bounding-box
                    BB bb = new double[4];
                    rleToBbox(R, bb, 1);

                    /// Turn segmentation to 'object proposal' instance
                    GOT::segmentation::ObjectProposal proposal;
                    proposal.set_pointcloud_indices(inds, left_camera.width(), left_camera.height());
                    proposal.set_bounding_box_2d(Eigen::Vector4d(bb[0], bb[1], bb[2], bb[3]));
                    proposal.set_score(score);
                    proposal.set_category(category_id);
                    proposal.set_second_category(second_category_id);
                    proposal.set_second_posterior(second_posterior);
                    proposal.set_segm_id(object_no);

                    object_proposals.push_back(proposal);

                    /// Let it go
                    delete[] bb;
                    delete[] M;
                    rleFree(R);

                    object_no++;
                }


                // Non-max-supp
                bool do_non_max_supp = parameter_map.at("do_non_maxima_suppression").as<bool>();
                double non_max_supp_thresh = parameter_map.at("non_maxima_suppression_iou_threshold").as<double>();
                if (do_non_max_supp) {
                    const int size_before_suppression = object_proposals.size();
                    printf("Running non-max-supp ...\r\n");
                    std::vector<GOT::segmentation::ObjectProposal> suppressed =
                            GOT::segmentation::utils::NonMaximaSuppression(object_proposals, non_max_supp_thresh);
                    object_proposals = suppressed;
                    printf("Size before suppression: %d, size after: %d.\r\n",
                            size_before_suppression, (int)object_proposals.size());
                }

                std::vector<GOT::segmentation::ObjectProposal> proposals_for_export;
                int proposal_idx = 0;
                for (const auto &proposal:object_proposals) {
                    auto proposal_indices = proposal.pointcloud_indices();
                    Eigen::Vector4d seg_median;
                    if (SUN::utils::observations::ComputeDetectionPoseUsingStereo(point_cloud, proposal_indices, seg_median)) {

                        // Create ObjectProposal instance
                        Eigen::Vector4d gen_bbox_2d;
                        Eigen::VectorXd gen_bbox_3d;

                        const int MIN_NUM_POINTS = 10;

                        GOT::segmentation::ObjectProposal proposal_copy = proposal;
                        if (GOT::segmentation::utils::PointsToProposalRepresentation(point_cloud, proposal_indices,
                                gen_bbox_2d, gen_bbox_3d, MIN_NUM_POINTS)) {
                            // Set 3D-bbox
                            proposal_copy.set_bounding_box_3d(gen_bbox_3d);

                            // Set pose (median)
                            proposal_copy.set_pos3d(seg_median);

                            Eigen::Matrix3d pos_cov_mat;
                            left_camera.ComputeMeasurementCovariance3d(seg_median.head<3>(), 0.5,
                                                                       left_camera.P().block(0,0,3,4),
                                                                       right_camera.P().block(0,0,3,4), pos_cov_mat);

                            // Add some 'Gaussian prior'
                            pos_cov_mat(0,0) += 0.5;
                            pos_cov_mat(2,2) += 0.18;
                            proposal_copy.set_pose_covariance_matrix(pos_cov_mat);
                            proposals_for_export.push_back(proposal_copy);
                        }
                    }
                }

                return proposals_for_export;
            }
        }
    }
}


#include "scene_segmentation/json_prop_tools.h"
