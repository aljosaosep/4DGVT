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

#include <scene_segmentation/utils_segmentation.h>

// boost
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

// utils
#include "utils_observations.h"
#include "camera.h"
#include "utils_kitti.h"
#include "ground_model.h"
#include "utils_common.h"
#include "utils_bounding_box.h"

using json = nlohmann::json;

namespace GOT {
    namespace segmentation {
        namespace utils {

            bool PointsToProposalRepresentation(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                                const std::vector<int> &object_indices,
                                                Eigen::Vector4d &bbox2d,
                                                Eigen::VectorXd &bbox3d, int min_cluster_size) {

                const double percentage = 1.0;
                if (object_indices.size() >= min_cluster_size) {
                    bbox2d = SUN::utils::bbox::BoundingBox2d(scene_cloud, object_indices, percentage);
                    bbox3d = SUN::utils::bbox::BoundingBox3d(scene_cloud, object_indices, percentage);
                    return true;
                }

                return false;
            }

            auto RectGaitingFnc = [](const ObjectProposal &obj1, const ObjectProposal &obj2, double thresh) -> bool {

                if (SUN::utils::bbox::IntersectionOverUnion2d(obj1.bounding_box_2d(), obj2.bounding_box_2d()) >
                    thresh) {
                    return true;
                }

                return false;
            };

            auto MaskIndicesGaitingFnc = [](const ObjectProposal &obj1, const ObjectProposal &obj2,
                                            double thresh) -> bool {
                if (SUN::utils::InterSectionOverUnionArrays(obj1.pointcloud_indices(), obj2.pointcloud_indices()) >
                    thresh) {
                    return true;
                }

                return false;
            };

            std::vector<ObjectProposal>
            MultiScaleSuppression(const std::vector<std::vector<ObjectProposal> > &proposals_per_scale,
                                  double IOU_threshold) {

                std::vector<ObjectProposal> accepted_proposals;
                for (const auto &current_scale_proposals:proposals_per_scale) {
                    // Loop through current-scale proposals. If you find IOU overlap with existing, merge the two.
                    // Otherwise, push to list.
                    for (const auto &prop_scale:current_scale_proposals) {
                        bool overlap_found = false;
                        for (auto &prop_keep:accepted_proposals) {
                            if (RectGaitingFnc(prop_scale, prop_keep, IOU_threshold)) { // DefaultGaitingFnc

                                overlap_found = true;
                                Eigen::MatrixXd merged_box2 = (prop_keep.bounding_box_2d()
                                                               + prop_scale.bounding_box_2d()) / 2.0;
                                //Eigen::MatrixXd merged_bbox_3d = (prop_keep.bounding_box_3d() + prop_scale.bounding_box_3d())/2.0;
                                Eigen::Vector4d merged_pos3 = (prop_keep.pos3d() + prop_scale.pos3d()) / 2.0;

                                // Merge masks
                                std::vector<int> merged_inds = prop_keep.pointcloud_indices();
                                std::vector<int> new_inds = prop_scale.pointcloud_indices();
                                merged_inds.insert(merged_inds.end(), new_inds.begin(), new_inds.end());
                                std::sort(merged_inds.begin(), merged_inds.end());
                                merged_inds.erase(std::unique(merged_inds.begin(), merged_inds.end()),
                                                  merged_inds.end());
                                prop_keep.set_pointcloud_indices(merged_inds,
                                                                 prop_keep.compressed_mask().w_,
                                                                 prop_keep.compressed_mask().h_);


                                // Set merged
                                prop_keep.set_pos3d(merged_pos3);
                                // TODO: box2, box3?

                                break;
                            }
                        }

                        if (!overlap_found)
                            accepted_proposals.push_back(prop_scale);
                    }
                }

                return accepted_proposals;
            }


            /*
             * @brief: Proposal NMS
             */
            std::vector<GOT::segmentation::ObjectProposal>
            NonMaximaSuppression(const std::vector<GOT::segmentation::ObjectProposal> &proposals_in,
                                 double iou_threshold) {

                std::vector<GOT::segmentation::ObjectProposal> active_set = proposals_in;
                std::vector<GOT::segmentation::ObjectProposal> final_supressed_set;

                while (active_set.size() > 0) {

                    // Get best-scoring element
                    auto it_to_max_element = std::max_element(active_set.begin(), active_set.end(),
                                                              [](const GOT::segmentation::ObjectProposal &a,
                                                                 const GOT::segmentation::ObjectProposal &b) {
                                                                  return a.score() < b.score();
                                                              });

                    GOT::segmentation::ObjectProposal best_scoring = *it_to_max_element;
                    std::set<int> overlapping_set;
                    std::vector<GOT::segmentation::ObjectProposal> overlapping_set_props;
                    for (int i = 0; i < active_set.size(); i++) {
                        const GOT::segmentation::ObjectProposal &element_to_test = active_set.at(i);
                        double geom_IOU = SUN::utils::bbox::IntersectionOverUnion2d(best_scoring.bounding_box_2d(),
                                                                                    element_to_test.bounding_box_2d());

                        // Skip if 2D-rect IoU < 0.5
                        if (geom_IOU < 0.5)
                            continue;

                        // Use compressed masks to compute IoU in O(sqrt(mask_area))
                        double IOU = element_to_test.compressed_mask().IoU(best_scoring.compressed_mask());

                        if (IOU > iou_threshold) {
                            overlapping_set_props.push_back(element_to_test);
                            overlapping_set.insert(i);
                        }
                    }

                    final_supressed_set.push_back(best_scoring);

                    std::vector<GOT::segmentation::ObjectProposal> proposals_to_keep;
                    for (int i = 0; i < active_set.size(); i++) {
                        if (overlapping_set.count(i) == 0 && overlapping_set.size() > 0) {
                            proposals_to_keep.push_back(active_set.at(i));
                        }
                    }
                    active_set = proposals_to_keep;
                }
                return final_supressed_set;
            }

            bool SerializeJson(const char *filename, const std::vector<ObjectProposal> &proposals) {
                json j;

                // Foreach proposal
                for (const auto &prop:proposals) {

                    // Get neccessary info
                    const Eigen::Vector4i bbox2 = prop.bounding_box_2d().cast<int>(); // bbox2D
                    const auto &bbox3 = prop.bounding_box_3d();
                    const Eigen::Vector4d &pos = prop.pos3d();
                    const auto &prop_mask = prop.compressed_mask();
                    const Eigen::Matrix3d &cov_mat = prop.pose_covariance_matrix();
                    const auto &gp_inds = prop.ground_plane_indices();

                    json jprop = json::object();

                    // Basic COCO segm. info (note: segm. order h, w is correct)
                    jprop.push_back({"segm", json::object({{"size",   json::array({prop_mask.h_, prop_mask.w_})},
                                                           {"counts", prop_mask.rle_string_}})});
                    jprop.push_back({"bbox", json::array({bbox2[0], bbox2[1], bbox2[2], bbox2[3]})});
                    jprop.push_back({"score", prop.score()});
                    jprop.push_back({"category_id", prop.category()});
                    jprop.push_back({"segm_id", prop.segm_id()});
                    jprop.push_back({"second_category_id", prop.second_category()});

                    // Pos3D
                    jprop.push_back({"pos3d", json::array({pos[0], pos[1], pos[2]})});

                    // Write bbox3D
                    jprop.push_back({"bbox3d", json::array({bbox3[0], bbox3[1], bbox3[2],
                                                            bbox3[3], bbox3[4], bbox3[5]})});

                    // Write cov mat
                    auto jcov_mat = json::array();
                    for (int i = 0; i < cov_mat.size(); i++) {
                        jcov_mat.push_back(cov_mat(i));
                    }
                    jprop.push_back({"cov_mat", jcov_mat});

                    // Write gp inds (TODO: mask?)
                    auto jgp_inds = json::array();
                    for (const auto ind:gp_inds) {
                        jgp_inds.push_back(ind);
                    }
                    jprop.push_back({"gp_inds", jgp_inds});


                    // Second posterior (encode mostly-zero vec to size, index:value pairs)
                    const auto &second_post = prop.second_posterior();
                    json second_post_json;
                    for (auto i = 0; i < second_post.size(); i++) {
                        auto val = second_post.at(i);
                        if (val > 0.00001) {
                            second_post_json[std::to_string(i)] = val;
                        }
                    }

                    jprop.push_back({"second_posterior",
                                     json::object({{"size",   second_post.size()},
                                                   {"values", second_post_json}})});

                    // Add obj info to the json
                    j.push_back(jprop);
                }

                // Serialize
                std::ofstream output(filename);
                if (!output.is_open()) {
                    std::cerr << "Failed serializing proposals to json." << std::endl;
                    return false;
                }

                //output << std::setw(4) << j << std::endl;
                output << j << std::endl;
                return true;
            }

            bool SafeEntryCheck(const json &el, const std::string &element_id, bool verbose = true) {

                if (el.count(element_id)) {
                    return true;
                }

                if (verbose) {
                    std::cout << "DeserializeJson: Warning! Element missing: " << element_id << std::endl;
                }

                return false;
            }

            bool ExtractBasicInfoFromJson(const json &json_prop,
                                          double &score,
                                          int &category_id,
                                          int &second_category_id,
                                          std::string &mask_counts,
                                          int &mask_im_w,
                                          int &mask_im_h,
                                          std::vector<float> &second_posterior) {

                // Category id
                if (!SafeEntryCheck(json_prop, "category_id")) {
                    return false;
                }

                // Second category id
                if (!SafeEntryCheck(json_prop, "second_category_id")) {
                    std::cout << "ExtractBasicInfoFromJson Warning: No second category id." << std::endl;
                    second_category_id = 0;
                } else {
                    second_category_id = json_prop.at("second_category_id").get<int>(); // TODO !!!
                }

                // Score
                if (!SafeEntryCheck(json_prop, "score")) {
                    return false;
                }

                // TODO: Test me
                if (!SafeEntryCheck(json_prop, "second_posterior", false)) {
                    std::cout << "ExtractBasicInfoFromJson Warning: No second posterior." << std::endl;
                    second_posterior = {0.0}; //std::vector<float>
                } else {
                    // Decode the compressed second posterior
                    json j_sec_post = json_prop.at("second_posterior"); // TODO !!!
                    int j_sec_post_size = j_sec_post.at("size").get<int>();
                    json j_sec_post_vals = j_sec_post.at("values");
                    std::vector<float> second_post(static_cast<size_t>(j_sec_post_size), 0.0f);
                    for (json::iterator it = j_sec_post_vals.begin(); it != j_sec_post_vals.end(); ++it) {
                        //std::cout << it.key() << " : " << it.value() << "\n";
                        int idx = std::stoi(it.key());
                        auto val = it.value().get<float>();
                        second_post.at(idx) = val;
                    }
                    second_posterior = second_post;
                }

                auto get_mask_info = [&json_prop](const std::string &mask_token, std::string &counts, int &maskw,
                                                  int &maskh) {
                    json jmask = json_prop.at(mask_token);
                    json jsize = jmask.at("size");
                    counts = jmask.at("counts").get<std::string>();
                    maskh = jsize.front().get<int>();
                    maskw = jsize.back().get<int>();
                };

                // Mask, inds; support both, 'segm' and 'segmentation'
                bool got_mask = false;

                // Set mask entries
                if (SafeEntryCheck(json_prop, "segm", false)) {
                    got_mask = true;
                    get_mask_info("segm", mask_counts, mask_im_w, mask_im_h);
                }

                if (SafeEntryCheck(json_prop, "segmentation", false)) {
                    got_mask = true;
                    get_mask_info("segmentation", mask_counts, mask_im_w, mask_im_h);
                }

                // Set other entries
                category_id = json_prop.at("category_id").get<int>();
                score = json_prop.at("score").get<double>();

                return got_mask;
            }

            bool
            DeserializeJson(const char *filename, std::vector<ObjectProposal> &proposals, int max_proposals_to_proc) {

                auto json_array_to_eigen = [](const json &jobj) -> Eigen::VectorXd {
                    Eigen::VectorXd vec_out;
                    vec_out.setZero(jobj.size());
                    int idx = 0;
                    for (auto val : jobj) {
                        vec_out[idx++] = val.get<double>();
                    }

                    return vec_out;
                };

                proposals.clear();

                std::ifstream i(filename);
                if (!i.is_open()) {
                    return false;
                }

                json j;
                i >> j;

                proposals.reserve(j.size());

                // Extract data
                int prop_idx = 0;
                for (json &el : j) {
                    if (prop_idx++ > max_proposals_to_proc) {
                        break;
                    }

                    ObjectProposal prop;

                    /// Parse-out the most basic proposal info first
                    double score;
                    int category_id;
                    int second_category_id;
                    std::string mask_counts;
                    int mask_im_w;
                    int mask_im_h;
                    std::vector<float> second_posterior;
                    if (!ExtractBasicInfoFromJson(el, score, category_id, second_category_id, mask_counts, mask_im_w,
                                                  mask_im_h, second_posterior)) {
                        return false;
                    }

                    prop.set_category(category_id);
                    prop.set_second_category(second_category_id);
                    prop.set_score(score);
                    prop.set_pointcloud_indices(mask_counts, mask_im_w, mask_im_h);
                    prop.set_second_posterior(second_posterior);

                    /// Parse the "extras"

                    // Pos 3D
                    if (SafeEntryCheck(el, "pos3d")) {
                        Eigen::VectorXd pos3 = json_array_to_eigen(el.at("pos3d"));
                        if (pos3.size() != 3) {
                            std::cerr << "Error parsing pos3d!" << std::endl;
                            return false;
                        }
                        prop.set_pos3d(Eigen::Vector4d(pos3[0], pos3[1], pos3[2], 1.0));
                    }

                    // Segm id
                    if (SafeEntryCheck(el, "segm_id")) {
                        int segm_id = el.at("segm_id").get<int>();
                        prop.set_segm_id(segm_id);
                    }

                    // Bounding box 2D
                    if (SafeEntryCheck(el, "bbox")) {
                        Eigen::VectorXd bb_2d = json_array_to_eigen(el.at("bbox"));
                        if (bb_2d.size() != 4) {
                            std::cerr << "Error parsing bbox2d!" << std::endl;
                            return false;
                        }
                        prop.set_bounding_box_2d(bb_2d);
                    }

                    // Bounding box 3D
                    if (SafeEntryCheck(el, "bbox3d")) {
                        Eigen::VectorXd bb_3d = json_array_to_eigen(el.at("bbox3d"));
                        if (bb_3d.size() != 6) {
                            std::cerr << "Error parsing bbox3d!" << std::endl;
                            return false;
                        }
                        prop.set_bounding_box_3d(bb_3d);
                    }

                    // Ground plane inds
                    if (SafeEntryCheck(el, "gp_inds")) {
                        json jgpi = el.at("gp_inds");
                        if (jgpi.size() > 0) {
                            std::vector<int> gp_inds;
                            gp_inds.reserve(jgpi.size());
                            for (auto val : jgpi) {
                                gp_inds.push_back(val.get<int>());
                            }
                            prop.set_groundplane_indices(gp_inds);
                        }
                    }

                    // Pose cov. mat
                    if (SafeEntryCheck(el, "cov_mat")) {
                        Eigen::VectorXd cov_vec = json_array_to_eigen(el.at("cov_mat"));
                        if (cov_vec.size() != 9) {
                            std::cerr << "Error parsing cov mat!" << std::endl;
                            return false;
                        }
                        Eigen::Matrix3d pose_cov_mat;
                        for (int i = 0; i < cov_vec.size(); i++) {
                            pose_cov_mat(i) = cov_vec[i];
                        }
                        prop.set_pose_covariance_matrix(pose_cov_mat);
                    }

                    proposals.push_back(prop);
                }

                return true;
            }

        }
    }
}