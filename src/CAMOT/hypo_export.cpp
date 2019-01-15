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
#include <fstream>

// external
#include "external/json.hpp"

#include "hypo_export.h"

using json = nlohmann::json;

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {

            bool SerializeHyposPerFrameJson(const char *filename,
                                            int frame,
                                            const std::vector<GOT::tracking::Hypothesis> &hypos_to_export) {
                json j;


                for (const auto &hypo : hypos_to_export) {

                    // Only take hypos at are defn. at t=frame.
                    // Add entries for those.
                    if (hypo.cache().Exists(frame)) {
                        json track_data_to_add = json::object(); // Json hypo obj
                        const auto &hypo_data = hypo.cache().at_frame(frame);

                        // Compute assoc. score
                        std::vector<HypothesisInlier> hypo_inliers = hypo.inliers();

                        double assoc_scores = 0.0, inlier_scores = 0.0;
                        int track_len = 0;
                        for (const HypothesisInlier &inlier:hypo_inliers) {
                            assoc_scores += inlier.association_score_;
                            inlier_scores += inlier.inlier_score_;
                            track_len++;
                        }

                        // Thingies to export
                        const Eigen::Vector4d &pos_cam = hypo_data.pose_cam();
                        const Eigen::Vector4i bbox2 = hypo_data.box2().cast<int>();
                        const Eigen::Vector4i bbox_ext = hypo_data.box_ext().cast<int>();
                        const Eigen::VectorXd &bbox3 = hypo_data.box3();
                        const auto &mask = hypo_data.mask();

                        // Add to json::obj
                        track_data_to_add.push_back({"pos3d_cam", json::array({pos_cam[0], pos_cam[1], pos_cam[2]})});
                        track_data_to_add.push_back({"bbox", json::array({bbox2[0], bbox2[1], bbox2[2], bbox2[3]})});
                        track_data_to_add.push_back(
                                {"bbox_ext", json::array({bbox_ext[0], bbox_ext[1], bbox_ext[2], bbox_ext[3]})});
                        track_data_to_add.push_back(
                                {"bbox3d", json::array({bbox3[0], bbox3[1], bbox3[2], bbox3[3], bbox3[4], bbox3[5]})});
                        track_data_to_add.push_back({"segmentation", json::object(
                                {{"size",   json::array({mask.h_, mask.w_})},
                                 {"counts", mask.rle_string_}})});

                        const auto hp = hypo.category_probability_distribution();
                        int cat_id = static_cast<int>(
                                std::distance(hp.begin(), std::max_element(hp.begin(), hp.end()))
                        );

                        track_data_to_add.push_back({"category_id", cat_id});
                        track_data_to_add.push_back({"track_id", hypo.id()});
                        track_data_to_add.push_back({"score", hypo.score()});

//                        // Extra scoring info
//                        track_data_to_add.push_back({"association_scores", assoc_scores});
//                        track_data_to_add.push_back({"inlier_scores", inlier_scores});
//                        track_data_to_add.push_back({"track_length", track_len});
//
//                        // Extra extra scoring info
//                        json mask_scores, motion_scores, segm_scores, tstamps, mh_dists;
//                        for (const HypothesisInlier &inlier:hypo_inliers) {
//                            if (inlier.assoc_data_.size() > 0) {
//                                auto inl_score = inlier.inlier_score_;
//                                auto mask_score = inlier.assoc_data_.at(0);
//                                auto motion_score = inlier.assoc_data_.at(1);
//                                auto mh_dist_sq = inlier.assoc_data_.at(2);
//
//                                mask_scores.push_back(mask_score);
//                                motion_scores.push_back(motion_score);
//                                segm_scores.push_back(inl_score);
//                                mh_dists.push_back(mh_dist_sq);
//                                tstamps.push_back(inlier.timestamp_);
//                            }
//                        }
//
//                        track_data_to_add.push_back({"mask_iou_scores_vec", mask_scores});
//                        track_data_to_add.push_back({"motion_scores_vec", motion_scores});
//                        track_data_to_add.push_back({"segmentation_scores_vec", segm_scores});
//                        track_data_to_add.push_back({"timestamps_vec", tstamps});
//                        track_data_to_add.push_back({"mahalanobis_squared_vec", mh_dists});
//                        track_data_to_add.push_back({"frame", frame});

                        j.push_back(track_data_to_add);
                    }

                }

                // Serialize
                std::ofstream output(filename);
                if (!output.is_open()) {
                    std::cerr << "Failed exporting track data to json :/" << std::endl;
                    return false;
                }

                output << j << std::endl;
                output.close();

                return true;
            }

            bool SerializeHyposJson(const char *filename,
                                    const std::vector<GOT::tracking::Hypothesis> &hypos_to_export,
                                    const std::map<int, std::string> &label_mapper) {
                json j;
                for (const auto &hypo : hypos_to_export) {
                    json cached_data = json::object(); // Json hypo obj

                    for (int frame = hypo.cache().start_frame(); frame < hypo.cache().curr_frame(); frame++) {

                        json curr_fr_cache = json::object(); // Json hypo obj
                        const auto &hypo_data = hypo.cache().at_frame(frame);

                        // Thingies to export
                        const Eigen::Vector4d &pos = hypo_data.pose();
                        const Eigen::Vector4d &pos_cam = hypo_data.pose_cam();
                        const Eigen::Vector4i bbox2 = hypo_data.box2().cast<int>();
                        const Eigen::Vector4i bbox_ext = hypo_data.box_ext().cast<int>();
                        const Eigen::VectorXd &bbox3 = hypo_data.box3();
                        const auto &mask = hypo_data.mask();

                        // Add to json::obj
                        curr_fr_cache.push_back({"pos3d", json::array({pos[0], pos[1], pos[2]})});
                        curr_fr_cache.push_back({"pos3d_cam", json::array({pos_cam[0], pos_cam[1], pos_cam[2]})});
                        curr_fr_cache.push_back({"bbox", json::array({bbox2[0], bbox2[1], bbox2[2], bbox2[3]})});
                        curr_fr_cache.push_back(
                                {"bbox_ext", json::array({bbox_ext[0], bbox_ext[1], bbox_ext[2], bbox_ext[3]})});
                        curr_fr_cache.push_back(
                                {"bbox3d", json::array({bbox3[0], bbox3[1], bbox3[2], bbox3[3], bbox3[4], bbox3[5]})});
                        curr_fr_cache.push_back({"segm", json::object({{"size",   json::array({mask.h_, mask.w_})},
                                                                       {"counts", mask.rle_string_}})});
                        curr_fr_cache.push_back({"segm_id", hypo_data.segm_id()});

                        cached_data.push_back({std::to_string(frame), curr_fr_cache});
                    }


                    json jprop = json::object(); // Json hypo obj
                    const auto hp = hypo.category_probability_distribution();
                    int cat_id = static_cast<int>(
                            std::distance(hp.begin(), std::max_element(hp.begin(), hp.end()))
                    );

                    jprop.push_back({"category_id", cat_id});
                    jprop.push_back({"category_name", label_mapper.at(cat_id)});
                    jprop.push_back({"score", hypo.score()});
                    jprop.push_back({"termination", hypo.terminated().FrameTerminated()});
                    jprop.push_back({"id", hypo.id()});
                    jprop.push_back({"track_info", cached_data});

                    // ========= Inlier info ==============
                    // Extra extra scoring info
                    json mask_scores, motion_scores, segm_scores, tstamps, mh_dists;
                    const auto &hypo_inliers = hypo.inliers();
                    for (const HypothesisInlier &inlier:hypo_inliers) {
                        if (inlier.assoc_data_.size() > 0) {
                            auto inl_score = inlier.inlier_score_;
                            auto mask_score = inlier.assoc_data_.at(0);
                            auto motion_score = inlier.assoc_data_.at(1);
                            auto mh_dist_sq = inlier.assoc_data_.at(2);

                            mask_scores.push_back(mask_score);
                            motion_scores.push_back(motion_score);
                            segm_scores.push_back(inl_score);
                            mh_dists.push_back(mh_dist_sq);
                            tstamps.push_back(inlier.timestamp_);
                        }
                    }

                    jprop.push_back({"mask_iou_scores_vec", mask_scores});
                    jprop.push_back({"motion_scores_vec", motion_scores});
                    jprop.push_back({"segmentation_scores_vec", segm_scores});
                    jprop.push_back({"timestamps_vec", tstamps});
                    jprop.push_back({"mahalanobis_squared_vec", mh_dists});
                    // ====================================

                    // Add obj info to the json
                    j.push_back(jprop);
                }

                // Serialize
                std::ofstream output(filename);
                if (!output.is_open()) {
                    std::cerr << "Failed exporting track data to json :/" << std::endl;
                    return false;
                }

                output << j << std::endl;
                output.close();

                return true;
            }
        }
    }
}