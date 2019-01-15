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

#include <stdexcept>

#include "inference.h"

// tracking lib
#include <tracking/hypothesis.h>
#include <tracking/qpbo.h>

//utils
#include "utils_bounding_box.h"

// CAMOT
#include "QPBO_fnc.h"

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {

            f_unary GetUnaryFnc(const std::string &unary_str) {
                auto unary_fnc = GOT::tracking::CAMOT_tracker::UnaryFncLogRatios;
                auto unary_fnc_str_lower = unary_str;
                std::transform(unary_fnc_str_lower.begin(), unary_fnc_str_lower.end(), unary_fnc_str_lower.begin(), ::tolower);
                if (unary_fnc_str_lower == "camot") {
                    std::cout << "[ Using CAMOT model ... ]" << std::endl;
                    unary_fnc = GOT::tracking::CAMOT_tracker::UnaryFncCAMOT;
                } else if (unary_fnc_str_lower == "4dgvt") {
                    std::cout << "[ Using 4DGVT model ... ]" << std::endl;
                } else {
                    throw std::runtime_error(unary_fnc_str_lower +
                                                     " is invalid unary function specifier (supported [CAMOT|4DGVT])");
                }

                return unary_fnc;
            }

            double UnaryFncCAMOT(const GOT::tracking::Hypothesis &hypo, int frame,
                                 const boost::program_options::variables_map &var_map) {

                // Sanity checks
                if ((var_map.count("inf_tau") + var_map.count("inf_window_size") + var_map.count("inf_w1") +
                     var_map.count("inf_w2")) != 4) {
                    throw std::runtime_error("CAMOT_tracker::UnaryFncCAMOT error, one of the params not specified!");
                }

                auto tau = var_map.at("inf_tau").as<int>();
                auto window_size = var_map.at("inf_window_size").as<int>();
                auto w1 = var_map.at("inf_w1").as<double>();
                auto w2 = var_map.at("inf_w2").as<double>();

                int window_start_frame = std::max(0, (frame - (window_size / 2)));
                int window_end_frame = frame + (window_size / 2);

                double assoc_scores = 0.0;
                double inlier_scores = 0.0;
                double normalizer = 0.0;
                for (const auto &inlier:hypo.inliers()) {
                    int inlier_t = inlier.timestamp_;
                    if (inlier_t >= window_start_frame && inlier_t <= window_end_frame) {
                        double exp_decay = GOT::tracking::CAMOT_tracker::ExpDecay(frame, inlier_t, tau);

                        assoc_scores += exp_decay * inlier.association_score_;
                        inlier_scores += exp_decay * inlier.inlier_score_;

                        normalizer += exp_decay;
                    }
                }

                double unary = w2 * assoc_scores + w1 * inlier_scores;
                return unary;
            }

            double UnaryFncHypoScore(const GOT::tracking::Hypothesis &hypo, int frame,
                                     const boost::program_options::variables_map &var_map) {
                return hypo.score();
            }

            using f_get_attr = std::function<double(const HypothesisInlier &)>;

            double ComputeLogRatios(const std::vector<HypothesisInlier> &inliers, f_get_attr f_attr, int K) {
                double sc_sum = 0.0;
                int num_intl_with_assoc_data = 0;
                for (int i = 1; i < inliers.size(); i++) {
                    const auto &inl = inliers.at(i);
                    if (inl.assoc_data_.size() == 0) {
                        continue;
                    }
                    double attr = f_attr(inl);
                    sc_sum += log(attr);
                    num_intl_with_assoc_data++;
                }
                return sc_sum - num_intl_with_assoc_data * log(1.0 / static_cast<double>(K));
            }

            double UnaryFncLogRatios(const GOT::tracking::Hypothesis &hypo, int frame,
                                     const boost::program_options::variables_map &var_map) {

                auto f_inl_score = [](const HypothesisInlier &inl) -> double {
                    return inl.inlier_score_;
                };

                auto f_motion_score = [](const HypothesisInlier &inl) -> double {
                    return inl.assoc_data_.at(1);
                };

                auto f_mask_consistency_score = [](const HypothesisInlier &inl) -> double {
                    return inl.assoc_data_.at(0);
                };

                if ((var_map.count("inf_alpha") + var_map.count("inf_beta")) != 2) {
                    throw std::runtime_error("CAMOT_tracker::UnaryFncCAMOT error, one of the params not specified!");
                }

                auto alpha = var_map.at("inf_alpha").as<double>();
                auto beta = var_map.at("inf_beta").as<double>();
                std::vector<HypothesisInlier> hypo_inliers = hypo.inliers();

                double objectness = ComputeLogRatios(hypo_inliers, f_inl_score, 10);
                double motion = ComputeLogRatios(hypo_inliers, f_motion_score, 1000);
                double mask_consistency = ComputeLogRatios(hypo_inliers, f_mask_consistency_score, 500);

                double score = beta * (alpha * motion + (1.0 - alpha) * mask_consistency) + (1.0 - beta) * objectness;
                return score;
            }


            auto f_overlap_mask_iom = [](const GOT::tracking::Hypothesis &h1, const GOT::tracking::Hypothesis &h2,
                                         const GOT::tracking::HypothesisInlier &i1,
                                         const GOT::tracking::HypothesisInlier &i2) -> double {
                double overlap = 0.0;
                const Eigen::Vector4d &bb1 = h1.cache().at_frame(i1.timestamp_).box2();
                const Eigen::Vector4d &bb2 = h2.cache().at_frame(i2.timestamp_).box2();
                if (SUN::utils::bbox::IntersectionOverUnion2d(bb1, bb2) > 0.1) {
                    const auto &m1 = h1.cache().at_frame(i1.timestamp_).mask();
                    const auto &m2 = h2.cache().at_frame(i2.timestamp_).mask();
                    const auto &p1_inds = m1.GetIndices();
                    const auto &p2_inds = m2.GetIndices();

                    auto area_p1 = static_cast<double>(p1_inds.size());
                    auto area_p2 = static_cast<double>(p2_inds.size());

                    // Warning: masks are decompressed. Should be computed using compressed representation.
                    // Compute index intersection
                    std::vector<int> set_intersection(std::max(p1_inds.size(), p2_inds.size()));
                    auto it_intersect = std::set_intersection(p1_inds.begin(), p1_inds.end(), p2_inds.begin(),
                                                              p2_inds.end(), set_intersection.begin());
                    set_intersection.resize(it_intersect - set_intersection.begin());
                    auto area_intersection = static_cast<double>(set_intersection.size());
                    overlap = area_intersection / std::max(1.0, std::min(area_p1, area_p2));
                }

                return overlap;
            };

            auto f_overlap_mask_iou = [](const GOT::tracking::Hypothesis &h1, const GOT::tracking::Hypothesis &h2,
                                         const GOT::tracking::HypothesisInlier &i1,
                                         const GOT::tracking::HypothesisInlier &i2) -> double {
                // Compute mask IoU
                double mask_iou = 0.0;
                const Eigen::Vector4d &bb1 = h1.cache().at_frame(i1.timestamp_).box2();
                const Eigen::Vector4d &bb2 = h2.cache().at_frame(i2.timestamp_).box2();
                if (SUN::utils::bbox::IntersectionOverUnion2d(bb1, bb2) > 0.1) {
                    const auto &m1 = h1.cache().at_frame(i1.timestamp_).mask();
                    const auto &m2 = h2.cache().at_frame(i2.timestamp_).mask();
                    mask_iou = m1.IoU(m2);
                }

                return mask_iou;
            };

            double PairwiseFncIoM(const GOT::tracking::Hypothesis &h1, const GOT::tracking::Hypothesis &h2, int frame,
                                  const boost::program_options::variables_map &var_map) {

                // Sanity checks
                if ((var_map.count("inf_tau") + var_map.count("inf_window_size")) != 2) {
                    throw std::runtime_error("PairwiseFncIoM error, one of the params not specified!");
                }

                auto tau = var_map.at("inf_tau").as<int>();
                auto window_size = var_map.at("inf_window_size").as<int>();

                return GOT::tracking::CAMOT_tracker::ComputePhysicalOverlap(frame, tau, window_size, h1, h2,
                                                                            f_overlap_mask_iom);
            }

            double PairwiseFncIoU(const GOT::tracking::Hypothesis &h1, const GOT::tracking::Hypothesis &h2, int frame,
                                  const boost::program_options::variables_map &var_map) {

                // Sanity checks
                if ((var_map.count("inf_tau") + var_map.count("inf_window_size")) != 2) {
                    throw std::runtime_error("PairwiseFncIoM_IoU error, one of the params not specified!");
                }

                auto tau = var_map.at("inf_tau").as<int>();
                auto window_size = var_map.at("inf_window_size").as<int>();
                auto overlap_iou = GOT::tracking::CAMOT_tracker::ComputePhysicalOverlap(frame, tau, window_size, h1, h2,
                                                                                        f_overlap_mask_iou);
                return overlap_iou;
            }

            std::vector<int> InferStateForFrame(int frame,
                                                const std::vector<GOT::tracking::Hypothesis> &hypos,
                                                f_unary unary_fnc,
                                                f_pairwise pairwise_fnc,
                                                const boost::program_options::variables_map &var_map) {

                // Sanity checks
                if ((var_map.count("inf_e1") + var_map.count("inf_e2") + var_map.count("inf_window_size")) != 3) {
                    throw std::runtime_error("CAMOT_tracker::InferStateForFrame error, "
                                                     "one of the params not specified!");
                }

                auto e1 = var_map.at("inf_e1").as<double>();
                auto e2 = var_map.at("inf_e2").as<double>();
                auto window_size = var_map.at("inf_window_size").as<int>();

                auto unary_handle = std::bind(unary_fnc, std::placeholders::_1, std::placeholders::_2, var_map);
                auto pairwise_handle = std::bind(pairwise_fnc, std::placeholders::_1, std::placeholders::_2,
                                                 std::placeholders::_3, var_map);

                // Determine window bounds
                const int half_window = window_size / 2;
                int window_start_frame = std::max(0, frame - half_window);
                int window_end_frame = frame + half_window;

                // Get relevant hypos
                std::vector<GOT::tracking::Hypothesis> hypos_window;
                std::vector<int> cached_inds;
                for (int i = 0; i < hypos.size(); i++) {
                    const auto &hypo = hypos.at(i);
                    int hypo_start_t = hypo.cache().timestamps().front();
                    int hypo_end_t = hypo.cache().timestamps().back();
                    if (hypo_start_t < window_end_frame && hypo_end_t > window_start_frame &&
                        hypo.cache().Exists(frame)) {
                        hypos_window.push_back(hypo);
                        cached_inds.push_back(i);
                    }
                }

                const auto num_hypos = hypos_window.size();

                Eigen::MatrixXd Q(num_hypos, num_hypos);
                Q.setZero();

                /// Unaries
                for (int i = 0; i < num_hypos; i++) {
                    const auto &hypo = hypos_window.at(i);
                    double score = unary_handle(hypo, frame);
                    Q(i, i) = -e1 + score;
                }

                /// Pairwise
                for (int i = 0; i < num_hypos; i++) {
                    for (int j = i + 1; j < num_hypos; j++) {
                        const auto &h1 = hypos_window.at(i);
                        const auto &h2 = hypos_window.at(j);

                        double pairwise_pen = e2 * pairwise_handle(h1, h2, frame);
                        Q(i, j) = -0.5 * pairwise_pen;
                        Q(j, i) = Q(i, j);
                    }
                }

                /// Run the solver
                // Result is the binary vector m, indicating 'selected'-1 or 'not selected'-0
                Eigen::VectorXi m;
                GOT::tracking::QPBO::SolveMultiBranch(Q, m);

                /// Map-back selected inds
                std::vector<int> selected_global;
                for (int i = 0; i < m.size(); i++) {
                    bool selected = static_cast<bool>(m[i]);
                    if (selected) {
                        selected_global.push_back(cached_inds.at(i));
                    }
                }

                return selected_global;
            }

        }
    }
}


