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

#include "QPBO_fnc.h"

// tracking
#include <tracking/hypothesis.h>

// utils
#include "sun_utils/utils_bounding_box.h"

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {
            bool DuplicateTest(const Hypothesis &hypothesis_1, const Hypothesis &hypothesis_2, double thresh_IoU) {
                const std::vector<HypothesisInlier> &hypo_inliers_1 = hypothesis_1.inliers();
                const std::vector<HypothesisInlier> &hypo_inliers_2 = hypothesis_2.inliers();

                if (hypo_inliers_1.size() != hypo_inliers_2.size())
                    return false;

                for (int i = 0; i < hypo_inliers_1.size(); i++) {
                    const auto i_1 = hypo_inliers_1.at(i);
                    const auto i_2 = hypo_inliers_2.at(i);

                    if (i_1.timestamp_ != i_2.timestamp_)
                        return false;

                    const auto &bb1 = hypothesis_1.cache().at_frame(i_1.timestamp_).box2();
                    const auto &bb2 = hypothesis_2.cache().at_frame(i_2.timestamp_).box2();

                    const auto &mask_1 = hypothesis_1.cache().at_frame(i_1.timestamp_).mask();
                    const auto &mask_2 = hypothesis_2.cache().at_frame(i_2.timestamp_).mask();

                    double IOU = SUN::utils::bbox::IntersectionOverUnion2d(bb1, bb2);

                    if (IOU < thresh_IoU)
                        return false;
                }

                return true;
            }

            double ExpDecay(int frame, int eval_frame, int tau) {
                return std::exp(-std::fabs(float(frame - eval_frame) / (float) tau));
            }

            double ComputePhysicalOverlap(int current_frame, int tau, int window_size, const Hypothesis &hypothesis_1,
                                          const Hypothesis &hypothesis_2,
                                          OverlapFncSpecInlier overlap_fnc) {
                int window_start_frame = std::max(0, (current_frame - (window_size / 2)));
                int window_end_frame = current_frame + (window_size / 2);

                const std::vector<HypothesisInlier> &hypo_inliers_1 = hypothesis_1.inliers();
                const std::vector<HypothesisInlier> &hypo_inliers_2 = hypothesis_2.inliers();

                int inliers_1_first_stamp = hypo_inliers_1.front().timestamp_;
                int inliers_1_back_stamp = hypo_inliers_1.back().timestamp_;
                int inliers_2_first_stamp = hypo_inliers_2.front().timestamp_;
                int inliers_2_back_stamp = hypo_inliers_2.back().timestamp_;
                int first_timestamp = std::max(inliers_1_first_stamp, inliers_2_first_stamp);
                int last_timestamp = std::min(inliers_1_back_stamp, inliers_2_back_stamp);

                double intersection_sum = 0.0;
                if (hypo_inliers_1.size() > 0 && hypo_inliers_2.size() > 0) {

                    for (const auto &inlier_1:hypo_inliers_1) {
                        const int t_inlier_1 = inlier_1.timestamp_;
                        // -----------------------------------------
                        if (t_inlier_1 < first_timestamp)
                            continue;
                        if (t_inlier_1 > last_timestamp)
                            continue;

                        if (t_inlier_1 < window_start_frame)
                            continue;
                        if (t_inlier_1 > window_end_frame)
                            continue;
                        // -----------------------------------------
                        for (const auto &inlier_2:hypo_inliers_2) {
                            const int t_inlier_2 = inlier_2.timestamp_;

                            // -----------------------------------------
                            if (t_inlier_2 < first_timestamp)
                                continue;
                            if (t_inlier_2 > last_timestamp)
                                continue;

                            if (t_inlier_2 < window_start_frame)
                                continue;

                            if (t_inlier_2 > window_end_frame)
                                continue;

                            // -----------------------------------------

                            if (t_inlier_1 == t_inlier_2) {
                                intersection_sum += overlap_fnc(hypothesis_1, hypothesis_2, inlier_1, inlier_2);
                            }
                        }
                    }

                    return intersection_sum;
                }
                return 0.0; // No intersection whatsoever
            }


            double ComputePhysicalOverlap_IOU_mask(const Hypothesis &hypothesis_1, const Hypothesis &hypothesis_2) {

                const auto &timestamps_1 = hypothesis_1.cache().timestamps();
                const auto &timestamps_2 = hypothesis_2.cache().timestamps();

                int hypo_1_first_stamp = timestamps_1.front();
                int hypo_1_back_stamp = timestamps_1.back();
                int hypo_2_first_stamp = timestamps_2.front();
                int hypo_2_back_stamp = timestamps_2.back();
                int first_timestamp = std::max(hypo_1_first_stamp, hypo_2_first_stamp);
                int last_timestamp = std::min(hypo_1_back_stamp, hypo_2_back_stamp);

                double intersection_sum = 0.0;
                if (timestamps_1.size() > 0 && timestamps_2.size() > 0) {

                    for (const auto &t_inlier_1:timestamps_1) {
                        // -----------------------------------------
                        if (t_inlier_1 < first_timestamp)
                            continue;
                        if (t_inlier_1 > last_timestamp)
                            continue;
                        // -----------------------------------------
                        for (const auto &t_inlier_2:timestamps_2) {

                            // -----------------------------------------
                            if (t_inlier_2 < first_timestamp)
                                continue;
                            if (t_inlier_2 > last_timestamp)
                                continue;
                            // -----------------------------------------

                            if (t_inlier_1 == t_inlier_2) {
                                // -------------------- (2) BOUNDING-BOX-2D -----------------------------------
                                const auto &bb1 = hypothesis_1.cache().at_frame(t_inlier_1).box2();
                                const auto &bb2 = hypothesis_2.cache().at_frame(t_inlier_2).box2();

                                const auto &mask_1 = hypothesis_1.cache().at_frame(t_inlier_1).mask();
                                const auto &mask_2 = hypothesis_2.cache().at_frame(t_inlier_2).mask();

                                double IOU = SUN::utils::bbox::IntersectionOverUnion2d(bb1, bb2);
                                double mask_IOU = 0.0;
                                if (IOU > 0.1) {
                                    mask_IOU = mask_1.IoU(mask_2);
                                }

                                intersection_sum += mask_IOU;
                                // ----------------------------------------------------------------------------
                            }
                        }
                    }
                    return intersection_sum;
                }
                return 0.0; // No intersection whatsoever
            }
        }
    }
}
