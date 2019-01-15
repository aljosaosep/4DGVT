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

#include <tracking/hypothesis.h>

// pcl
#include <pcl/common/transforms.h>

// utils
#include "sun_utils/camera.h"
#include "sun_utils/shared_types.h"

namespace GOT {
    namespace tracking {

        std::vector<Eigen::Vector4d> HypoCacheToPoses(const HypoCache<HypoData> &cache, int frame_limit) {
            std::vector<Eigen::Vector4d> poses;
            poses.reserve(cache.size());
            for (auto t : cache.timestamps()) {
                if (frame_limit > -1 && t > frame_limit) {
                    break;
                }
                poses.push_back(cache.at_frame(t).pose());
            }

            return poses;
        }

        void ClearCachedPredictions(HypoCache<HypoData> &cache) {
            for (auto t : cache.timestamps()) {
                auto &ref_el = cache.at_frame(t);
                ref_el.predicted_segment().clear();
                ref_el.predicted_mask().rle_string_.clear();
                ref_el.predicted_mask().w_ = ref_el.predicted_mask().h_ = 0;
            }
        }

        Hypothesis::Hypothesis() {
            this->id_ = -1;
            kalman_filter_ = nullptr;
            shape_model_ = nullptr;
            last_frame_selected_ = -1;
            creation_timestamp_ = -1;
        }

        void Hypothesis::AddInlier(const HypothesisInlier &inlier) {
            this->inliers_.push_back(inlier);
        }

        void Hypothesis::AddEntry(const Eigen::Vector4d &position, int frame_of_detection) {
            this->cache_v2_.Update(frame_of_detection).pose() = position; //  , position, cache_.poses());
        }

        // Setters / Getters
        int Hypothesis::id() const {
            return this->id_;
        }

        void Hypothesis::set_id(int id) {
            this->id_ = id;
        }

        TerminationInfo Hypothesis::terminated() const {
            return this->termination_info_;
        }

        bool Hypothesis::IsHypoTerminatedInFrame(int frame) const {
            return (this->termination_info_.IsTerminated()) && (this->termination_info_.FrameTerminated() <= frame);
        }

        void Hypothesis::set_terminated(const TerminationInfo &terminated) {
            this->termination_info_ = terminated;
        }

        const std::vector<HypothesisInlier> &Hypothesis::inliers() const {
            return inliers_;
        }

        void Hypothesis::set_inliers(const std::vector<HypothesisInlier>& inliers) {
            inliers_ = inliers;
        }

        ShapeModel::Ptr& Hypothesis::shape_model() {
            return shape_model_;
        }

        ShapeModel::ConstPtr Hypothesis::shape_model_const() const{
            return shape_model_;
        }

        ExtendedKalmanFilter::Ptr& Hypothesis::kalman_filter() {
            return this->kalman_filter_;
        }

        ExtendedKalmanFilter::ConstPtr Hypothesis::kalman_filter_const() const {
            return this->kalman_filter_;
        }

        const double Hypothesis::score() const {
            return score_;
        }

        void Hypothesis::set_score(double score) {
            score_ = score;
        }

        const Eigen::VectorXd& Hypothesis::color_histogram() const {
            return color_histogram_;
        }

        void Hypothesis::set_color_histogram(const Eigen::VectorXd color_histogram) {
            color_histogram_ = color_histogram;
        }

        int Hypothesis::last_frame_selected() const {
            return last_frame_selected_;
        }

        void Hypothesis::set_last_frame_selected(int last_frame_selected) {
            last_frame_selected_ = last_frame_selected;
        }

        int Hypothesis::creation_timestamp() const {
            return this->creation_timestamp_;
        }

        void Hypothesis::set_creation_timestamp(int timestamp) {
            this->creation_timestamp_ = timestamp;
        }

        const std::vector<float> &Hypothesis::category_probability_distribution() const {
            return category_probability_distribution_;
        }

        std::vector<float> &Hypothesis::category_probability_distribution() {
            return category_probability_distribution_;
        }

        const HypoCache<HypoData> &Hypothesis::cache() const {
            return this->cache_v2_;
        }

        HypoCache<HypoData> &Hypothesis::cache() {
            return this->cache_v2_;
        }
    }
}
