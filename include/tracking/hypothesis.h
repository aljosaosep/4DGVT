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

#ifndef GOT_HYPOTHESIS_H
#define GOT_HYPOTHESIS_H

// std
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

// Eigen
#include <Eigen/Core>

// Tracking
#include <tracking/data_queue.h>
#include <tracking/shape_model.h>
#include <tracking/extended_kalman_filter.h>

// Forward declarations
namespace GOT { namespace tracking { class HypothesisInlier; }}
namespace SUN { namespace utils { class Camera; }}
namespace SUN { namespace shared_types { enum CategoryTypeKITTI; }}

namespace GOT {
    namespace tracking {

        class HypoData {
        private:
            // Data to cache foreach hypo
            Eigen::VectorXd pose_;
            Eigen::VectorXd pose_cam_;
            Eigen::VectorXd box2_;
            Eigen::VectorXd box3_;
            Eigen::VectorXd box_ext_;
            SUN::shared_types::CompressedMask mask_;
            SUN::shared_types::CompressedMask predicted_mask_;
            pcl::PointCloud<pcl::PointXYZRGBA> predicted_segment_;
            int segm_id_ = -1;

        public:
            // Const Ref
            const int &segm_id() const { return segm_id_; }

            const Eigen::VectorXd &pose() const { return pose_; }

            const Eigen::VectorXd &pose_cam() const { return pose_cam_; }

            const Eigen::VectorXd &box2() const { return box2_; }

            const Eigen::VectorXd &box3() const { return box3_; }

            const Eigen::VectorXd &box_ext() const { return box_ext_; }

            const SUN::shared_types::CompressedMask &mask() const { return mask_; }

            const SUN::shared_types::CompressedMask &predicted_mask() const { return predicted_mask_; }

            const pcl::PointCloud<pcl::PointXYZRGBA> &predicted_segment() const { return predicted_segment_; }

            // Ref
            int &segm_id() { return segm_id_; }

            Eigen::VectorXd &pose() { return pose_; }

            Eigen::VectorXd &pose_cam() { return pose_cam_; }

            Eigen::VectorXd &box2() { return box2_; }

            Eigen::VectorXd &box3() { return box3_; }

            Eigen::VectorXd &box_ext() { return box_ext_; }

            SUN::shared_types::CompressedMask &mask() { return mask_; }

            SUN::shared_types::CompressedMask &predicted_mask() { return predicted_mask_; }

            pcl::PointCloud<pcl::PointXYZRGBA> &predicted_segment() { return predicted_segment_; }

            // Got
            bool got_segm_id() const { return segm_id_ != -1; }

            bool got_pose() const { return pose_.size() > 0; }

            bool got_pose_cam() const { return pose_.size() > 0; }

            bool got_box2() const { return pose_.size() > 0; }

            bool got_box3() const { return pose_.size() > 0; }

            bool got_box_ext() const { return pose_.size() > 0; }

            bool got_mask() const { return mask_.rle_string_.size() > 0; }

            bool got_predicted_mask() const { return predicted_mask_.rle_string_.size() > 0; }

            bool got_predicted_segment() const { return predicted_segment_.size() > 0; }
        };

        template<typename T>
        class HypoCache {
        private:
            int start_frame_ = -1;
            int curr_frame_ = -1;

            std::unordered_map<int, T> hash_map_;
            std::vector<int> timestamps_; // Need to store this for fast access (TODO: dowe? Can't use hash instead?)

            void ResetTimestamps() {
                start_frame_ = -1;
                curr_frame_ = -1;
                timestamps_.clear();
            }

        public:
            // TODO:
            // Keep in mind that we can insert data in fwd-frame or backward!

            const std::vector<int> &timestamps() const { return timestamps_; }

            const size_t size() const { return timestamps_.size(); };

            // -----------------------------------
            //    +++  Reset +++
            // -----------------------------------
            void Reset() {
                ResetTimestamps();
                hash_map_.clear(); // Easy
            }

            // -----------------------------------
            //    +++  Getters +++
            // -----------------------------------
            int start_frame() const {
                return this->start_frame_;
            }

            int curr_frame() const {
                return this->curr_frame_;
            }

            // -----------------------------------
            //    +++  AddElement +++
            // -----------------------------------
            void Add(int frame, const T &el) {

                if (frame < 0) {
                    throw std::runtime_error("HypoCache error, frame < 0!");
                }

                // Is this first-frame insertion?
                if (start_frame_ == -1) {
                    start_frame_ = frame;
                }

                if (curr_frame_ > -1) {
                    if (std::abs(curr_frame_ - frame) > 1) {
                        throw std::runtime_error("Error d(curr_frame_, frame) > 1.");
                    }
                }

                curr_frame_ = frame;

                hash_map_.insert({frame, el});
                timestamps_.push_back(frame);

                // Sanity check
                int frame_dist = std::abs(curr_frame_ - start_frame_);
                if (timestamps_.size() != (frame_dist + 1) || hash_map_.size() != (frame_dist + 1)) {
                    throw std::runtime_error(
                            "HypoCache::Add error, timestamps_.size() != frame_dist || hash_map_.size() != frame_dist");
                }
            }


            // Either get ref to existing HypoData, or insert new entry in case we do not have one yet.
            T &Update(int frame) {

                if (frame < 0) {
                    throw std::runtime_error("HypoCache::Update() error, frame < 0!");
                }

                if (!Exists(frame)) {
                    // TODO: safety checks
                    Add(frame, T());
                }

                return hash_map_.at(frame);
            }

            // -----------------------------------
            //    +++  Exists +++
            // -----------------------------------
            bool Exists(int frame) const {

                if (frame < 0) {
                    throw std::runtime_error("HypoCache::Exists() error, frame < 0!");
                }

                return static_cast<bool>(hash_map_.count(frame));
            }

            // -----------------------------------
            //    +++  At_frame +++
            // -----------------------------------
            T &at_frame(int frame) {

                if (frame < 0) {
                    throw std::runtime_error("HypoCache::at_frame(int) error, frame < 0!");
                }

                if (!Exists(frame)) {
                    throw std::runtime_error("HypoCache::at_frame(int) error, frame not in hash_map!");
                }

                return hash_map_.at(frame);
            }

            const T &at_frame(int frame) const {

                if (frame < 0) {
                    throw std::runtime_error("HypoCache::at_frame(int) error, frame < 0!");
                }

                if (!Exists(frame)) {
                    throw std::runtime_error("HypoCache::at_frame(int) error, frame not in hash_map!");
                }

                return hash_map_.at(frame);
            }

            // -----------------------------------
            //    +++  Frame_predcessor +++
            // -----------------------------------
            T &predecessor_frame(int frame) {

                if (frame <= 0) {
                    throw std::runtime_error("HypoCache::predecessor_frame(int) error, no predecessor for frame 0!");
                }

                int predecessor_frame = -1;
                if (curr_frame_ > start_frame_) {
                    predecessor_frame = curr_frame_ - 1; // Fwd time
                } else {
                    predecessor_frame = curr_frame_ + 1; // Backward time
                }

                return at_frame(predecessor_frame);
            }

            const T &predecessor_frame(int frame) const {

                if (frame <= 0) {
                    throw std::runtime_error("HypoCache::predecessor_frame(int) error, no predecessor for frame 0!");
                }

                int predecessor_frame = -1;
                if (curr_frame_ > start_frame_) {
                    predecessor_frame = curr_frame_ - 1; // Fwd time
                } else {
                    predecessor_frame = curr_frame_ + 1; // Backward time
                }

                return at_frame(predecessor_frame);
            }

            // -----------------------------------
            //    +++  At_idx +++
            // -----------------------------------
            T &at_idx(int idx) {

                if (idx < 0) {
                    throw std::runtime_error("HypoCache::at(int) error, idx < 0!");
                }

                auto frame_at_idx = timestamps().at(idx);
                return at_frame(frame_at_idx);
            }

            const T &at_idx(int idx) const {

                if (idx < 0) {
                    throw std::runtime_error("HypoCache::at(int) error, idx < 0!");
                }

                auto frame_at_idx = timestamps().at(idx);
                return at_frame(frame_at_idx);
            }

            // -----------------------------------
            //    +++  Back +++
            // -----------------------------------
            T &back() {

                if (this->size() <= 0) {
                    throw std::runtime_error("HypoCache::back() error, size()==0!");
                }

                if (!Exists(curr_frame_)) {
                    throw std::runtime_error("HypoCache::back() error, no entry for curr_frame in the hash_map!");
                }

                return hash_map_.at(curr_frame_);
            }

            const T &back() const {

                if (this->size() <= 0) {
                    throw std::runtime_error("HypoCache::back() error, size()==0!");
                }

                if (!Exists(curr_frame_)) {
                    throw std::runtime_error("HypoCache::back() error, no entry for curr_frame in the hash_map!");
                }

                return hash_map_.at(curr_frame_);
            }

            // -----------------------------------
            //    +++  Front +++
            // -----------------------------------

            T &front() {

                if (this->size() <= 0) {
                    throw std::runtime_error("HypoCache::front() error, size()==0!");
                }

                if (!Exists(start_frame_)) {
                    throw std::runtime_error("HypoCache::front() error, no entry for start_frame in the hash_map!");
                }

                return hash_map_.at(start_frame_);
            }

            const T &front() const {

                if (this->size() <= 0) {
                    throw std::runtime_error("HypoCache::front() error, size()==0!");
                }

                if (!Exists(start_frame_)) {
                    throw std::runtime_error("HypoCache::front() error, no entry for start_frame in the hash_map!");
                }

                return hash_map_.at(start_frame_);
            }

        };

        std::vector<Eigen::Vector4d> HypoCacheToPoses(const HypoCache<HypoData> &cache, int frame_limit = -1);

        void ClearCachedPredictions(HypoCache<HypoData> &cache);

        struct HypothesisInlier {
            int timestamp_; // Timestamp of the inlier
            int index_; // Index of the inlier
            double association_score_; // Inlier-to-hypo 'fitness' score
            double inlier_score_; // Score of the inlier (eg. detection score, proposal score, ...)

            std::vector<double> assoc_data_; // You might wanna store additional assoc. info here for dbg. purposes

            HypothesisInlier(int timestamp, int index, double inlier_score, double association_score) {
                AddInlier(timestamp, index, inlier_score, association_score);
            }

            void AddInlier(int timestamp, int index, double inlier_score, double association_score) {
                this->timestamp_ = timestamp;
                this->index_ = index;
                this->inlier_score_ = inlier_score;
                this->association_score_ = association_score;
            }
        };

        struct TerminationInfo {
            bool is_terminated_;
            unsigned t_terminated_;

            TerminationInfo() {
                is_terminated_ = false;
                t_terminated_ = 0;
            }

            TerminationInfo(bool is_terminated, unsigned t_terminated) {
                is_terminated_ = is_terminated;
                t_terminated_ = t_terminated;
            }

            bool IsTerminated() const {
                return is_terminated_;
            }

            unsigned FrameTerminated() const {
                return t_terminated_;
            }
        };

        /**
           * @brief Represents a basic tracker unit.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class Hypothesis {
        public:

            /**
             * @brief Constructor initializes the hypothesis.
             */
            Hypothesis();

            /**
             * @brief Adds hypo pose+timestamp pair.
             * @param frame_of_detection Frame, in which the detection was identified.

             */
            void AddEntry(const Eigen::Vector4d &position, int frame_of_detection);

            /**
             * @brief Adds inlier info to the hypothesis. Inlier is uniquely determined by frame number and detection table index.
             * @note inlier_entry_index is index of inlier in the Resources structure (not inlier_id!)
             * @param inlier Is a HypothesisInlier instance.
             */
            void AddInlier(const HypothesisInlier &inlier);

            // -------------------------------------------
            // Cached Data
            // -------------------------------------------
            const HypoCache<HypoData> &cache() const;

            HypoCache<HypoData> &cache();


            // -------------------------------------------
            // Etc
            // -------------------------------------------
            // Getters
            int id() const;

            void set_id(int id);

            TerminationInfo terminated() const;

            const double score() const;

            const Eigen::VectorXd &color_histogram() const;

            int last_frame_selected() const;

            int creation_timestamp() const;

            ShapeModel::Ptr &shape_model();

            ShapeModel::ConstPtr shape_model_const() const;

            ExtendedKalmanFilter::Ptr &kalman_filter();

            ExtendedKalmanFilter::ConstPtr kalman_filter_const() const;

            bool IsHypoTerminatedInFrame(int frame) const;

            const std::vector<float> &category_probability_distribution() const;

            std::vector<float> &category_probability_distribution();

            // Setters
            const std::vector<HypothesisInlier> &inliers() const;

            void set_inliers(const std::vector<HypothesisInlier> &inliers);

            void set_terminated(const TerminationInfo &terminated);

            void set_score(double score);

            void set_color_histogram(const Eigen::VectorXd color_histogram);

            void set_last_frame_selected(int last_frame_selected);

            void set_creation_timestamp(int timestamp);
            // -------------------------------------------

        protected:
            HypoCache<HypoData> cache_v2_;
            std::vector<HypothesisInlier> inliers_;
            ShapeModel::Ptr shape_model_;
            ExtendedKalmanFilter::Ptr kalman_filter_;
            Eigen::VectorXd color_histogram_;
            std::vector<float> category_probability_distribution_;
            TerminationInfo termination_info_;
            double score_;
            int last_frame_selected_;
            int id_;
            int creation_timestamp_;
        };

        typedef std::vector<Hypothesis> HypothesesVector;
    }
}


#endif
