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

#include <scene_segmentation/object_proposal.h>

// Project utils
#include "utils_pointcloud.h"
#include "utils_bounding_box.h"
#include "shared_types.h"

namespace GOT {
    namespace segmentation {
        void ObjectProposal::init(const std::vector<int> &groundplane_indices,
                                  const std::vector<int> &pointcloud_indices,
                                  const Eigen::Vector4d &bounding_box_2d,
                                  const Eigen::VectorXd &bounding_box_3d,
                                  double score,
                                  int image_width,
                                  int image_height) {
            this->score_ = score;
            this->set_pointcloud_indices(pointcloud_indices, image_width, image_height);
            this->category_ = -1;
            this->bounding_box_2d_ = bounding_box_2d;
            this->bounding_box_3d_ = bounding_box_3d;
            this->groundplane_indices_ = groundplane_indices;
            this->pos3d_ = Eigen::Vector4d(bounding_box_3d[0], bounding_box_3d[1], bounding_box_3d[2], 1.0);
        }

        Eigen::Vector4d ObjectProposal::pos3d() const {
            return pos3d_;
        }

        double ObjectProposal::score() const {
            return score_;
        }

        const Eigen::Vector4d &ObjectProposal::bounding_box_2d() const {
            return this->bounding_box_2d_;
        }

        const Eigen::VectorXd &ObjectProposal::bounding_box_3d() const {
            return this->bounding_box_3d_;
        }

        const std::vector<int> &ObjectProposal::pointcloud_indices() const {
            return cached_indices_;
        }

        void ObjectProposal::set_score(double new_score) {
            this->score_ = new_score;
        }

        void ObjectProposal::set_bounding_box_2d(const Eigen::Vector4d &bbox2d) {
            this->bounding_box_2d_ = bbox2d;
        }

        void ObjectProposal::set_pos3d(const Eigen::Vector4d &pos3d) {
            this->pos3d_ = pos3d;
        }

        void ObjectProposal::set_bounding_box_3d(const Eigen::VectorXd &bbox3d) {
            this->bounding_box_3d_ = bbox3d;
        }

        void
        ObjectProposal::set_pointcloud_indices(const std::vector<int> &indices, int image_width, int image_height) {
            compressed_mask_ = SUN::shared_types::CompressedMask(indices, image_width, image_height);
            cached_indices_ = indices;
        }

        void ObjectProposal::set_pointcloud_indices(const std::string &mask_rle, int image_width, int image_height) {
            compressed_mask_ = SUN::shared_types::CompressedMask(mask_rle, image_width,
                                                                 image_height);
            cached_indices_ = compressed_mask_.GetIndices();
        }

        void ObjectProposal::add_scale_pair(const std::pair<float, float> &scale_pair) {
            this->scale_pairs_.push_back(scale_pair);
        }

        void ObjectProposal::add_scale_pairs(const std::vector<std::pair<float, float> > &scale_pairs) {
            this->scale_pairs_.insert(scale_pairs_.begin(), scale_pairs.begin(), scale_pairs.end());
        }

        const std::vector<std::pair<float, float> > &ObjectProposal::scale_pairs() const {
            return scale_pairs_;
        }

        const std::vector<int> &ObjectProposal::ground_plane_indices() const {
            return this->groundplane_indices_;
        }

        void ObjectProposal::set_groundplane_indices(const std::vector<int> &indices) {
            groundplane_indices_ = indices;
        }

        void ObjectProposal::set_pose_covariance_matrix(const Eigen::Matrix3d &pose_cov_mat) {
            this->pose_covariance_matrix_ = pose_cov_mat;
        }

        const Eigen::Matrix3d &ObjectProposal::pose_covariance_matrix() const {
            return pose_covariance_matrix_;
        }

        const SUN::shared_types::CompressedMask &ObjectProposal::compressed_mask() const {
            return this->compressed_mask_;
        }

        void ObjectProposal::free() {
            this->cached_indices_.clear();
            this->groundplane_indices_.clear();
        }

        void ObjectProposal::set_category(const int category_index) {
            this->category_ = category_index;
        }

        const int ObjectProposal::category() const {
            return this->category_;
        }

        const int ObjectProposal::second_category() const {
            return second_category_;
        }

        const std::vector<float> &ObjectProposal::posterior() const {
            return posterior_;
        }

        const std::vector<float> &ObjectProposal::second_posterior() const {
            return second_posterior_;
        }

        void ObjectProposal::set_second_category(int category_index) {
            second_category_ = category_index;
        }

        void ObjectProposal::set_posterior(const std::vector<float> &p) {
            posterior_ = p;
        }

        void ObjectProposal::set_second_posterior(const std::vector<float> &p) {
            second_posterior_ = p;
        }

        void ObjectProposal::set_segm_id(int id) {
            segm_id_ = id;
        }

        int ObjectProposal::segm_id() const {
            return segm_id_;
        }
    }
}
