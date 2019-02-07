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

#ifndef GOT_OBJECT_PROPOSAL_H
#define GOT_OBJECT_PROPOSAL_H

#include "shared_types.h"

#include <vector>

#include <Eigen/Dense>
#include <Eigen/StdVector>

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>


namespace GOT {
    namespace segmentation {

        /**
           * @brief This class represents one object proposal.
           *        Object proposal is 3D region, mostly defined by set of 3d points.
           *        These 3d points are stored here as
           *        (linear) indices, corresponding to 'cells' of (organized) pcl::PointCloud.
           *        Thus, indices relate the region to both, 2d image plane and a set of 3d points.
           *        2D image indices can be obtained using GOT::utils::geometry::ConvertIndex2Coordinates(...)
           *
           *        For convenience, we also store 2d and 3d bounding box, 3d pos. (in local coord. frame of point cloud)
           *        and id of the proposal.
           */
        class ObjectProposal {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            typedef std::vector<ObjectProposal, Eigen::aligned_allocator<ObjectProposal> > Vector;

        public:
            void init(const std::vector<int> &groundplane_indices,
                      const std::vector<int> &pointcloud_indices,
                      const Eigen::Vector4d &bounding_box_2d,
                      const Eigen::VectorXd &bounding_box_3d,
                      double score,
                      int image_width,
                      int image_height);

            // Setters / getters
            Eigen::Vector4d pos3d() const;

            double score() const;

            const Eigen::Vector4d &bounding_box_2d() const;

            const Eigen::VectorXd &bounding_box_3d() const;

            const std::vector<int> &pointcloud_indices() const;

            const std::vector<int> &ground_plane_indices() const;

            const SUN::shared_types::CompressedMask &compressed_mask() const;

            const std::vector<std::pair<float, float> > &scale_pairs() const;

            const Eigen::Matrix3d &pose_covariance_matrix() const;

            const int category() const;

            const int second_category() const;

            const std::vector<float> &posterior() const; // Agnostic
            const std::vector<float> &second_posterior() const; // Specific

            void set_bounding_box_2d(const Eigen::Vector4d &bbox2d);

            void set_bounding_box_3d(const Eigen::VectorXd &bbox3d);

            void set_pos3d(const Eigen::Vector4d &pos3d);

            void set_score(double new_score);

            void set_pointcloud_indices(const std::vector<int> &indices, int image_width, int image_height);

            void set_pointcloud_indices(const std::string &mask_rle, int image_width, int image_height);

            void set_groundplane_indices(const std::vector<int> &indices);

            void set_pose_covariance_matrix(const Eigen::Matrix3d &pose_cov_mat);

            void set_category(int category_index);

            void set_second_category(int category_index);

            void set_posterior(const std::vector<float> &p); // Agnostic
            void set_second_posterior(const std::vector<float> &p); // Specific

            void add_scale_pair(const std::pair<float, float> &scale_pair);

            void add_scale_pairs(const std::vector<std::pair<float, float> > &scale_pairs);

            void set_segm_id(int id);

            int segm_id() const;

            void free();

        private:
            std::vector<int> groundplane_indices_;

            Eigen::Vector4d pos3d_;
            double score_;

            int category_; // Agnostic
            int second_category_; // Specific

            Eigen::Vector4d bounding_box_2d_; // 2D bounding box: [min_x min_y width height]
            Eigen::VectorXd bounding_box_3d_; // 3D bounding box: [centroid_x centroid_y centroid_z width height length q.w q.x q.y q.z]

            Eigen::Matrix3d pose_covariance_matrix_;

            // Scales info
            std::vector<std::pair<float, float> > scale_pairs_;
            //std::vector<std::pair<int, int>> convex_hull_;

            std::vector<int> cached_indices_;
            SUN::shared_types::CompressedMask compressed_mask_;

            // NEW
            std::vector<float> posterior_; // Agnostic
            std::vector<float> second_posterior_; // Specific

            int segm_id_;
        };
    }
}

#endif
