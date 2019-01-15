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

#include "utils_bounding_box.h"

// eigen
#include <Eigen/Core>

// utils
#include "utils_filtering.h"
#include "camera.h"
#include "utils_common.h"


namespace SUN {
    namespace utils {
        namespace bbox {

            Eigen::Vector4d Intersection2d(const Eigen::Vector4d &rect1, const Eigen::Vector4d &rect2) {
                const double rect1_x = rect1[0];
                const double rect1_y = rect1[1];
                const double rect1_w = rect1[2];
                const double rect1_h = rect1[3];

                const double rect2_x = rect2[0];
                const double rect2_y = rect2[1];
                const double rect2_w = rect2[2];
                const double rect2_h = rect2[3];

                const double left = rect1_x > rect2_x ? rect1_x : rect2_x;
                const double top = rect1_y > rect2_y ? rect1_y : rect2_y;
                double lhs = rect1_x + rect1_w;
                double rhs = rect2_x + rect2_w;
                const double right = lhs < rhs ? lhs : rhs;
                lhs = rect1_y + rect1_h;
                rhs = rect2_y + rect2_h;
                const double bottom = lhs < rhs ? lhs : rhs;

                Eigen::Vector4d rect_intersection;
                rect_intersection[0] = right < left ? 0 : left;
                rect_intersection[1] = bottom < top ? 0 : top;
                rect_intersection[2] = right < left ? 0 : right - left;
                rect_intersection[3] = bottom < top ? 0 : bottom - top;

                return rect_intersection;
            }

            double IntersectionOverUnion2d(const Eigen::Vector4d &rect1, const Eigen::Vector4d &rect2) {
                Eigen::Vector4d rect_intersection = Intersection2d(rect1, rect2);
                const double intersection_area =
                        rect_intersection[2] * rect_intersection[3]; // Surface of the intersection of the rects
                const double union_of_rects =
                        rect1[2] * rect1[3] + rect2[2] * rect2[3] - intersection_area; // Union of the area of the rects
                return intersection_area / union_of_rects; // Intersection over union
            }

            Eigen::Vector4d BoundingBox2d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                          const std::vector<int> &indices, double percentage) {

                pcl::PointIndices filtered_indices;
                filtered_indices.indices = indices;

                int bbx_min = static_cast<int>(1e10);
                int bbx_max = static_cast<int>(-1e10);
                int bby_min = static_cast<int>(1e10);
                int bby_max = static_cast<int>(-1e10);

                for (auto ind:filtered_indices.indices) {
                    int x = -1, y = -1;
                    UnravelIndex(ind, scene_cloud->width, &x, &y);
                    if (x < bbx_min)
                        bbx_min = x;
                    if (x > bbx_max)
                        bbx_max = x;
                    if (y < bby_min)
                        bby_min = y;
                    if (y > bby_max)
                        bby_max = y;
                }

                // [min_x min_y w h]
                Eigen::Vector4d bb2d_out;
                bb2d_out[0] = static_cast<double>(bbx_min);
                bb2d_out[1] = static_cast<double>(bby_min);
                bb2d_out[2] = static_cast<double>(bbx_max - bbx_min);
                bb2d_out[3] = static_cast<double>(bby_max - bby_min);

                return bb2d_out;
            }

            Eigen::Vector4d EnlargeBoundingBox2d(const Eigen::Vector4d &bounding_box, double scale_x, double scale_y) {
                auto bbox_copy = bounding_box;
                double new_w = bbox_copy[2] * scale_x;
                double new_h = bbox_copy[3] * scale_y;

                double t_x = std::abs((new_w - bbox_copy[2]) / 2.0);
                double t_y = std::abs((new_h - bbox_copy[3]) / 2.0);
                bbox_copy[0] -= t_x;
                bbox_copy[1] -= t_y;

                bbox_copy[0] = std::max(bbox_copy[0], 0.0);
                bbox_copy[1] = std::max(bbox_copy[1], 0.0);

                bbox_copy[2] = new_w;
                bbox_copy[3] = new_h;

                return bbox_copy;
            }

            Eigen::VectorXd BoundingBox3d(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr scene_cloud,
                                          const std::vector<int> &indices, double percentage) {
                // Only keep the 'inner quartile' -- point closes to the median of the points. Makes 3D bbox computation robust.
                pcl::PointIndices filtered_indices;
                filtered_indices.indices = SUN::utils::filter::FilterKeepInnerqQuartile(scene_cloud, indices);

                if (filtered_indices.indices.size() < 5) {
                    std::cout << " ==== 3D-BBOX-ESTIM-PANIC: filtered inds < 5. ====" << std::endl;
                }

                // Compute mean, covariance matrix 3d
                Eigen::Matrix3d cov_mat3d;
                Eigen::Vector4d mean3d;
                pcl::computeMeanAndCovarianceMatrix(*scene_cloud, indices, cov_mat3d, mean3d);

                // Let's restrict ourselves to 2D ground-plane projection. More robust.
                Eigen::Matrix2d cov_mat2d;
                cov_mat2d(0, 0) = cov_mat3d(0, 0);
                cov_mat2d(1, 1) = cov_mat3d(2, 2);
                cov_mat2d(0, 1) = cov_mat2d(1, 0) = cov_mat3d(0, 2);

                // Compute Eigen vectors, values..
                // Here, we get 2 eigenvectors, corresponding to dominant axes of 2D proj. of object (to the ground plane).
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(cov_mat2d, Eigen::ComputeEigenvectors);
                Eigen::Matrix2d eigen_vectors = eigen_solver.eigenvectors();

                // Def. local coord. sys.
                Eigen::Matrix3d p2w(Eigen::Matrix3d::Identity());
                p2w.block<2, 2>(0, 0) = eigen_vectors.transpose();
                Eigen::Vector2d c_2d(mean3d[0], mean3d[2]); // Mean on gp. proj.
                p2w.block<2, 1>(0, 2) = -1.f * (p2w.block<2, 2>(0, 0) * c_2d); //centroid.head<2>());

                // Find bbox extent (width, depth)
                float bbx_min = 1e10;
                float bbx_max = -1e10;
                float bbz_min = 1e10;
                float bbz_max = -1e10;

                for (auto ind : filtered_indices.indices) {
                    const auto &p_ref = scene_cloud->at(ind);

                    if (std::isnan(p_ref.x))
                        continue;

                    Eigen::Vector3d p_eig(p_ref.x, p_ref.z, 1.0);
                    p_eig = p2w * p_eig;

                    const double tmp_x = p_eig[0];
                    const double tmp_z = p_eig[1];

                    if (tmp_x < bbx_min)
                        bbx_min = tmp_x;
                    if (tmp_x > bbx_max)
                        bbx_max = tmp_x;
                    if (tmp_z < bbz_min)
                        bbz_min = tmp_z;
                    if (tmp_z > bbz_max)
                        bbz_max = tmp_z;
                }

                // Find out orientation
                Eigen::Matrix3d R_mat;
                R_mat.setIdentity();
                R_mat(0, 0) = eigen_vectors(0, 0);
                R_mat(0, 2) = eigen_vectors(0, 1);
                R_mat(2, 0) = eigen_vectors(1, 0);
                R_mat(2, 2) = eigen_vectors(1, 1);
                const Eigen::Vector2d mean_diag = 0.5f * (Eigen::Vector2d(bbx_min + bbx_max, bbz_min + bbz_max));
                const Eigen::Vector2d tfinal = eigen_vectors * mean_diag + Eigen::Vector2d(mean3d[0], mean3d[2]);

                // Final transform
                const Eigen::Quaterniond qfinal(R_mat);
                double bb1 = bbx_max - bbx_min;
                double bb3 = bbz_max - bbz_min;

                // Get height
                Eigen::Vector4f cloud_min, cloud_max;
                pcl::getMinMax3D(*scene_cloud, filtered_indices, cloud_min, cloud_max);
                double min_y = cloud_min[1];
                double max_y = cloud_max[1];
                double cloud_height = std::abs(max_y - min_y); // Difference between Y-coords.

                // Resulting data structure
                Eigen::VectorXd bb3d_out;
                bb3d_out.setZero(10, 1); // center_x, center_y, center_z, width, height, depth, quaternion

                // X-Z center from 2d-gp-PCA, Y compute from 3D points
                bb3d_out(0) = tfinal[0];
                bb3d_out(1) = min_y + (max_y - min_y) / 2.0;
                bb3d_out(2) = tfinal[1];

                // Width, height, depth
                bb3d_out(3) = std::abs(bb1);
                bb3d_out(4) = std::abs(cloud_height);
                bb3d_out(5) = std::abs(bb3);

                // Quaternion, representing orientation
                bb3d_out(6) = qfinal.w();
                bb3d_out(7) = qfinal.x();
                bb3d_out(8) = qfinal.y();
                bb3d_out(9) = qfinal.z();

                return bb3d_out;
            }

            //! Test if point is inside or outside AABB
            // WARN: Fromat [cx cy cz w h l angle_y], hence, angle is specified as rot. around Y-axis, and not quaternion!!!
            bool IsPointInOBB3d(const double x, const double y, const double z,
                                const Eigen::VectorXd &bounding_box_3d, double angle_Y) {
                // Get: x-y-z min and max
                const double w_by_2 = bounding_box_3d[3] / 2.0;
                const double h_by_2 = bounding_box_3d[4] / 2.0;
                const double l_by_2 = bounding_box_3d[5] / 2.0;
                const double cx = bounding_box_3d[0];
                const double cy = bounding_box_3d[1];
                const double cz = bounding_box_3d[2];
                //const double angle_Y = bounding_box_3d[6];

                // Y-axis angle to quaternion
                Eigen::Quaterniond q = static_cast<Eigen::Quaterniond>(Eigen::AngleAxisd(angle_Y + M_PI_2,
                                                                                         Eigen::Vector3d::UnitY()));

                // Assemble pose matrix which needs to be transformed
                Eigen::Matrix<double, 4, 4> label_pose = Eigen::MatrixXd::Identity(4, 4);
                label_pose(0, 3) = cx;
                label_pose(1, 3) = cy;//- h_by_2;
                label_pose(2, 3) = cz;
                label_pose.block<3, 3>(0, 0) = q.toRotationMatrix();

                Eigen::Matrix4d pose_inv;
                InvertPose(label_pose, pose_inv);

                Eigen::Vector4d ref_point(x, y, z, 1);
                ref_point = pose_inv * ref_point;

                const double x_t = ref_point[0];
                const double y_t = ref_point[1];
                const double z_t = ref_point[2];

                if (x_t < w_by_2 && x_t > -w_by_2 && // x-axes
                    y_t < h_by_2 && y_t > -h_by_2 && // y-axes
                    z_t < l_by_2 && z_t > -l_by_2)  // z-axes
                {
                    return true;
                }

                return false;
            }
        }
    }
}