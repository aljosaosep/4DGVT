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

#include "utils_observations.h"

// opencv
#include <opencv2/imgproc.hpp>

// utils
#include "utils_common.h"
#include "ground_model.h"


namespace SUN {
    namespace utils {
        namespace observations {

            Eigen::Vector3d ComputeVelocity(const cv::Mat &velocity_map,
                                            const std::vector<int> &indices, double dt, int min_samples) {


                const Eigen::Vector3d NaN_vec = Eigen::Vector3d(std::numeric_limits<double>::quiet_NaN(),
                                                                std::numeric_limits<double>::quiet_NaN(),
                                                                std::numeric_limits<double>::quiet_NaN());


                if (velocity_map.data == nullptr) {
                    return NaN_vec;
                }

                std::vector<Eigen::Vector3d> det_flows;

                for (auto ind:indices) {
                    int x, y;
                    SUN::utils::UnravelIndex(ind, velocity_map.cols, &x, &y);
                    const cv::Vec3f &velocity_meas = velocity_map.at<cv::Vec3f>(y, x);
                    if (!std::isnan(velocity_meas[0])) {
                        Eigen::Vector3d flow_vec_eigen(velocity_meas[0], velocity_meas[1], velocity_meas[2]);
                        flow_vec_eigen /= dt;
                        det_flows.push_back(flow_vec_eigen);
                    }
                }

                if (det_flows.size() < 4) {
                    return NaN_vec;
                }

                std::sort(det_flows.begin(), det_flows.end(), [](const Eigen::Vector3d &e1, const Eigen::Vector3d &e2) {
                    return e1.squaredNorm() < e2.squaredNorm();
                });

                // Compute 'mean' flow from the inner quartile
                const unsigned quartile_size = det_flows.size() / 4;

                Eigen::Vector3d mean_flow;
                mean_flow.setZero();

                int num_samples = 0;
                for (int i = quartile_size; i < det_flows.size() - quartile_size; i++) { // Loop through inner quartile
                    mean_flow += det_flows.at(i);
                    num_samples++;
                }

                if (num_samples < min_samples) {
                    return NaN_vec;
                }

                return mean_flow / static_cast<double>(num_samples);
            }

            bool ComputePoseCovariance(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                       const Eigen::Vector4d &pose_3d,
                                       const std::vector<int> &indices,
                                       const Eigen::Matrix<double, 3, 4> &P_left,
                                       const Eigen::Matrix<double, 3, 4> &P_right,
                                       Eigen::Matrix3d &covariance_matrix_3d, int min_num_points) {

                Eigen::Matrix3d variance_sum;
                variance_sum.setZero();

                int num_pts = 0;
                for (int i = 0; i < indices.size(); i++) {
                    const auto &p = point_cloud->points.at(indices.at(i));
                    if (!std::isnan(p.x)) {
                        Eigen::Vector3d p_eig = p.getVector3fMap().cast<double>();

                        Eigen::Vector2d diff_vec_2D =
                                Eigen::Vector2d(p_eig[0], p_eig[2]) - Eigen::Vector2d(pose_3d[0], pose_3d[2]);
                        if (diff_vec_2D.norm() > 3.0)
                            continue;

                        Eigen::Matrix3d cov3d;
                        SUN::utils::Camera::ComputeMeasurementCovariance3d(p_eig, 0.5, P_left, P_right, cov3d);
                        //cov3d += 0.1*Eigen::Matrix3d::Identity(); // Gaussian prior
                        variance_sum += cov3d + (pose_3d.head<3>() - p_eig) *
                                                (pose_3d.head<3>() - p_eig).transpose(); // Outer product
                        num_pts++;
                    }
                }

                if (num_pts <= min_num_points)
                    return false;

                variance_sum /= (num_pts - 1);
                covariance_matrix_3d = variance_sum;

                return true;
            }

            bool ComputeDetectionPoseUsingStereo(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                 const std::vector<int> &indices, Eigen::Vector4d &median) {

                std::vector<double> pts_det_x;
                std::vector<double> pts_det_z;
                std::vector<double> pts_det_y;

                for (int idx:indices) {
                    int u, v;
                    SUN::utils::UnravelIndex(idx, point_cloud->width, &u, &v);

                    const auto &point_3d = point_cloud->at(u, v); // (col, row)
                    if (!std::isnan(point_3d.x)) {
                        pts_det_x.push_back(point_3d.x);
                        pts_det_y.push_back(point_3d.y);
                        pts_det_z.push_back(point_3d.z);
                    }

                }

                if (pts_det_x.size() > 5 && pts_det_z.size() > 5) {
                    std::sort(pts_det_x.begin(), pts_det_x.end());
                    std::sort(pts_det_z.begin(), pts_det_z.end());
                    std::sort(pts_det_y.begin(), pts_det_y.end());
                    double median_x = pts_det_x.at(static_cast<unsigned>(pts_det_x.size() / 2));
                    double median_y = pts_det_y.at(static_cast<unsigned>(pts_det_y.size() / 2));
                    double median_z = pts_det_z.at(static_cast<unsigned>(pts_det_z.size() / 2));
                    median = Eigen::Vector4d(median_x, median_y, median_z, 1.0);
                    return true;
                }

                return false;
            }
        }
    }
}