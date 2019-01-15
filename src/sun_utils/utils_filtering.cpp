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
#include "utils_filtering.h"

// pcl
#include <pcl/features/integral_image_normal.h>

// utils
#include "ground_model.h"
#include "utils_common.h"

namespace SUN {
    namespace utils {
        namespace filter {

            std::vector<int> FilterKeepInnerqQuartile(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud,
                                                      const std::vector<int> inds) {

                struct PointXYZIDX {
                    Eigen::Vector3d p_;
                    int idx_;

                    PointXYZIDX(const Eigen::Vector3d &p, int idx) {
                        p_ = p;
                        idx_ = idx;
                    }
                };

                std::vector<PointXYZIDX> pts;
                std::vector<double> pts_det_x;
                std::vector<double> pts_det_z;
                std::vector<double> pts_det_y;
                std::vector<int> inds_out;

                for (int idx:inds) {
                    int u, v;
                    SUN::utils::UnravelIndex(idx, point_cloud->width, &u, &v);
                    const auto &point_3d = point_cloud->at(u, v); // (col, row)
                    if (!std::isnan(point_3d.x)) {
                        pts.push_back(PointXYZIDX(Eigen::Vector3d(point_3d.x, point_3d.y, point_3d.z), idx));
                        pts_det_x.push_back(point_3d.x);
                        pts_det_y.push_back(point_3d.y);
                        pts_det_z.push_back(point_3d.z);
                    }
                }

                if (pts_det_x.size() > 5 && pts_det_z.size() > 5) {
                    auto num_pts = pts_det_z.size();
                    // Get median point
                    std::sort(pts_det_x.begin(), pts_det_x.end());
                    std::sort(pts_det_z.begin(), pts_det_z.end());
                    std::sort(pts_det_y.begin(), pts_det_y.end());
                    double median_x = pts_det_x.at(static_cast<unsigned>(pts_det_x.size() / 2));
                    double median_y = pts_det_y.at(static_cast<unsigned>(pts_det_y.size() / 2));
                    double median_z = pts_det_z.at(static_cast<unsigned>(pts_det_z.size() / 2));
                    Eigen::Vector3d median = Eigen::Vector3d(median_x, median_y, median_z);

                    auto DistCmpFnc = [median](const PointXYZIDX &a, const PointXYZIDX &b) -> bool {
                        return (median - a.p_).squaredNorm() < (median - b.p_).squaredNorm();
                    };
                    std::sort(pts.begin(), pts.end(), DistCmpFnc);

                    for (auto i = static_cast<size_t>(num_pts * (1.0 / 4.0));
                         i < static_cast<size_t>(num_pts * (3.0 / 4.0)); i++) {
                        // Loop the inner quartile
                        const auto pt_xyzidx = pts.at(i);
                        inds_out.push_back(pt_xyzidx.idx_);
                    }
                }

                return inds_out;
            }

            void FilterPointCloudBasedOnSemanticMapRemoveCategory(
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned,
                    const cv::Mat &semantic_map, const cv::Vec3b semantic_label_to_remove,
                    bool only_color_outlier_points) {
                const int w = semantic_map.cols;
                const int h = semantic_map.rows;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        cv::Vec3b pixel_semantic_label = semantic_map.at<cv::Vec3b>(y, x);
                        if (semantic_label_to_remove == pixel_semantic_label) {
                            auto &p = cloud_to_be_cleaned->at(x, y);
                            if (only_color_outlier_points) {
                                p.r = static_cast<uint8_t>(255);
                                p.g = static_cast<uint8_t>(0);
                                p.b = static_cast<uint8_t>(0);
                                p.a = static_cast<uint8_t>(255);
                            } else {
                                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                                p.r = p.g = p.b = p.a = std::numeric_limits<uint8_t>::quiet_NaN();
                            }
                        }
                    }
                }
            }

            //! WARNING: cv::Vec3b assumes BGR order, rather than RGB.
            void FilterPointCloudBasedOnSemanticMap(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned,
                                                    const cv::Mat &semantic_map, const cv::Vec3b semantic_label_to_keep,
                                                    bool only_color_outlier_points) {
                const int w = semantic_map.cols;
                const int h = semantic_map.rows;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        cv::Vec3b pixel_semantic_label = semantic_map.at<cv::Vec3b>(y, x);
                        if (semantic_label_to_keep != pixel_semantic_label) {
                            auto &p = cloud_to_be_cleaned->at(x, y);
                            if (only_color_outlier_points) {
                                // p.rgb = GOT::utils::pointcloud::PackRgbValuesToUint32(255, 0,0 );
                                p.r = static_cast<uint8_t>(255);
                                p.g = static_cast<uint8_t>(0);
                                p.b = static_cast<uint8_t>(0);
                                p.a = static_cast<uint8_t>(255);
                            } else {
                                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                                p.r = p.g = p.b = p.a = std::numeric_limits<uint8_t>::quiet_NaN();
                            }
                        }
                    }
                }
            }

            void
            FilterPointCloudBasedOnDistanceToGroundPlane(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_to_be_cleaned,
                                                         std::shared_ptr<SUN::utils::GroundModel> ground_model,
                                                         const double minDistance, const double maxDistance,
                                                         bool only_color_outlier_points) {
                for (auto &p:cloud_to_be_cleaned->points) {
                    double dist_to_ground = ground_model->DistanceToGround(p.getVector3fMap().cast<double>());

                    if ((dist_to_ground < minDistance) || (std::abs(dist_to_ground) > maxDistance)) {
                        if (only_color_outlier_points) {
                            p.r = static_cast<uint8_t>(255);
                            p.g = static_cast<uint8_t>(0);
                            p.b = static_cast<uint8_t>(0);
                            p.a = static_cast<uint8_t>(255);
                        } else {
                            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                            p.r = p.g = p.b = p.a = std::numeric_limits<uint8_t>::quiet_NaN();
                        }
                    }
                }
            }
        }
    }
}