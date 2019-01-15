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


#include "utils_visualization.h"

// std
#include <random>

// OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

// pcl
#include <pcl/common/transforms.h>

// eigen
#include <Eigen/Dense>

// Project
#include "camera.h"
#include "sun_utils/utils_visualization.h"
#include "utils_common.h"
#include "ground_model.h"

// Boost
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

namespace Eigen {
    namespace internal {
        template<typename Scalar>
        struct scalar_normal_dist_op {
            static boost::mt19937 rng;    // The uniform pseudo-random algorithm
            mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

            EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

            template<typename Index>
            inline const Scalar operator()(Index, Index = 0) const { return norm(rng); }
        };

        template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

        template<typename Scalar>
        struct functor_traits<scalar_normal_dist_op<Scalar> > {
            enum {
                Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false
            };
        };
    } // end namespace internal
} // end namespace Eigen

namespace SUN {
    namespace utils {
        namespace visualization {

            const int color_array_size = 50;
            unsigned char color_array[] = {
                    240, 62, 36,
                    245, 116, 32,
                    251, 174, 24,
                    213, 223, 38,
                    153, 204, 112,
                    136, 201, 141,
                    124, 201, 169,
                    100, 199, 230,
                    64, 120, 188,
                    61, 88, 167,
                    204, 0, 255,
                    255, 0, 0,
                    0, 178, 255,
                    255, 0, 191,
                    255, 229, 0,
                    0, 255, 102,
                    89, 255, 0,
                    128, 0, 255,
                    242, 0, 255,
                    242, 255, 0,
                    255, 0, 77,
                    51, 0, 255,
                    0, 255, 140,
                    0, 255, 25,
                    204, 255, 0,
                    255, 191, 0,
                    89, 0, 255,
                    0, 217, 255,
                    0, 64, 255,
                    255, 115, 0,
                    255, 0, 115,
                    166, 0, 255,
                    13, 0, 255,
                    0, 25, 255,
                    0, 255, 217,
                    0, 255, 64,
                    255, 38, 0,
                    255, 0, 153,
                    0, 140, 255,
                    255, 77, 0,
                    255, 153, 0,
                    0, 255, 179,
                    0, 102, 255,
                    255, 0, 38,
                    13, 255, 0,
                    166, 255, 0,
                    0, 255, 255,
                    128, 255, 0,
                    255, 0, 230,
                    51, 255, 0
            };

            void GenerateColor(unsigned int id, uint8_t &r, uint8_t &g, uint8_t &b) {
                int col_index = id;
                if (col_index > color_array_size)
                    col_index = col_index % color_array_size;
                int seed_idx = col_index * 3;
                r = static_cast<uint8_t>(color_array[seed_idx]);
                g = static_cast<uint8_t>(color_array[seed_idx + 1]);
                b = static_cast<uint8_t>(color_array[seed_idx + 2]);
            }

            void GenerateColor(unsigned int id, cv::Vec3f &color) {
                uint8_t r, g, b;
                GenerateColor(id, r, g, b);
                color = cv::Vec3f(static_cast<float>(b) / 255.0f, static_cast<float>(g) / 255.0f,
                                  static_cast<float>(r) / 255.0f);
            }

            void GenerateColor(unsigned int id, cv::Vec3b &color) {
                uint8_t r, g, b;
                GenerateColor(id, r, g, b);
                color = cv::Vec3b(b, g, r);
            }

            void
            GenerateHeatmapValue(double value, double min_val, double max_val, uint8_t &r, uint8_t &g, uint8_t &b) {
                double ratio = 2 * (value - min_val) / (max_val - min_val);
                r = static_cast<uint8_t>(std::max(0.0, 255 * (1.0 - ratio)));
                b = static_cast<uint8_t>(std::max(0.0, 255 * (ratio - 1.0)));
                g = static_cast<uint8_t>(255 - b - r);
            }

            void DrawObjectFilled(const std::vector<int> &indices, const Eigen::Vector4d &bounding_box_2d,
                                  const cv::Vec3b &color, double alpha, cv::Mat &ref_image) {
                // Draw overlay over proposal region
                for (auto ind:indices) {
                    int x, y;
                    SUN::utils::UnravelIndex(ind, ref_image.cols, &x, &y);
                    if (y > 0 && x > 0 && y < ref_image.rows &&
                        x < ref_image.cols)
                        ref_image.at<cv::Vec3b>(y, x) =
                                ref_image.at<cv::Vec3b>(y, x) * alpha +
                                (1 - alpha) * color;
                }
            }

            void
            ArrowedLine(cv::Point2d pt1, cv::Point2d pt2, const cv::Scalar &color, cv::Mat &ref_image, int thickness,
                        int line_type, int shift, double tipLength) {
                const double tipSize = cv::norm(pt1 - pt2) *
                                       tipLength; // Factor to normalize the size of the tip depending on the length of the arrow
                cv::line(ref_image, pt1, pt2, color, thickness, line_type, shift);
                const double angle = atan2((double) pt1.y - pt2.y, (double) pt1.x - pt2.x);
                cv::Point2d p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
                              cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
                cv::line(ref_image, p, pt2, color, thickness, line_type, shift);
                p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
                p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
                cv::line(ref_image, p, pt2, color, thickness, line_type, shift);
            }

            void DrawLine(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const SUN::utils::Camera &camera,
                          cv::Mat &ref_image, const cv::Scalar &color,
                          int thickness, const cv::Point2i &offset) {
                Eigen::Vector4d p1_4d, p2_4d;
                p1_4d[3] = p2_4d[3] = 1.0;
                p1_4d.head<3>() = p1;
                p2_4d.head<3>() = p2;
                Eigen::Vector3i projected_point_1 = camera.CameraToImage(p1_4d);
                Eigen::Vector3i projected_point_2 = camera.CameraToImage(p2_4d);
                auto cv_p1 = cv::Point2i(projected_point_1[0], projected_point_1[1]);
                auto cv_p2 = cv::Point2i(projected_point_2[0], projected_point_2[1]);

                bool p1_in_bounds = true;
                bool p2_in_bounds = true;
                if ((cv_p1.x < 0) && (cv_p1.y < 0) && (cv_p1.x > ref_image.cols) && (cv_p1.y > ref_image.rows))
                    p1_in_bounds = false;

                if ((cv_p2.x < 0) && (cv_p2.y < 0) && (cv_p2.x > ref_image.cols) && (cv_p2.y > ref_image.rows))
                    p2_in_bounds = false;

                // Draw line, but only if both end-points project into the image!
                if (p1_in_bounds || p2_in_bounds) { // This is correct. Won't draw only if both lines are out of bounds.
                    // Draw line
                    auto p1_offs = offset + cv_p1;
                    auto p2_offs = offset + cv_p2;
                    if (cv::clipLine(cv::Size(/*0, 0, */ref_image.cols, ref_image.rows), p1_offs, p2_offs)) {
                        cv::line(ref_image, p1_offs, p2_offs, color, thickness, cv::LINE_AA);
                    }
                }
            }

            void
            DrawTransparentSquare(cv::Point center, cv::Vec3b color, int radius, double alpha, cv::Mat &ref_image) {
                for (int i = -radius; i < radius; i++) {
                    for (int j = -radius; j < radius; j++) {
                        int coord_y = center.y + i;
                        int coord_x = center.x + j;

                        if (coord_x > 0 && coord_y > 0 && coord_x < ref_image.cols && coord_y < ref_image.rows) {
                            ref_image.at<cv::Vec3b>(cv::Point(coord_x, coord_y)) =
                                    (1.0 - alpha) * ref_image.at<cv::Vec3b>(cv::Point(coord_x, coord_y)) +
                                    alpha * color;

                        }
                    }
                }
            }

            cv::Mat GetHeatmapFromEigenMatrix(const Eigen::MatrixXd &mat_to_visualize) {
                cv::Mat cv_heatmap(mat_to_visualize.cols(), mat_to_visualize.rows(), CV_32FC1);
                cv::eigen2cv(mat_to_visualize, cv_heatmap);
                double min, max;
                cv::minMaxIdx(cv_heatmap, &min, &max);
                return GetHeatmapFromEigenMatrix(mat_to_visualize, min, max);
            }

            cv::Mat GetHeatmapFromEigenMatrix(const Eigen::MatrixXd &mat_to_visualize, double min, double max) {
                cv::Mat cv_heatmap(mat_to_visualize.cols(), mat_to_visualize.rows(), CV_32FC1);
                cv::eigen2cv(mat_to_visualize, cv_heatmap);
                cv::Mat temp;
                cv_heatmap.convertTo(temp, CV_8UC1, 255 / (max - min), -min * 255 / (max - min));
                cv::applyColorMap(temp, temp, cv::COLORMAP_JET);
                //cv::flip(temp, temp, 0);
                return temp;
            }

            void DrawHeatmapBar(cv::Mat &ref_img, double minimum, double maximum,
                                std::function<void(double, double, double, uint8_t &, uint8_t &,
                                                   uint8_t &)> f_heatmap) {
                const int bar_width = 100;
                const int bar_height = 25;
                const int offset_x = 20, offset_y = 20;
                for (int i = 0; i < bar_width; i++) {
                    double val = (static_cast<double>(i) / static_cast<double>(bar_width)) *
                                 maximum; // TODO is minimum taken into account?
                    //double ratio = 2 * (val-minimum) / (maximum - minimum);
                    uint8_t r, g, b;
                    f_heatmap(val, minimum, maximum, r, g, b);
                    for (int j = 0; j < bar_height; j++) {
                        ref_img.at<cv::Vec3b>(j + offset_y, i + offset_x) = cv::Vec3b(r, g, b);
                    }
                }
            }

            void DrawBoundingBox2d(const Eigen::VectorXd &bounding_box_2d, cv::Mat &ref_image, uint8_t r, uint8_t g,
                                   uint8_t b, int thickness) {
                cv::Rect rect(cv::Point2d(bounding_box_2d[0], bounding_box_2d[1]),
                              cv::Size(bounding_box_2d[2], bounding_box_2d[3]));
                cv::rectangle(ref_image, rect, cv::Scalar(b, g, r), thickness);
            }

            void DrawBoundingBox3d(const Eigen::VectorXd &bounding_box_3d, cv::Mat &ref_image,
                                   const SUN::utils::Camera &camera, uint8_t r, uint8_t g, uint8_t b) {
                // Params
                const int line_width = 2;
                auto cv_color_triplet = cv::Scalar(r, g, b);

                // Center
                Eigen::Vector4d center;
                center.head(3) = bounding_box_3d.head(3);
                center[3] = 1.0;

                // Direction vector, obtain from the hypo points.
                Eigen::Vector3d dir(0.0, 0.0, 1.0); // Fixed, frontal dir. //= //hypo.GetDirectionVector(4);

                // Dimensions
                auto width = bounding_box_3d[3];
                auto height = bounding_box_3d[4];
                auto length = bounding_box_3d[5];

                // Bounding box is defined by up-vector, direction vector and a vector, orthogonal to the two.
                Eigen::Vector3d ground_plane_normal(0.0, -1.0, 0.0);
                Eigen::Vector3d ort_to_dir_vector = dir.cross(ground_plane_normal);
                Eigen::Vector3d center_proj_to_ground_plane = center.head(3);
                center_proj_to_ground_plane = camera.ground_model()->ProjectPointToGround(center_proj_to_ground_plane);

                // Re-compute dir and or vectors
                Eigen::Vector3d dir_recomputed = ort_to_dir_vector.cross(
                        ground_plane_normal); //ort_to_dir_vector.cross(ground_plane_normal);
                Eigen::Vector3d ort_recomputed = ground_plane_normal.cross(dir_recomputed);

                // Scale these vectors by bounding-box dimensions
                Eigen::Vector3d dir_scaled = dir_recomputed.normalized() * (length / 2.0) * -1.0;
                Eigen::Vector3d ort_scaled = ort_recomputed.normalized() * (width / 2.0);
                Eigen::Vector3d ground_plane_normal_scaled = ground_plane_normal * height;

                std::vector<Eigen::Vector3d> rect_3d_points(8);
                std::vector<int> rect_3d_visibility_of_points(8);

                // Render 3d bounding-rectangle into the image
                // Compute 8 corner of a rectangle
                rect_3d_points.at(0) = center_proj_to_ground_plane + dir_scaled + ort_scaled;
                rect_3d_points.at(1) = center_proj_to_ground_plane - dir_scaled + ort_scaled;
                rect_3d_points.at(2) = center_proj_to_ground_plane + dir_scaled - ort_scaled;
                rect_3d_points.at(3) = center_proj_to_ground_plane - dir_scaled - ort_scaled;
                rect_3d_points.at(4) =
                        center_proj_to_ground_plane + dir_scaled + ort_scaled + ground_plane_normal_scaled;
                rect_3d_points.at(5) =
                        center_proj_to_ground_plane - dir_scaled + ort_scaled + ground_plane_normal_scaled;
                rect_3d_points.at(6) =
                        center_proj_to_ground_plane + dir_scaled - ort_scaled + ground_plane_normal_scaled;
                rect_3d_points.at(7) =
                        center_proj_to_ground_plane - dir_scaled - ort_scaled + ground_plane_normal_scaled;

                // Compute point visibility
                for (int i = 0; i < 8; i++) {
                    Eigen::Vector4d pt_4d;
                    pt_4d.head<3>() = rect_3d_points.at(i);
                    pt_4d[3] = 1.0;
                    rect_3d_visibility_of_points.at(i) = camera.IsPointInFrontOfCamera(pt_4d);
                }

                // Render lines
                DrawLine(rect_3d_points.at(0), rect_3d_points.at(1), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(1), rect_3d_points.at(3), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(3), rect_3d_points.at(2), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(2), rect_3d_points.at(0), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(4), rect_3d_points.at(5), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(5), rect_3d_points.at(7), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(7), rect_3d_points.at(6), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(6), rect_3d_points.at(4), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(0), rect_3d_points.at(4), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(1), rect_3d_points.at(5), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(2), rect_3d_points.at(6), camera, ref_image, cv_color_triplet, line_width);
                DrawLine(rect_3d_points.at(3), rect_3d_points.at(7), camera, ref_image, cv_color_triplet, line_width);
            }

            void RenderBoundingBox3D(pcl::visualization::PCLVisualizer &viewer, const Eigen::VectorXd &bbox, double r,
                                     double g, double b, std::string &id, const int viewport) {

                assert(bbox.size() >= 6);

                const double w_by_2 = bbox[3] / 2.0;
                const double h_by_2 = bbox[4] / 2.0;
                const double l_by_2 = bbox[5] / 2.0;
                const double cx = bbox[0];
                const double cy = bbox[1];
                const double cz = bbox[2];

                std::vector<pcl::PointXYZ> pts3d(8);
                pts3d[0] = pcl::PointXYZ(cx + w_by_2, cy - h_by_2, cz - l_by_2); // 1
                pts3d[1] = pcl::PointXYZ(cx + w_by_2, cy + h_by_2, cz - l_by_2); // 2
                pts3d[2] = pcl::PointXYZ(cx - w_by_2, cy + h_by_2, cz - l_by_2); // 3
                pts3d[3] = pcl::PointXYZ(cx - w_by_2, cy - h_by_2, cz - l_by_2); // 4
                pts3d[4] = pcl::PointXYZ(cx + w_by_2, cy - h_by_2, cz + l_by_2); // 5
                pts3d[5] = pcl::PointXYZ(cx + w_by_2, cy + h_by_2, cz + l_by_2); // 6
                pts3d[6] = pcl::PointXYZ(cx - w_by_2, cy + h_by_2, cz + l_by_2); // 7
                pts3d[7] = pcl::PointXYZ(cx - w_by_2, cy - h_by_2, cz + l_by_2); // 8

                Eigen::Vector3f center(cx, cy, cz);

                Eigen::Quaternionf q;
                if (bbox.size() == 10) {
                    q = Eigen::Quaternionf(bbox[6], bbox[7], bbox[8], bbox[9]);
                } else {
                    q = Eigen::Quaternionf::Identity();
                }
                viewer.addCube(center, q, bbox[3], bbox[4], bbox[5], id, viewport);

                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4.0, id, viewport);
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, id);
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.7, id);
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                                   pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
            }


            void DrawGroundPoints(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr point_cloud, const Camera &camera,
                                  cv::Mat &ref_image, double minDistance, double maxDistance,
                                  double max_dist_from_camera) {
                const double alpha = 0.5;
                for (int i = 0; i < point_cloud->width; i++) {
                    for (int j = 0; j < point_cloud->height; j++) {
                        const auto &pt = point_cloud->at(i, j);

                        if (pt.z > max_dist_from_camera)
                            continue;

                        double dist_to_ground = camera.ground_model()->DistanceToGround(
                                pt.getVector3fMap().cast<double>());
                        if ((dist_to_ground < minDistance) || (std::abs(dist_to_ground) > maxDistance)) {
                            ref_image.at<cv::Vec3b>(j, i) =
                                    alpha * ref_image.at<cv::Vec3b>(j, i) + (1.0 - alpha) * cv::Vec3b(0, 0, 255);
                        }
                    }
                }
            }

            void DrawFlowMapAsArrows(const cv::Mat &flow_map, const cv::Scalar &arrow_color, cv::Mat &ref_image,
                                     int pixel_skip_for_visualization) {
                for (int32_t v = 0; v < flow_map.rows; v++) {
                    for (int32_t u = 0; u < flow_map.cols; u++) {
                        if ((v % pixel_skip_for_visualization == 0) && (u % pixel_skip_for_visualization == 0)) {
                            if (u > 0 && v > 0 && u < flow_map.cols && v < flow_map.rows) {
                                const auto &flow_u = flow_map.at<cv::Vec2f>(v, u)[0];
                                const auto &flow_v = flow_map.at<cv::Vec2f>(v, u)[1];
                                bool is_valid_meas = !(std::isnan(flow_u) || std::isnan(flow_v));
                                if (is_valid_meas) {
                                    const auto center_point = cv::Point2d(u, v);
                                    SUN::utils::visualization::ArrowedLine(
                                            center_point,
                                            cv::Point2d(center_point.x + flow_u, center_point.y + flow_v),
                                            arrow_color, ref_image,
                                            1, 8, 0, 0.1);
                                    ref_image.at<cv::Vec3b>(center_point) = cv::Vec3b(255, 255, 0);
                                }
                            }
                        }
                    }
                }
            }

            void
            DrawTransparentBoundingBox(const Eigen::Vector4d &bounding_box_2d, const cv::Vec3b &color, double alpha,
                                       cv::Mat &ref_image) {
                for (int u = bounding_box_2d[0]; u < (bounding_box_2d[0] + bounding_box_2d[2]); u++) {
                    for (int v = bounding_box_2d[1]; v < (bounding_box_2d[1] + bounding_box_2d[3]); v++) {
                        ref_image.at<cv::Vec3b>(v, u) = ref_image.at<cv::Vec3b>(v, u) * alpha + (1 - alpha) * color;
                    }
                }
                SUN::utils::visualization::DrawBoundingBox2d(bounding_box_2d, ref_image, 0, 0, 0);
            }

            // Adapted from Vincent Spruyt
            void
            DrawCovarianceMatrix2dEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image,
                                          cv::Vec3f color) {
                // Get the eigenvalues and eigenvectors
                cv::Mat eigenvalues, eigenvectors;
                cv::eigen(covmat, eigenvalues, eigenvectors);

                //Calculate the angle between the largest eigenvector and the x-axis
                double angle = atan2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(0, 0));

                // Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
                if (angle < 0)
                    angle += 6.28318530718;

                // Convert to degrees instead of radians
                angle = 180 * angle / 3.14159265359;

                // Calculate the size of the minor and major axes
                double half_majoraxis_size = chisquare_val * sqrt(std::abs(eigenvalues.at<double>(0)));
                double half_minoraxis_size = chisquare_val * sqrt(std::abs(eigenvalues.at<double>(1)));

                // Return the oriented ellipse
                // The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
                cv::RotatedRect rot_rect(mean, cv::Size2f(half_majoraxis_size, half_minoraxis_size), /*-*/angle);
                cv::ellipse(ref_image, rot_rect, cv::Scalar(color[0], color[1], color[2]), 1);
            }

            // Based on: http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
            void
            DrawCovarianceMatrix2dSmooth(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image,
                                         cv::Vec3f color) {
                // Get the eigenvalues and eigenvectors
                cv::Mat eigenvalues, eigenvectors;
                cv::eigen(covmat, eigenvalues, eigenvectors);

                int size = 2; // Dimensionality (rows)
                int nn = 10000; // How many samples (columns) to draw
                Eigen::internal::scalar_normal_dist_op<double> randN; // Gaussian functor
                Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1); // Seed the rng

                // Define mean and covariance of the distribution
                Eigen::VectorXd meanE(size);
                Eigen::MatrixXd covar(size, size);
                covmat.at<double>(0, 0);
                meanE << mean.x, mean.y;
                covar << covmat.at<double>(0, 0), covmat.at<double>(0, 1),
                        covmat.at<double>(1, 0), covmat.at<double>(1, 1);

                Eigen::MatrixXd normTransform(size, size);
                Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);

                // We can only use the cholesky decomposition if
                // the covariance matrix is symmetric, pos-definite.
                // But a covariance matrix might be pos-semi-definite.
                // In that case, we'll go to an EigenSolver
                if (cholSolver.info() == Eigen::Success) {
                    // Use cholesky solver
                    normTransform = cholSolver.matrixL();
                } else {
                    // Use eigen solver
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
                    Eigen::MatrixXd eigenvaluesEigen;
                    cv::cv2eigen(eigenvalues, eigenvaluesEigen);
                    normTransform = eigenSolver.eigenvectors() * eigenvaluesEigen.cwiseAbs().cwiseSqrt().asDiagonal();
                    if (!((normTransform.array() == normTransform.array())).all()) {
                        std::cout << "nan" << std::endl;
                    }
                }

                Eigen::MatrixXd samples =
                        (normTransform * Eigen::MatrixXd::NullaryExpr(size, nn, randN)).colwise() + meanE;
                auto TransparentCircleFloat = [](cv::Point center, cv::Vec3f color, int radius, double alpha,
                                                 cv::Mat &ref_image) {
                    for (int i = -radius; i < radius; i++) {
                        for (int j = -radius; j < radius; j++) {
                            int coord_y = center.y + i;
                            int coord_x = center.x + j;
                            if (coord_x > 0 && coord_y > 0 && coord_x < ref_image.cols && coord_y < ref_image.rows) {
                                ref_image.at<cv::Vec3f>(cv::Point(coord_x, coord_y)) =
                                        (1.0 - alpha) * ref_image.at<cv::Vec3f>(cv::Point(coord_x, coord_y)) +
                                        alpha * color;

                            }
                        }
                    }
                };

                const double alpha = 0.5; //03;
                if (covar(0, 0) != 0) {
                    for (int i = 0; i < nn; i++) {
                        int x = static_cast<int>(samples(1, i));
                        int y = static_cast<int>(samples(0, i));
                        if (x > 0 && y > 0 && x < ref_image.rows && y < ref_image.cols) {
                            TransparentCircleFloat(cv::Point(y, x), color, 2.0, alpha, ref_image);
                        }
                    }
                }
            }
        }
    }
}
