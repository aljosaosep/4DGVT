/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Dirk Klostermann (osep, klostermann -at- vision.rwth-aachen.de)

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

#ifndef SUN_UTILS_VISUALIZATION
#define SUN_UTILS_VISUALIZATION

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// pcl
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

// project
#include "camera.h"


namespace SUN {
    namespace utils {
        namespace visualization {

            // -------------------------------------------------------------------------------
            // +++ COLOR TABLES +++
            // -------------------------------------------------------------------------------
            void GenerateColor(unsigned int id, uint8_t &r, uint8_t &g, uint8_t &b);

            void GenerateColor(unsigned int id, cv::Vec3f &color);

            void GenerateColor(unsigned int id, cv::Vec3b &color);

            // -------------------------------------------------------------------------------
            // +++ HEATMAP +++
            // -------------------------------------------------------------------------------
            void GenerateHeatmapValue(double value, double min_val, double max_val, uint8_t &r, uint8_t &g, uint8_t &b);

            cv::Mat GetHeatmapFromEigenMatrix(const Eigen::MatrixXd &mat_to_visualize);

            cv::Mat GetHeatmapFromEigenMatrix(const Eigen::MatrixXd &mat_to_visualize, double min, double max);

            void
            DrawHeatmapBar(cv::Mat &ref_img, double minimum, double maximum, std::function<void(double, double, double,
                                                                                                uint8_t &, uint8_t &,
                                                                                                uint8_t &)> f_heatmap);

            // -------------------------------------------------------------------------------
            // +++ PRIMITIVES +++
            // -------------------------------------------------------------------------------
            void
            DrawTransparentBoundingBox(const Eigen::Vector4d &bounding_box_2d, const cv::Vec3b &color, double alpha,
                                       cv::Mat &ref_image);

            void DrawLine(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const SUN::utils::Camera &camera,
                          cv::Mat &ref_image,
                          const cv::Scalar &color, int thickness = 1, const cv::Point2i &offset = cv::Point2i(0, 0));

            void DrawBoundingBox2d(const Eigen::VectorXd &bounding_box_2d, cv::Mat &ref_image, uint8_t r = 255,
                                   uint8_t g = 0, uint8_t b = 0, int thickness = 1.0);

            void DrawBoundingBox3d(const Eigen::VectorXd &bounding_box_3d, cv::Mat &ref_image,
                                   const SUN::utils::Camera &camera,
                                   uint8_t r = 255, uint8_t g = 0, uint8_t b = 0);

            void DrawObjectFilled(const std::vector<int> &indices, const Eigen::Vector4d &bounding_box_2d,
                                  const cv::Vec3b &color, double alpha, cv::Mat &ref_image);

            void ArrowedLine(cv::Point2d pt1, cv::Point2d pt2, const cv::Scalar &color, cv::Mat &ref_image,
                             int thickness = 1, int line_type = 8, int shift = 0,
                             double tipLength = 0.1);

            void DrawTransparentSquare(cv::Point center, cv::Vec3b color, int radius, double alpha, cv::Mat &ref_image);

            // -------------------------------------------------------------------------------
            // +++ COVARIANCE MATRICES +++
            // -------------------------------------------------------------------------------
            /**
              * @brief Draws an iso-contour of the covariance matrix (iso-contour is picked via chisquare_val)
              */
            void
            DrawCovarianceMatrix2dEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image,
                                          cv::Vec3f color);

            /**
              * @brief Draws smooth representation of covariance matrix (via particles).
              * @author Dirk (klostermann@rwth-aachen.de)
              */
            void
            DrawCovarianceMatrix2dSmooth(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image,
                                         cv::Vec3f color);


            // -------------------------------------------------------------------------------
            // +++ ETC +++
            // -------------------------------------------------------------------------------
            /**
             * @brief Specify your *-flow map (optical-flow, ego-flow, ...) map and this fnc. will draw you 'arros' for flow vectors.
             * @param[in] flow_map Your flow map, representing pixel diff. past-to-current frame.
             * @param[in] arrow_color Tell fnc what color you like for the arrows.
             * @param[out] ref_image Draw arrows on this image.
             * @param[in] pixel_skip_for_visualization Num. pixels you want to skip -- you can't draw all arrows (well you can, but you won't see much)
             */
            void DrawFlowMapAsArrows(const cv::Mat &flow_map, const cv::Scalar &arrow_color, cv::Mat &ref_image,
                                     int pixel_skip_for_visualization = 5);

            void RenderBoundingBox3D(pcl::visualization::PCLVisualizer &viewer, const Eigen::VectorXd &bbox,
                                     double r, double g, double b, std::string &id, const int viewport = 0);

        }
    }
}

#endif
