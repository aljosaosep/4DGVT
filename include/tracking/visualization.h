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

#ifndef GOT_TRACKING_VISUALIZATION
#define GOT_TRACKING_VISUALIZATION

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// PCL
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

// Tracking
#include <tracking/hypothesis.h>


// Utils
#include "sun_utils/utils_kitti.h"

// Fwd. decl.
namespace SUN { namespace utils { class Camera; }}
namespace SUN { namespace utils { namespace scene_flow { class VelocityInfo; }}}

namespace GOT {
    namespace tracking {
        typedef std::function<void(int, const GOT::tracking::Hypothesis &, const SUN::utils::Camera &,
                                   cv::Mat &)> DrawHypoFrameFnc;
        typedef std::function<void(const GOT::tracking::Hypothesis &, const SUN::utils::Camera &,
                                   cv::Mat &)> DrawHypoFnc;
        typedef std::function<void(const GOT::tracking::Observation &, const SUN::utils::Camera &, cv::Mat &,
                                   int)> DrawObsFnc;

        namespace draw_hypos {
            /**
              * @brief Draws trajectory of the hypothesis into the image.
              */
            void
            DrawTrajectoryToGroundPlane(const std::vector<Eigen::Vector4d> &poses, const SUN::utils::Camera &camera,
                                        const cv::Scalar &color, cv::Mat &ref_image, int line_width = 1,
                                        int num_poses_to_draw = 100, int smoothing_window_size = 20);


            // These are new; can specify ref. frame
            void
            DrawHypothesis2dForFrame(int frame, const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                     cv::Mat &ref_image);

            void DrawHypothesisMaskForFrame(int frame, const GOT::tracking::Hypothesis &hypo,
                                            const SUN::utils::Camera &camera, cv::Mat &ref_image);

            void DrawHypothesis2dWithCategoryInfoForFrame(int frame, const GOT::tracking::Hypothesis &hypo,
                                                          const SUN::utils::Camera &camera, cv::Mat &ref_image,
                                                          const std::map<int, std::string> &category_map);


            void
            DrawHypothesis3dForFrame(int frame, const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                     cv::Mat &ref_image);

            void DrawHypothesisMask(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                    cv::Mat &ref_image);

            void DrawHypothesis3d(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                  cv::Mat &ref_image);

            void DrawHypoShapeModel(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                    cv::Mat &ref_image, double alpha, const cv::Vec3b &custom_color);

            void DrawHypothesis2d(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                  cv::Mat &ref_image);

        }

        namespace draw_observations {
            void DrawObservationByID(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam,
                                     cv::Mat &ref_img, int index);
        }

        class Visualizer {

        public:
            Visualizer();

            const void GetColor(int index, double &r, double &g, double &b) const;

            const void GetColor(int index, uint8_t &r, uint8_t &g, uint8_t &b) const;

            // -------------------------------------------------------------------------------
            // +++ 3D VISUALIZER METHODS +++
            // -------------------------------------------------------------------------------

            /**
              * @brief Renders 3D bounding box (axis-aligned).
              */
            static void
            RenderBoundingBox3D(pcl::visualization::PCLVisualizer &viewer, const Eigen::VectorXd &bbox, double r,
                                double g, double b,
                                std::string &id, const int viewport);

            /**
              * @brief Renders 3D bounding box (oriented).
              */
            static void RenderBoundingBox3D(pcl::visualization::PCLVisualizer &viewer, const Eigen::VectorXd &bbox,
                                            double yaw_angle, double r, double g,
                                            double b, std::string &id, const int viewport);

            /**
              * @brief Renders 3D bounding-box representation of the hypothesis.
              */
            void RenderHypo3D(pcl::visualization::PCLVisualizer &viewer, const GOT::tracking::Hypothesis &hypo,
                              const SUN::utils::Camera &camera,
                              const int viewport);


            /**
              * @brief Renders trajectory of the tracked object.
              */
            void RenderTrajectory(int current_frame,
                                  const GOT::tracking::Hypothesis &hypo,
                                  const SUN::utils::Camera &camera,
                                  const std::string &traj_id,
                                  double r, double g, double b,
                                  pcl::visualization::PCLVisualizer &viewer,
                                  int viewport = 0);

            // -------------------------------------------------------------------------------
            // +++ DRAW HYPOS/OBSERVATIONS +++
            // -------------------------------------------------------------------------------
            void DrawHypotheses(const std::vector<GOT::tracking::Hypothesis> &hypos, const SUN::utils::Camera &camera,
                                cv::Mat &ref_image,
                                DrawHypoFnc draw_hypo_fnc) const;

            void DrawPredictions(const std::vector<GOT::tracking::Hypothesis> &hypos, const SUN::utils::Camera &camera,
                                 cv::Mat &ref_image);


            void DrawObservations(const std::vector<GOT::tracking::Observation> &observations, cv::Mat &ref_img,
                                  const SUN::utils::Camera &cam,
                                  DrawObsFnc draw_obs_fnc) const;

            // -------------------------------------------------------------------------------
            // +++ TOOLS +++
            // -------------------------------------------------------------------------------

            void DrawSparseFlow(const std::vector<SUN::utils::scene_flow::VelocityInfo> &sparse_flow_info,
                                const SUN::utils::Camera &camera, cv::Mat &ref_image);

            /**
             * @brief Computes vertices of oriented 3D bounding box, that represents the tracked object.
             */
            static std::vector<Eigen::Vector3d> ComputeOrientedBoundingBoxVertices(int frame,
                                                                                   const GOT::tracking::Hypothesis &hypo,
                                                                                   const SUN::utils::Camera &camera,
                                                                                   bool filtered_orientation = false);
        };

    }
}

#endif
