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

// tracking
#include <tracking/visualization.h>

// std
#include <algorithm>
#include <iostream>
#include <memory>
#include <functional>

// OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

// pcl
#include <pcl/common/transforms.h>
#include <src/sun_utils/utils_flow.h>

// Utils
#include "sun_utils/utils_common.h"
#include "sun_utils/utils_io.h"
#include "sun_utils/utils_visualization.h"
#include "sun_utils/ground_model.h"

#define MAX_PATH_LEN 500

namespace GOT {
    namespace tracking {

        Visualizer::Visualizer() {

        }

        std::vector<Eigen::Vector4d>
        SmoothTrajectoryPoses(const std::vector<Eigen::Vector4d> &poses, int kernel_size) {
            const int width = std::floor(kernel_size / 2.0);
            const int num_poses = poses.size();

            // Kernel size unreasonably small -> return original points.
            if (width <= 0)
                return poses;

            std::vector<Eigen::Vector4d> smooth_poses;
            for (int i = 0; i < num_poses; i++) {
                const int local_window = std::min(std::min(i, width), num_poses - i -
                                                                      1); // Make sure we don't spill over the boundaries
                const int left_pos = i - local_window;
                const int right_pos = i + local_window;

                double mean_x = 0.0;
                for (int j = left_pos; j <= right_pos; j++)
                    mean_x += poses.at(j)[0];

                if ((right_pos - left_pos) > 0) {
                    mean_x *= (1.0 / ((right_pos - left_pos) + 1));
                    Eigen::Vector4d smooth_pose = poses.at(i);
                    smooth_pose[0] = mean_x;
                    smooth_poses.push_back(smooth_pose);
                }
            }

            return smooth_poses;
        }

        std::vector<Eigen::Vector3d>
        Visualizer::ComputeOrientedBoundingBoxVertices(int frame, const GOT::tracking::Hypothesis &hypo,
                                                       const SUN::utils::Camera &camera, bool filtered_orientation) {

            if (hypo.cache().size() < 1) {
                return std::vector<Eigen::Vector3d>();
            }

            // Get last bounding-box
            auto bb3d = hypo.cache().at_frame(frame).box3();
            Eigen::Vector4d center;
            center.head(3) = bb3d.head(3);
            center[3] = 1.0;

//            // Orientation: either use velocity dir, or axis aligned (if object assumed static).
//            // Not great, but looks ok-ish
//            Eigen::Vector2d velocity_dir = hypo.kalman_filter_const()->GetVelocityGroundPlane();
//            Eigen::Vector3d dir(-velocity_dir[0], 0.0, velocity_dir[1]);
//            dir = camera.R().transpose()*dir;


            // Not sure what the fuck happening here
            // Orientation: either use velocity dir, or axis aligned (if object assumed static).
            // Not great, but looks ok-ish
            Eigen::Vector3d dir = Eigen::Vector3d(0.0, 0.0,
                                                  -1.0); // Is there better way to get the orientation that just set it fixed?
            if (hypo.kalman_filter_const() != nullptr) {
                Eigen::Vector2d velocity_dir = hypo.kalman_filter_const()->GetVelocityGroundPlane();
                dir = Eigen::Vector3d(/*-*/velocity_dir[0], 0.0, velocity_dir[1]);
                dir = camera.R().transpose() * dir;
            }

            if (hypo.kalman_filter_const()->GetVelocityGroundPlane().norm() < 2.0) {
                dir = Eigen::Vector3d(0.0, 0.0,
                                      -1.0); // Is there better way to get the orientation that just set it fixed?
            }

            //Eigen::Vector3d dir = dir = Eigen::Vector3d(0.0, 0.0, -1.0); // Is there better way to get the orientation that just set it fixed?

            // Dimensions
            auto width = bb3d[3];
            auto height = bb3d[4];
            auto length = bb3d[5];

            /// Bounding box is defined by up-vector, direction vector and a vector, orthogonal to the two.
            Eigen::Vector3d ground_plane_normal = camera.ground_model()->Normal(center.head<3>()); //(0.0,-1.0,0.0);
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

            /// Render 3d bounding-rectangle into the image
            // Compute 8 corner of a rectangle
            rect_3d_points.at(0) = center_proj_to_ground_plane + dir_scaled + ort_scaled;
            rect_3d_points.at(1) = center_proj_to_ground_plane - dir_scaled + ort_scaled;
            rect_3d_points.at(2) = center_proj_to_ground_plane + dir_scaled - ort_scaled;
            rect_3d_points.at(3) = center_proj_to_ground_plane - dir_scaled - ort_scaled;

            rect_3d_points.at(4) = center_proj_to_ground_plane + dir_scaled + ort_scaled + ground_plane_normal_scaled;
            rect_3d_points.at(5) = center_proj_to_ground_plane - dir_scaled + ort_scaled + ground_plane_normal_scaled;
            rect_3d_points.at(6) = center_proj_to_ground_plane + dir_scaled - ort_scaled + ground_plane_normal_scaled;
            rect_3d_points.at(7) = center_proj_to_ground_plane - dir_scaled - ort_scaled + ground_plane_normal_scaled;

            return rect_3d_points;
        }

        const void Visualizer::GetColor(int index, double &r, double &g, double &b) const {
            uint8_t ri, bi, gi;
            SUN::utils::visualization::GenerateColor(index, bi, gi, ri);

            b = static_cast<double>(ri) / 255.0;
            g = static_cast<double>(gi) / 255.0;
            r = static_cast<double>(bi) / 255.0;
        }

        const void Visualizer::GetColor(int index, uint8_t &r, uint8_t &g, uint8_t &b) const {
            SUN::utils::visualization::GenerateColor(index, b, g, r);
        }

        void Visualizer::DrawObservations(const std::vector<GOT::tracking::Observation> &observations, cv::Mat &ref_img,
                                          const SUN::utils::Camera &cam, DrawObsFnc draw_obs_fnc) const {
            auto observations_copy = observations;
            std::sort(observations_copy.begin(), observations_copy.end(),
                      [](const GOT::tracking::Observation &o1, const GOT::tracking::Observation &o2) {
                          return o1.score() < o2.score();
                      });


            for (int i = 0; i < observations.size(); i++) {
                draw_obs_fnc(observations.at(i), cam, ref_img, i);
            }
        }

        void Visualizer::DrawPredictions(const std::vector<GOT::tracking::Hypothesis> &hypos,
                                         const SUN::utils::Camera &camera, cv::Mat &ref_image) {
            for (const auto &hypo:hypos) {
                auto predicted_segment_cloud = hypo.cache().back().predicted_segment();

                // Draw prediction
                auto alpha = 0.5f;
                for (const auto &p:predicted_segment_cloud.points) {
                    Eigen::Vector4d p_eig;
                    p_eig.head<3>() = p.getVector3fMap().cast<double>();
                    p_eig[3] = 1.0;
                    Eigen::Vector3i p_proj = camera.CameraToImage(p_eig);
                    auto u = p_proj[0];
                    auto v = p_proj[1];
                    if (u >= 0 && v >= 0 && u < ref_image.cols && v < ref_image.rows) {
                        auto color = cv::Vec3b(0, 0, 255);
                        ref_image.at<cv::Vec3b>(v, u) =
                                ref_image.at<cv::Vec3b>(v, u) * alpha +
                                (1 - alpha) * color;
                    }
                }
            }
        }

        void Visualizer::DrawSparseFlow(const std::vector<SUN::utils::scene_flow::VelocityInfo> &sparse_flow_info,
                                        const SUN::utils::Camera &camera, cv::Mat &ref_image) {
            std::vector<Eigen::Vector3d> pts_p3d, pts_vel;
            for (const auto &inf:sparse_flow_info) {
                pts_p3d.push_back(inf.p_3d);
                pts_vel.push_back(inf.p_vel);
                Eigen::Vector3i p_proj = camera.CameraToImage(
                        Eigen::Vector4d(inf.p_3d[0], inf.p_3d[1], inf.p_3d[2], 1.0));
                Eigen::Vector3i p_prev = camera.CameraToImage(
                        Eigen::Vector4d(inf.p_prev[0], inf.p_prev[1], inf.p_prev[2], 1.0));
                Eigen::Vector3i p_prev_to_curr = camera.CameraToImage(
                        Eigen::Vector4d(inf.p_prev_to_curr[0], inf.p_prev_to_curr[1], inf.p_prev_to_curr[2], 1.0));
                SUN::utils::visualization::DrawTransparentSquare(cv::Point(inf.p[0], inf.p[1]), cv::Vec3b(0, 0, 255),
                                                                 2.0, 0.5, ref_image);
                SUN::utils::visualization::DrawLine(inf.p_3d, inf.p_3d + inf.p_vel, camera, ref_image,
                                                    cv::Vec3b(255, 0, 0));
            }
        }


        void Visualizer::DrawHypotheses(const std::vector<GOT::tracking::Hypothesis> &hypos,
                                        const SUN::utils::Camera &camera, cv::Mat &ref_image,
                                        DrawHypoFnc draw_hypo_fnc) const {
            GOT::tracking::HypothesesVector hypos_copy = hypos;
            std::sort(hypos_copy.begin(), hypos_copy.end(),
                      [](const GOT::tracking::Hypothesis &i, const GOT::tracking::Hypothesis &j) {
                          return (i.cache().back().pose_cam()[2]) > (j.cache().back().pose_cam()[2]);
                      });

            for (const auto &hypo:hypos_copy) {
                draw_hypo_fnc(hypo, camera, ref_image);
            }
        }


        void Visualizer::RenderBoundingBox3D(pcl::visualization::PCLVisualizer &viewer, const Eigen::VectorXd &bbox,
                                             double r, double g, double b, std::string &id, const int viewport) {
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
            Eigen::Quaternionf q(bbox[6], bbox[7], bbox[8], bbox[9]);

            viewer.addCube(center, q, bbox[3], bbox[4], bbox[5], id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3.0, id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, id);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                               pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
        }

        void Visualizer::RenderBoundingBox3D(pcl::visualization::PCLVisualizer &viewer, const Eigen::VectorXd &bbox,
                                             double yaw_angle, double r, double g, double b, std::string &id,
                                             const int viewport) {
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
            q = Eigen::AngleAxisf(yaw_angle/*+M_PI_2*/, Eigen::Vector3f::UnitY());
            //Eigen::Quaternionf q(bbox[6], bbox[7], bbox[8], bbox[9]);

            viewer.addCube(center, q, bbox[3], bbox[4], bbox[5], id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3.0, id, viewport);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, id);
            viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, id);
        }

        void Visualizer::RenderTrajectory(int current_frame,
                                          const GOT::tracking::Hypothesis &hypo,
                                          const SUN::utils::Camera &camera,
                                          const std::string &traj_id,
                                          double r, double g, double b,
                                          pcl::visualization::PCLVisualizer &viewer,
                                          int viewport) {
            // Get traj. up to current_frame
            auto hypo_traj = GOT::tracking::HypoCacheToPoses(hypo.cache(), current_frame);

            std::vector<Eigen::Vector4d> poses_copy = SmoothTrajectoryPoses(hypo_traj, 20);
            const int start_pose = std::max(1, static_cast<int>(poses_copy.size() - 40));
            for (int j = start_pose; j < poses_copy.size(); j++) {
                std::string pt_id = traj_id + std::to_string(j) + "_" + std::to_string(j);
                pcl::PointXYZRGBA p1, p2;

                Eigen::Vector4d p1_eig = poses_copy[j - 1];
                Eigen::Vector4d p2_eig = poses_copy[j];

                p1_eig[1] -= 0.1;
                p2_eig[1] -= 0.1;

                p1_eig = camera.WorldToCamera(p1_eig);
                p2_eig = camera.WorldToCamera(p2_eig);
                p1.getVector3fMap() = p1_eig.head<3>().cast<float>();
                p2.getVector3fMap() = p2_eig.head<3>().cast<float>();
                viewer.addLine(p1, p2, r, g, b, pt_id, viewport);
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4.0, pt_id, viewport);
            }
        }

        namespace draw_hypos {


            void
            DrawHypothesis2d(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                             cv::Mat &ref_image) {

                //if (hypo.terminated().IsTerminated())
                //    return;

                Eigen::Vector2d posterior_velocity_gp = hypo.kalman_filter_const()->GetVelocityGroundPlane();

                // Draw bbox + traj. + hypo id
                DrawHypothesis2dForFrame(hypo.cache().timestamps().back(), hypo, camera, ref_image);
            }


            void
            DrawTrajectoryToGroundPlane(const std::vector<Eigen::Vector4d> &poses, const SUN::utils::Camera &camera,
                                        const cv::Scalar &color, cv::Mat &ref_image, int line_width,
                                        int num_poses_to_draw, int smoothing_window_size) {
                std::vector<Eigen::Vector4d> poses_copy = SmoothTrajectoryPoses(poses, smoothing_window_size);
                const int start_pose = std::max(1, static_cast<int>(poses_copy.size() - num_poses_to_draw));
                for (int i=start_pose; i<poses_copy.size(); i++) {
                    // Take two consecutive poses, project them to camera space
                    auto pose1 = poses_copy.at(i - 1);
                    auto pose2 = poses_copy.at(i);
                    if (camera.IsPointInFrontOfCamera(pose1) && camera.IsPointInFrontOfCamera(pose2)) {
                        pose2 = camera.WorldToCamera(pose2);
                        pose1 = camera.WorldToCamera(pose1);
                        SUN::utils::visualization::DrawLine(pose1.head<3>(), pose2.head<3>(), camera, ref_image, color,
                                                            line_width);
                    }
                }
            }

            void
            DrawHypothesis3dForFrame(int frame, const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                     cv::Mat &ref_image) {
                if (hypo.cache().size() < 2)
                    return;

                // Params
                const int line_width = 4;
                const int num_poses_to_draw = 50;

                // Get R,G,B triplet from color lookup table.
                uint8_t r, g, b;
                SUN::utils::visualization::GenerateColor(hypo.id(), r, g, b); // RGB, BGR!
                auto cv_color_triplet = cv::Scalar(r, g, b);

                auto bb3d = hypo.cache().at_frame(frame).box3();
                Eigen::Vector4d center;
                center.head(3) = bb3d.head(3);
                center[3] = 1.0;

                // Draw 3D-bbox center 3D
                Eigen::Vector3i center_in_image = camera.CameraToImage(center);
                cv::circle(ref_image, cv::Point(center_in_image[0], center_in_image[1]), 1.0, cv::Scalar(0, 0, 255),
                           -1);
                std::vector<Eigen::Vector3d> rect_3d_points = Visualizer::ComputeOrientedBoundingBoxVertices(frame,
                                                                                                             hypo,
                                                                                                             camera);
                std::vector<int> rect_3d_visibility_of_points(8);

                // Compute point visibility
                for (int i = 0; i < 8; i++) {
                    Eigen::Vector4d pt_4d;
                    pt_4d.head<3>() = rect_3d_points.at(i);
                    pt_4d[3] = 1.0;
                    rect_3d_visibility_of_points.at(i) = camera.IsPointInFrontOfCamera(pt_4d);
                }

                // Render lines
                SUN::utils::visualization::DrawLine(rect_3d_points.at(0), rect_3d_points.at(1), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(1), rect_3d_points.at(3), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(3), rect_3d_points.at(2), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(2), rect_3d_points.at(0), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(4), rect_3d_points.at(5), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(5), rect_3d_points.at(7), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(7), rect_3d_points.at(6), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(6), rect_3d_points.at(4), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(0), rect_3d_points.at(4), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(1), rect_3d_points.at(5), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(2), rect_3d_points.at(6), camera, ref_image,
                                                    cv_color_triplet, line_width);
                SUN::utils::visualization::DrawLine(rect_3d_points.at(3), rect_3d_points.at(7), camera, ref_image,
                                                    cv_color_triplet, line_width);


                auto poses_subvec = GOT::tracking::HypoCacheToPoses(hypo.cache(), frame);
                DrawTrajectoryToGroundPlane(poses_subvec, camera, cv_color_triplet, ref_image, line_width,
                                            num_poses_to_draw, 20.0);
            }

            void
            DrawHypothesis2dForFrame(int frame, const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                     cv::Mat &ref_image) {

                bool is_terminated_in_this_frame = hypo.IsHypoTerminatedInFrame(frame);

                // Params
                const int line_width = 4;
                const int num_poses_to_draw = 50;

                // Get R,G,B triplet from color lookup table.
                int col_id = hypo.id();
                if (hypo.id() < 0)
                    col_id = 0;

                uint8_t r, g, b;
                SUN::utils::visualization::GenerateColor(col_id, r, g, b); // RGB, BGR!

                // Signal hypo termination with red color (if confused: r=0, becuse BGR!)
                if (is_terminated_in_this_frame) {
                    r = 0;
                    g = 0;
                    b = 255;
                }

                auto cv_color_triplet = cv::Scalar(r, g, b);

                /// Draw last bounding-box 2d.
                if (!hypo.cache().Exists(frame)) {
                    return;
                }

                /// Draw trajectory
                auto it = std::find(hypo.cache().timestamps().begin(), hypo.cache().timestamps().end(), frame);
                if (it == hypo.cache().timestamps().end()) {
                    return;
                }

                auto poses_subvec = GOT::tracking::HypoCacheToPoses(hypo.cache(), frame);
                DrawTrajectoryToGroundPlane(poses_subvec, camera, cv_color_triplet, ref_image, line_width,
                                            num_poses_to_draw, 20.0);

                auto bb2d = hypo.cache().at_frame(frame).box2();
                cv::Rect rect(cv::Point2d(bb2d[0], bb2d[1]), cv::Size(bb2d[2], bb2d[3]));
                cv::rectangle(ref_image, rect, cv_color_triplet, line_width);

                // Draw hypo id string
                cv::putText(ref_image, std::to_string(hypo.id()), cv::Point2d(bb2d[0] + 5, bb2d[1] + 20),
                            cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, cv_color_triplet, 2);
            }

            void DrawHypothesis2dWithCategoryInfoForFrame(int frame, const GOT::tracking::Hypothesis &hypo,
                                                          const SUN::utils::Camera &camera, cv::Mat &ref_image,
                                                          const std::map<int, std::string> &category_map) {
                bool is_terminated_in_this_frame = hypo.IsHypoTerminatedInFrame(frame);

                DrawHypothesis2dForFrame(frame, hypo, camera, ref_image);

                const auto post = hypo.category_probability_distribution();
                auto category_idx = static_cast<int>(
                        std::distance(post.begin(), std::max_element(post.begin(), post.end()))
                );

                if (category_map.count(category_idx) <= 0) {
                    assert(false);
                    return;
                }

                std::string categ_name = category_map.at(category_idx);
                if (is_terminated_in_this_frame) {
                    categ_name = "TERMINATED";
                }

                uint8_t r, g, b;
                int col_id = hypo.id();
                if (hypo.id() < 0) col_id = 0;
                SUN::utils::visualization::GenerateColor(col_id, r, g, b); // RGB, BGR!
                auto cv_color_triplet = cv::Scalar(r, g, b);

                // Need bounding box for text placing
                if (!hypo.cache().Exists(frame)) {
                    return;
                }
                const auto &bb2d = hypo.cache().at_frame(frame).box2();
                cv::putText(ref_image, categ_name, cv::Point2d(bb2d[0] + 5, bb2d[1] + 40), cv::FONT_HERSHEY_PLAIN, 0.8,
                            cv_color_triplet, 2);
            }

            std::vector<int> GetDilatedInds(const SUN::shared_types::CompressedMask &mask, int dilation_size = 4) {

                // 1. Draw mask to an image
                auto inds = mask.GetIndices();
                cv::Mat mask_image(mask.h_, mask.w_, CV_8UC1);
                mask_image *= 0;
                for (int ind : inds) {
                    int col, row;
                    SUN::utils::UnravelIndex(ind, mask.w_, &col, &row);
                    if (col >= 0 && row >= 0 && col < mask.w_ && row < mask.h_) {
                        mask_image.at<uchar>(row, col) = 255;
                    }
                }

                // 2. OpenCV dilation
                cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                                            cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                                            cv::Point(dilation_size, dilation_size));
                /// Apply the dilation operation
                cv::dilate(mask_image, mask_image, element);

                // 3. Turn binary mask back to inds
                std::vector<int> inds_out;
                for (int i = 0; i < mask_image.rows; i++) {
                    for (int j = 0; j < mask_image.cols; j++) {
                        if (mask_image.at<uchar>(i, j) == 255) {
                            // Add to inds
                            int ind;
                            SUN::utils::RavelIndex(j, i, mask_image.cols, &ind);
                            inds_out.push_back(ind);
                        }
                    }
                }

                mask_image.release();

                return inds_out;
            }

            void DrawHypothesisMaskForFrame(int frame, const GOT::tracking::Hypothesis &hypo,
                                            const SUN::utils::Camera &camera, cv::Mat &ref_image) {

                if (hypo.IsHypoTerminatedInFrame(frame)) {
                    return;
                }

                const float alpha = 0.5;

                // Params
                const int line_width = 4;
                const int num_poses_to_draw = 50;

                // Get R,G,B triplet from color lookup table.
                int col_id = hypo.id();
                if (hypo.id() < 0)
                    col_id = 0;

                uint8_t r, g, b;
                SUN::utils::visualization::GenerateColor(col_id, r, g, b); // RGB, BGR!

                auto im_copy = ref_image.clone();

                auto cv_color_triplet = cv::Scalar(r, g, b);

                /// Draw mask
                if (!hypo.cache().Exists(frame)) {
                    return;
                }
                const auto &mask = hypo.cache().at_frame(frame).mask();
                const auto &inds = mask.GetIndices();
                auto dilated_inds = GetDilatedInds(mask, 4);
                for (const auto &ind : dilated_inds) {
                    int x, y;
                    SUN::utils::UnravelIndex(ind, camera.width(), &x, &y);
                    ref_image.at<cv::Vec3b>(y, x) =
                            alpha * cv::Vec3b(255, 255, 255) + (1.0f - alpha) * ref_image.at<cv::Vec3b>(y, x);
                }

                dilated_inds.clear();
                for (const auto &ind : inds) {
                    int x, y;
                    SUN::utils::UnravelIndex(ind, camera.width(), &x, &y);
                    ref_image.at<cv::Vec3b>(y, x) =
                            alpha * cv::Vec3b(r, g, b) + (1.0f - alpha) * im_copy.at<cv::Vec3b>(y, x);
                }
                im_copy.release();

                /// Draw trajectory
                auto it = std::find(hypo.cache().timestamps().begin(), hypo.cache().timestamps().end(), frame);
                if (it == hypo.cache().timestamps().end()) {
                    return;
                }

                auto poses_subvec = GOT::tracking::HypoCacheToPoses(hypo.cache(), frame);
                DrawTrajectoryToGroundPlane(poses_subvec, camera, cv_color_triplet, ref_image, line_width,
                                            num_poses_to_draw, 20.0);
            }

            void DrawHypothesisMask(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                    cv::Mat &ref_image) {

                if (hypo.terminated().IsTerminated())
                    return;
                DrawHypothesisMaskForFrame(hypo.cache().timestamps().back(), hypo, camera, ref_image);
            }

            void DrawHypothesis3d(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                  cv::Mat &ref_image) {
                if (hypo.cache().size() < 2)
                    return;
                DrawHypothesis3dForFrame(hypo.cache().timestamps().back(), hypo, camera, ref_image);
            }

            void DrawHypoShapeModel(const GOT::tracking::Hypothesis &hypo, const SUN::utils::Camera &camera,
                                    cv::Mat &ref_image, double alpha, const cv::Vec3b &custom_color) {
                if (hypo.shape_model_const()) {
                    auto points = hypo.shape_model_const()->integrated_points();
                    if (points) {
                        for (const auto &pt:points->points) {
                            Eigen::Vector4f pt_eig_f = pt.getVector4fMap();
                            Eigen::Vector3i proj_pt = camera.WorldToImage(pt_eig_f.cast<double>());

                            const int col = proj_pt[0];
                            const int row = proj_pt[1];

                            if (col > 0 && row > 0 && col < ref_image.cols && row < ref_image.rows) {
                                auto color = custom_color;
                                if (color[0] == 0 && color[1] == 0 && color[2] == 0)
                                    color = cv::Vec3b(static_cast<uint8_t>(pt.b), static_cast<uint8_t>(pt.g),
                                                      static_cast<uint8_t>(pt.r));
                                SUN::utils::visualization::DrawTransparentSquare(cv::Point(col, row), color, 2, alpha,
                                                                                 ref_image);

                            }
                        }
                    }
                }
            }
        }

        namespace draw_observations {

            void DrawObservationDefault(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam,
                                        cv::Mat &ref_img, int index) {

                cv::Vec3b color;
                SUN::utils::visualization::GenerateColor(time(0), color);
                double alpha = 0.75;
                cv::Vec3b color_to_draw;
                if (observation.detection_avalible())
                    color_to_draw = cv::Vec3b(0, 255, 0);
                else
                    color_to_draw = cv::Vec3b(0, 255, 255);

                const auto &inds = observation.pointcloud_indices();
                for (const auto &ind:inds) {
                    int x, y;
                    SUN::utils::UnravelIndex(ind, cam.width(), &x, &y);
                    ref_img.at<cv::Vec3b>(y, x) = alpha * color_to_draw + (1.0f - alpha) * ref_img.at<cv::Vec3b>(y, x);
                }

                // Draw pos
                auto proj_pt = cam.CameraToImage(observation.footpoint());
                auto cv_proj_pt = cv::Point(proj_pt[0], proj_pt[1]);
                cv::circle(ref_img, cv_proj_pt, 3.0, cv::Scalar(255, 0, 0), -1);

                // Draw velocity measurement
                Eigen::Vector4d vel_end_pt;
                vel_end_pt.head<3>() = observation.footpoint().head<3>() + observation.velocity() * 0.1;
                auto proj_vel_end_pt = cam.CameraToImage(vel_end_pt);
                auto proj_vel_end_pt_cv = cv::Point(proj_vel_end_pt[0], proj_vel_end_pt[1]);
                SUN::utils::visualization::ArrowedLine(cv_proj_pt, proj_vel_end_pt_cv, cv::Scalar(0, 0, 255), ref_img,
                                                       2, 8, 0, 0.1);
            }

            void DrawObservationByID(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam,
                                     cv::Mat &ref_img, int index) {
                uint8_t r, g, b;
                SUN::utils::visualization::GenerateColor(index, r, g, b);
                const auto bb2d = observation.bounding_box_2d();
                SUN::utils::visualization::DrawObjectFilled(observation.pointcloud_indices(),
                                                            observation.bounding_box_2d(), cv::Vec3b(r, g, b), 0.5,
                                                            ref_img);
            }

            void DrawObservationTyped(const GOT::tracking::Observation &observation, const SUN::utils::Camera &cam,
                                      cv::Mat &ref_img, int index,
                                      const std::map<int, std::string> &category_map) {

                int categ_id = observation.detection_category();
                uint8_t r, g, b;
                SUN::utils::visualization::GenerateColor(categ_id, r, g, b);
                const std::string category_str = category_map.at(categ_id);
                SUN::utils::visualization::DrawObjectFilled(observation.pointcloud_indices(),
                                                            observation.bounding_box_2d(), cv::Vec3b(b, g, r), 0.5,
                                                            ref_img);

                cv::putText(ref_img, category_str,
                            cv::Point2d(observation.bounding_box_2d()[0] + 5,
                                        observation.bounding_box_2d()[1] + 20),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(255, 0, 0), 1);
            }
        }
    }
}