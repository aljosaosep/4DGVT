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

#include "CAMOT_dynamics_handler.h"
#include "kalman_filter_const_velocity_const_elevation.h"
#include "sun_utils/ground_model.h"

namespace GOT {
    namespace tracking {

        auto observation_top_point = [](const Observation &observation,
                                        const SUN::utils::Camera &camera) -> Eigen::Vector3d {
            Eigen::Vector3d ground_plane_normal = camera.ground_model()->Normal(observation.footpoint().head<3>());
            Eigen::Vector3d top_point = observation.bounding_box_3d().head<3>() /*+*/+
                                        observation.bounding_box_3d()[4] / 2.0 * ground_plane_normal;
            return top_point;
        };

        auto extend_bbox3D_to_ground = [](const Observation &observation,
                                                        const SUN::utils::Camera &camera) -> Eigen::Vector3d {
            Eigen::Vector3d top_point = observation_top_point(observation, camera);
            double hidden_height = camera.ground_model()->DistanceToGround(top_point);
            Eigen::Vector3d size_out = observation.bounding_box_3d().segment(3, 3);
            size_out[1] = hidden_height; // Replace height from segment by 'latent' height
            return size_out;
        };

        auto position_camera_to_world = [](const Observation &observation,
                                                  const SUN::utils::Camera &camera) -> Eigen::Vector3d {
            Eigen::Vector4d footpoint_world = camera.CameraToWorld(observation.footpoint());
            return footpoint_world.head<3>();
        };

        auto covariance_camera_to_world = [](const Observation &observation,
                                                   const SUN::utils::Camera &camera) -> Eigen::Matrix3d {
            return camera.R() * observation.covariance3d() * camera.R().transpose();
        };


        auto project_covariance_to_2D = [](const Eigen::Matrix3d &cov_3d) -> Eigen::Matrix2d {
            Eigen::Matrix2d cov_2d;
            cov_2d << cov_3d(0, 0), cov_3d(0, 2), cov_3d(2, 0), cov_3d(2, 2);
            return cov_2d;
        };

        CAMOTDynamicsHandler::CAMOTDynamicsHandler(const po::variables_map &params) : DynamicsModelHandler(params) {
            state_dim_ = 8;
            this->InitializeMatrices();
        }

        void
        CAMOTDynamicsHandler::InitializeState(const SUN::utils::Camera &camera, const Observation &obs, bool forward,
                                              Hypothesis &hypo) {

            /// Create filter object
            ConstantVelocitySize3DKalmanFilter::Parameters params;
            params.dt = this->parameters_["dt"].as<double>();
            params.state_vector_dim_ = this->state_dim_;
            hypo.kalman_filter() = std::shared_ptr<ConstantVelocitySize3DKalmanFilter>(
                    new ConstantVelocitySize3DKalmanFilter(params));

            /// Ground-plane pose
            Eigen::Vector3d footpoint_world = position_camera_to_world(obs, camera);
            Eigen::Matrix3d pose_cov_world = covariance_camera_to_world(obs, camera);
            Eigen::Matrix2d pose_2d_cov_world = project_covariance_to_2D(pose_cov_world);

            /// Physical size
            Eigen::Vector3d size_3d = extend_bbox3D_to_ground(obs, camera);
            size_3d[0] = std::min(4.0, size_3d[0]);
            size_3d[1] = std::min(4.0, size_3d[1]);
            size_3d[2] = std::min(4.0, size_3d[2]);

            if (std::isnan(size_3d[0]) || std::isnan(size_3d[1]) || std::isnan(size_3d[2])) {
                assert(false);
                printf("Error, nans in size!\r\n");
            }

            // Velocity
            bool use_velocity_measurement = true;
            if (std::isnan(obs.velocity()[0])) {
                use_velocity_measurement = false;
            }
            Eigen::Vector3d velocity_world;
            velocity_world.setZero();
            if (use_velocity_measurement) {
                if (forward)
                    velocity_world = camera.R() * obs.velocity();
                else
                    velocity_world = camera.R() * (-1.0 * obs.velocity());
            }

            /// Initialize the state
            Eigen::VectorXd kalman_init_state(params.state_vector_dim_);
            kalman_init_state.setZero();
            kalman_init_state << footpoint_world,
                    velocity_world[0], velocity_world[2],
                    size_3d;

            /// Initial state unc.
            Eigen::MatrixXd P0_measured;
            P0_measured.setZero(params.state_vector_dim_, params.state_vector_dim_);
            P0_measured.block(0, 0, 3, 3) = pose_cov_world;

            if (use_velocity_measurement)
                P0_measured.block(3, 3, 2, 2) = 2.0 * pose_2d_cov_world;
            else {
                //P0_measured.block(3, 3, 2, 2) = Eigen::Matrix2d::Identity() * 50.0;
                P0_measured(3, 3) = 1.0;
                P0_measured(4, 4) = 45.0; // Huuuge velocity unc. in Z-dir
            }

            // w,h,d -- no idea how to set the uncertainties. TODO! These factors seem very wrong.
            P0_measured(5, 5) = 0.2 * 0.2;
            P0_measured(6, 6) = 0.2 * 0.2;
            P0_measured(7, 7) = 0.5 * 0.5;

            hypo.kalman_filter()->Init(kalman_init_state, P0_measured);
            hypo.kalman_filter()->set_G(G_generic_);
        }

        void CAMOTDynamicsHandler::ApplyTransition(const SUN::utils::Camera &camera, const Eigen::VectorXd &u,
                                                   Hypothesis &hypo) {
            // Simply do the prediction.
            hypo.kalman_filter()->Prediction(u);

            // Make sure 2d-bbox size doesn't 'drift' below 10x10 px
            Eigen::VectorXd kf_state_after_projection = hypo.kalman_filter_const()->x();
        }

        void
        CAMOTDynamicsHandler::ApplyCorrection(const SUN::utils::Camera &camera, const Observation &obs, bool forward,
                                              Hypothesis &hypo) {


            bool use_velocity_measurement = true;
            if (std::isnan(obs.velocity()[0])) {
                use_velocity_measurement = false;
            }

            Eigen::Vector3d velocity_world;
            if (use_velocity_measurement) {
                if (forward)
                    velocity_world = camera.R() * obs.velocity();
                else
                    velocity_world = camera.R() * (-1.0 * obs.velocity());
            }


            /// Ground-plane pose and velocity. Perform projection camera->world, take into account time arrow!
            Eigen::Vector3d footpoint_world = position_camera_to_world(obs, camera);
            Eigen::Matrix3d pose_cov_world = covariance_camera_to_world(obs, camera);
            Eigen::Matrix2d pose_2d_cov_world = project_covariance_to_2D(pose_cov_world);

            /// Physical size
            Eigen::Vector3d size_3d = extend_bbox3D_to_ground(obs, camera);
            size_3d[0] = std::min(4.0, size_3d[0]);
            size_3d[1] = std::min(4.0, size_3d[1]);
            size_3d[2] = std::min(4.0, size_3d[2]);

            if (std::isnan(size_3d[0]) || std::isnan(size_3d[1]) || std::isnan(size_3d[2])) {
                assert(false);
                printf("Error, nans in size!\r\n");
            }

            double unc_W = 0.2;
            double unc_H = 0.2;
            double unc_L = 0.5;

            // We observe pos. on ground-plane, size and bounding-box 2D
            if (!use_velocity_measurement) {
                Eigen::VectorXd measurement;
                measurement.setZero(6);
                measurement << footpoint_world, size_3d;

                // Maps state to the measurement
                Eigen::MatrixXd H;
                H.setZero(6, 8);
                H.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
                H.block(3, 5, 3, 3) = Eigen::MatrixXd::Identity(3, 3);

                Eigen::MatrixXd R;
                R.setZero(6, 6);
                R.block(0, 0, 3, 3) = pose_cov_world;
                // w,h,d -- no idea how to set the uncertainties. TODO!
                R(3, 3) = unc_W * unc_W;
                R(4, 4) = unc_H * unc_H;
                R(5, 5) = unc_L * unc_L;
                // --

                hypo.kalman_filter()->Correction(measurement, R, H);
            } else {
                // Use velocity measurement, too
                Eigen::VectorXd measurement;
                measurement.setZero(8);
                measurement << footpoint_world,
                        velocity_world[0], velocity_world[2],
                        size_3d;

                // Maps state to the measurement
                Eigen::MatrixXd H = Eigen::MatrixXd::Identity(8, 8);

                Eigen::MatrixXd R;
                R.setZero(8, 8);
                R.block(0, 0, 3, 3) = pose_cov_world;
                R.block(3, 3, 2, 2) = 2.0 * pose_2d_cov_world;
                // w,h,d -- no idea how to set the uncertainties.
                R(5, 5) = unc_W * unc_W;
                R(6, 6) = unc_H * unc_H;
                R(7, 7) = unc_L * unc_L;
                // --

                hypo.kalman_filter()->Correction(measurement, R, H);
            }
        }

        void CAMOTDynamicsHandler::InitializeMatrices() {
            G_generic_pos_variance_ = Eigen::Vector3d(0.5, 0.1, 0.5);
            G_generic_velocity_variance_ = Eigen::Vector2d(1.0, 1.0);

            // 'System matrix'
            G_generic_.setZero(state_dim_, state_dim_);
            G_generic_(0, 0) = G_generic_pos_variance_[0] * G_generic_pos_variance_[0];
            G_generic_(1, 1) = G_generic_pos_variance_[1] * G_generic_pos_variance_[1];
            G_generic_(2, 2) = G_generic_pos_variance_[2] * G_generic_pos_variance_[2];
            // --
            G_generic_(3, 3) = G_generic_velocity_variance_[0] * G_generic_velocity_variance_[0];
            G_generic_(4, 4) = G_generic_velocity_variance_[1] * G_generic_velocity_variance_[1];
            // --
            G_generic_(5, 5) = 0.05 * 0.05; // w
            G_generic_(6, 6) = 0.05 * 0.05; // h
            G_generic_(7, 7) = 0.05 * 0.05; // d
            // --
        }
    }
}