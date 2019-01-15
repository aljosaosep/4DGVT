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

#include "kalman_filter_const_velocity_const_elevation.h"

namespace GOT {
    namespace tracking {

        // Interface
        Eigen::Vector3d ConstantVelocitySize3DKalmanFilter::GetSize3d() const {
            return x_.segment(5, 3);
        }

        Eigen::Vector2d ConstantVelocitySize3DKalmanFilter::GetPoseGroundPlane() const {
            return Eigen::Vector2d(x_[0], x_[2]);
        }

        Eigen::Vector2d ConstantVelocitySize3DKalmanFilter::GetVelocityGroundPlane() const {
            return x_.segment(3, 2);
        }

        Eigen::Matrix2d ConstantVelocitySize3DKalmanFilter::GetPoseCovariance() const {
            Eigen::Matrix2d pos_gp_cov;
            pos_gp_cov(0, 0) = P_(0, 0);
            pos_gp_cov(1, 1) = P_(2, 2);
            pos_gp_cov(0, 1) = P_(0, 2);
            pos_gp_cov(1, 0) = P_(2, 0);
            return pos_gp_cov;
        }

        Eigen::Matrix3d ConstantVelocitySize3DKalmanFilter::GetSizeCovariance() const {
            return P_.block(5, 5, 3, 3);
        }

        Eigen::Matrix2d ConstantVelocitySize3DKalmanFilter::GetVelocityCovariance() const {
            return P_.block(3, 3, 2, 2);
        }

        void ConstantVelocitySize3DKalmanFilter::Init(const Eigen::VectorXd &x_0,
                                                      const Eigen::MatrixXd &P_0) {
            // Set params
            this->state_vector_dim_ = params_.state_vector_dim_;
            const double dt = params_.dt; // Assume delta_t parameter is given

            assert(this->state_vector_dim_ == x_0.size());
            assert(this->state_vector_dim_ == P_0.rows());
            assert(this->state_vector_dim_ == P_0.cols());


            // Set initial state
            this->x_ = x_0;
            this->P_ = P_0;

            //! Set up transition matrix A and obsv. matrix H
            A_.setIdentity(state_vector_dim_, state_vector_dim_);
            A_(0, 3) = dt;
            A_(2, 4) = dt;

            G_.setIdentity(state_vector_dim_, state_vector_dim_);
            G_ *= 0.1 * 0.1; // Default process noise.
        }

        Eigen::VectorXd ConstantVelocitySize3DKalmanFilter::NonlinearStateProjection() const {
            Eigen::VectorXd x_prio = A_ * x_; // OK, its only linear ...
            return x_prio;
        }

        Eigen::MatrixXd ConstantVelocitySize3DKalmanFilter::LinearizeTransitionMatrix() const {
            // Update the transition matrix
            Eigen::MatrixXd A_linearized = this->A_; // OK, its only linear ...
            return A_linearized;
        }

        Eigen::VectorXd ConstantVelocitySize3DKalmanFilter::compute_measurement_residual(const Eigen::VectorXd z_t,
                                                                                         const Eigen::MatrixXd &H) {
            return z_t - H * x_; // OK, its only linear ...
        }

        ConstantVelocitySize3DKalmanFilter::ConstantVelocitySize3DKalmanFilter(
                const ConstantVelocitySize3DKalmanFilter::Parameters &params) : params_(params), ExtendedKalmanFilter(
                static_cast<KalmanFilter::Parameters>(params)) {
        }

        Eigen::Vector3d ConstantVelocitySize3DKalmanFilter::GetFullPoseGroundPlane() const {
            return x_.head<3>();
        }
    }
}

