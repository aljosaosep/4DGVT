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

#include <tracking/kalman_filter.h>

// eigen
#include <Eigen/LU>

namespace GOT {
    namespace tracking {

        // -------------------------------------------------------------------------------
        // +++ Implementation: Kalman Filter Base +++
        // -------------------------------------------------------------------------------

        KalmanFilter::KalmanFilter(const Parameters &params) : state_vector_dim_(0), params_(params) {

        }

        void KalmanFilter::ComputePrediction(const Eigen::VectorXd &u, Eigen::VectorXd &x_prio,
                                             Eigen::MatrixXd &P_prio) const {
            x_prio = A_ * x_ + u; // Linear

            // Project covariance ahead
            P_prio = A_ * P_ * A_.transpose() + G_;
        }

        void KalmanFilter::Prediction() {
            assert(this->state_vector_dim_ == A_.rows());
            assert(this->state_vector_dim_ == A_.cols());
            assert(this->state_vector_dim_ == G_.rows());
            assert(this->state_vector_dim_ == G_.cols());

            Eigen::VectorXd u_zero, x_prio;
            Eigen::MatrixXd P_prio;
            u_zero.setZero(x_.size());
            this->ComputePrediction(u_zero, x_prio, P_prio);

            if (x_prio.size() != this->params_.state_vector_dim_) {
                return;
            }

            x_ = x_prio;
            P_ = P_prio;
        }

        void KalmanFilter::Prediction(const Eigen::VectorXd &u) {
            assert(this->state_vector_dim_ == A_.rows());
            assert(this->state_vector_dim_ == A_.cols());
            assert(this->state_vector_dim_ == G_.rows());
            assert(this->state_vector_dim_ == G_.cols());

            Eigen::VectorXd x_prio;
            Eigen::MatrixXd P_prio;
            this->ComputePrediction(u, x_prio, P_prio);

            if (x_prio.size() != this->params_.state_vector_dim_) {
                return;
            }

            x_ = x_prio;
            P_ = P_prio;
        }

        /// H is the observation model matrix (residual = z_t - H*x)
        void KalmanFilter::Correction(const Eigen::VectorXd z_t, const Eigen::MatrixXd &observation_cov_t,
                                      const Eigen::MatrixXd &H) {
            Eigen::MatrixXd H_transposed = H.transpose();

            // Compute Kalman gain
            Eigen::MatrixXd HPH_plus_obs_cov = (H * P_ * H_transposed) + observation_cov_t;
            Eigen::MatrixXd K = P_ * H_transposed * HPH_plus_obs_cov.inverse();

            // Update the estimate
            Eigen::VectorXd measurement_residual = z_t - H * x_;

            x_ = x_ + K * measurement_residual; //(z_t - H_*x_);

            // Update the covariance
            Eigen::MatrixXd KH = K * H;
            P_ = (Eigen::MatrixXd::Identity(KH.rows(), KH.cols()) - KH) * P_;
        }

        // Setters / Getters
        const Eigen::VectorXd &KalmanFilter::x() const {
            return x_;
        }

        void KalmanFilter::set_x(const Eigen::VectorXd &state) {
            this->x_ = state;
        }

        const Eigen::MatrixXd &KalmanFilter::P() const {
            return this->P_;
        }

        void KalmanFilter::set_G(const Eigen::MatrixXd &G) {
            this->G_ = G;
        }

        const Eigen::MatrixXd &KalmanFilter::G() const {
            return this->G_;
        }

        const KalmanFilter::Parameters KalmanFilter::parameters() const {
            return params_;
        }

        void KalmanFilter::set_A(const Eigen::MatrixXd &A) {
            this->A_ = A;
        }

        const Eigen::MatrixXd &KalmanFilter::A() const {
            return this->A_;
        }
    }
}