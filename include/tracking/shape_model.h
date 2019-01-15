/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep, Dirk Klostermann, Denis Mitzel (osep, mitzel -at- vision.rwth-aachen.de, klostermann -at- rwth-aachen.de)

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

#ifndef GOT_SHAPE_MODEL_H
#define GOT_SHAPE_MODEL_H

// pcl
#include <pcl/common/common.h>

// Eigen
#include <Eigen/Core>

// std
#include <memory>

namespace SUN { namespace utils { class Camera; }}


namespace GOT {
    namespace tracking {
        /**
           * @brief Base class for 3d object-shape representation of the tracked object. Just an interface.
           * @author Aljosa (osep@vision.rwth-aachen.de)
           */
        class ShapeModel {
        public:
            typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;

            /**
               * @brief Updated model with new measurements. Points assumed in world coord. frame. Need to update member integrated_points.
               * @param new_measurement_points New measurement (3d points), in world coordinate frame.
               * @param Rt Rigid transformation, from model to current measurement.
               * @return True, if update is successful, false otherwise.
               */
            virtual bool UpdateModel(PointCloudRGBA::ConstPtr new_measurement_points, const Eigen::Matrix4d &Rt, const SUN::utils::Camera &camera)=0;

            /**
               * @brief
               * @param first_measurement_points First measurement (3d points), in world coordinate frame.
               * @return True, if init is successful, false otherwise.
               */
            virtual bool InitModel(PointCloudRGBA::ConstPtr first_measurement_points, double orientation, const SUN::utils::Camera &camera)=0;

            virtual void RepresentationToFile(const std::string &filename) const =0;

            virtual double ProposedOrientation() const;

            virtual std::vector<double> weights() const;
            virtual std::vector<double> variances() const;

            const std::vector<Eigen::Matrix4d>& transformations() const; // Pose-transformations (full-history).


            // Setters / Getters
            PointCloudRGBA::ConstPtr integrated_points() const;
            Eigen::Vector4d last_centroid_pose() const;

            // Typedefs
            typedef std::shared_ptr<const ShapeModel> ConstPtr;
            typedef std::shared_ptr<ShapeModel> Ptr;

            std::vector<Eigen::Vector4d> centroid_poses_; // Centroid of the inlier detections (full-history).
            std::vector<Eigen::Matrix4d> transformations_; // Pose-transformations (full-history).

            PointCloudRGBA::Ptr integrated_points_;

        protected:
            std::vector<double> weights_;
            std::vector<double> variances_;

            //std::vector<Eigen::Vector4d> centroid_poses_; // Centroid of the inlier detections (full-history).
            //std::vector<Eigen::Matrix4d> transformations_; // Pose-transformations (full-history).
        };
    }
}

#endif // GOT_SHAPE_MODEL_H
