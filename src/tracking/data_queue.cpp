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

// tracking lib
#include <tracking/data_queue.h>

// pcl
#include <pcl/io/io.h>

// utils
#include "sun_utils/camera.h"

namespace GOT {
    namespace tracking {
        DataQueue::DataQueue(int temporal_window_size) {
            temporal_window_size_ = temporal_window_size;
            current_frame_ = 0;
        }

        bool DataQueue::FrameIndexToQueueIndex(const int frame, int &queue_index) const {
            queue_index = this->scene_cloud_queue_.size() - (current_frame_ - frame) - 1;
            if (queue_index >= 0 && queue_index < scene_cloud_queue_.size()) {
                return true;
            }
            return false;
        }

        SUN::utils::Camera DataQueue::GetCamera(int frame, bool &lookup_success) const {
            int queue_lookup_index = -1;
            lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);
            if (lookup_success) {
                return camera_queue_.at(queue_lookup_index);
            }
            return SUN::utils::Camera();
        }

        bool DataQueue::GetCamera(int frame, SUN::utils::Camera &cam) const {
            int queue_lookup_index = -1;
            bool lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);
            if (lookup_success) {
                cam = camera_queue_.at(queue_lookup_index);
                return true;
            }

            return false;
        }

        PointCloudRGBA::ConstPtr DataQueue::GetPointCloud(int frame) const {
            int queue_lookup_index = -1;
            bool lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);

            if (lookup_success) {
                return this->scene_cloud_queue_.at(queue_lookup_index);
            }

            return nullptr;
        }

        void DataQueue::AddNewMeasurements(int frame, PointCloudRGBA::ConstPtr reference_cloud,
                                           const SUN::utils::Camera &camera) {
            // Make a new copy of the point cloud
            PointCloudRGBA::Ptr cloud_copy(new PointCloudRGBA);
            pcl::copyPointCloud(*reference_cloud, *cloud_copy);

            if (scene_cloud_queue_.size() >=
                static_cast<unsigned int>(this->temporal_window_size_)) { // Make sure queue size is never > T!
                // Point Cloud
                scene_cloud_queue_.pop_front();
                scene_cloud_queue_.push_back(cloud_copy);
                camera_queue_.pop_front();
                camera_queue_.push_back(camera);
            } else {
                scene_cloud_queue_.push_back(cloud_copy);
                camera_queue_.push_back(camera);
            }

            current_frame_ = frame;
        }

        void DataQueue::AddNewObservations(const std::vector<Observation> &observations) {
            if (this->observation_queue_.size() >= static_cast<unsigned int>(this->temporal_window_size_)) {
                observation_queue_.pop_front();
                observation_queue_.push_back(observations);
            } else {
                observation_queue_.push_back(observations);
            }
        }


        std::vector<Observation> DataQueue::GetObservations(int frame, bool &lookup_success) const {
            int queue_lookup_index = -1;
            lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);

            if (lookup_success) {
                return observation_queue_.at(queue_lookup_index);
            }
            return std::vector<Observation>();
        }

        bool DataQueue::GetInlierObservation(int frame, int inlier_index, Observation &obs) const {
            bool ret_val = false;
            int queue_lookup_index = -1;
            bool lookup_success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);

            if (lookup_success) {
                const auto &obs_vec = observation_queue_.at(queue_lookup_index);
                if (obs_vec.size() > inlier_index) {
                    obs = obs_vec.at(inlier_index);
                    ret_val = true;
                }
            }

            return ret_val;
        }

        void DataQueue::GetEgoEstimate(int frame, Eigen::Matrix4d &ego_estimate, bool &success) const {
            int queue_lookup_index = -1;
            success = this->FrameIndexToQueueIndex(frame, queue_lookup_index);
            if (success) {
                ego_estimate = ego_transformations_.at(queue_lookup_index);
            }
        }

        void DataQueue::AddEgoEstimate(const Eigen::Matrix4d &ego_estimate) {
            if (this->ego_transformations_.size() >= static_cast<unsigned int>(this->temporal_window_size_)) {
                ego_transformations_.pop_front();
                ego_transformations_.push_back(ego_estimate);
            } else {
                ego_transformations_.push_back(ego_estimate);
            }
        }
    }
}