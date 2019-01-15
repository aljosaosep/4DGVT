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

#ifndef GOT_CAMOT_TRACKER_H
#define GOT_CAMOT_TRACKER_H

// tracking lib
#include <tracking/multi_object_tracker_base.h>

// CAMOT
#include "CAMOT_dynamics_handler.h"

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {

            std::vector<Hypothesis> HypothesisNonMaximaSuppression(const std::vector<Hypothesis> &input_hypos, double iou_threshold=0.8);

            class CAMOTTracker : public MultiObjectTracker3DBase {
            public:
                // -------------------------------------------------------------------------------
                // +++ CONSTRUCTOR +++
                // -------------------------------------------------------------------------------
                CAMOTTracker(const po::variables_map &params) : MultiObjectTracker3DBase(params) {
                    this->parameter_map_ = params;
                    dynamics_model_handler_.reset(new CAMOTDynamicsHandler(params));
                }

                /**
                 * @brief The tracker 'main' function
                 */
                void ProcessFrame(DataQueue::ConstPtr detections, int current_frame);


                // -------------------------------------------------------------------------------
                // +++ TRACKLET HYPOTHESES HANDLING +++
                // -------------------------------------------------------------------------------

                int AdvanceHypo(DataQueue::ConstPtr detections, int reference_frame, bool is_forward, Hypothesis &ref_hypo, bool allow_association);

                /**
                 * @brief Starts new tracklets.
                 */
                std::vector <Hypothesis> StartNewHypotheses(DataQueue::ConstPtr detections, int current_frame);

                /**
                 * @brief Extends 'old' tracklets.
                 */

                std::vector<Hypothesis> ExtendHypotheses(DataQueue::ConstPtr detections, int current_frame);

                /**
                 * @brief Initializes the tracklet.
                 */
                void HypothesisInit(DataQueue::ConstPtr detections, bool is_forward_update,
                                    int inlier_index, int current_frame,
                                    Hypothesis &hypo);

                /**
                 * @brief Updates tracklet entries.
                 */
                void HypothesisUpdate(DataQueue::ConstPtr detections, bool is_forward_update,
                                      std::tuple<int, double, std::vector<double>> data_assoc_context, int current_frame,
                                      Hypothesis &hypo);

                /**
                 * @brief NEW: mask prediction.
                 */
                std::tuple<pcl::PointCloud<pcl::PointXYZRGBA>, SUN::shared_types::CompressedMask, bool> GetPredictedSegment(int frame, bool forward, DataQueue::ConstPtr detections, const Hypothesis &hypo);
                bool HypoAddPredictedSegment(int frame, bool forward, DataQueue::ConstPtr detections, Hypothesis &hypo);

                // -------------------------------------------------------------------------------
                // +++ OPTIMIZATION +++
                // -------------------------------------------------------------------------------

                /**
                 * @brief Appends all 'valid' active hypos to the 'exported hypos' set,
                 *        this is needed for windowed-CRF.
                 */
                void AppendActiveTrackletsToExported();

                /**
                 * @brief Computes 'global' (or windowed...) unaries for-all hypos.
                 */
                void ComputeUnariesGlobal(std::vector<Hypothesis> &hypos);


                // -------------------------------------------------------------------------------
                // +++ SETTERS +++
                // -------------------------------------------------------------------------------
                // NO SETTERS OH NOES

                // -------------------------------------------------------------------------------
                // +++ GETTERS +++
                // -------------------------------------------------------------------------------
                std::vector<GOT::tracking::Hypothesis> GetConfidentHypotheses() const;
                const std::vector<GOT::tracking::Hypothesis> &exported_tracklets() const;

            protected:
                std::shared_ptr <DynamicsModelHandler> dynamics_model_handler_;
                std::set<int> detection_indices_used_for_extensions_;

                // All terminated tracks will go here
                std::vector<GOT::tracking::Hypothesis> exported_tracklets_;
            };
        }
    }
}

#endif
