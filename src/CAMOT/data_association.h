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

#ifndef GOT_DATA_ASSOCIATION_H
#define GOT_DATA_ASSOCIATION_H

// std
#include <vector>

// tracking lib
#include <tracking/data_queue.h>

// boost
#include <boost/program_options.hpp>

// app
#include "CAMOT_tracker.h"

namespace po = boost::program_options;

// Forward decl.
namespace GOT { namespace tracking { class Observation; }}
namespace GOT { namespace tracking { class Hypothesis; }}
namespace SUN { namespace utils { class Camera; }}


namespace object_tracking = GOT::tracking;
namespace CAMOT = GOT::tracking::CAMOT_tracker;

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {
            namespace data_association {
                std::tuple<std::vector<double>, std::vector<std::vector<double> > > data_association_motion_mask(
                        GOT::tracking::DataQueue::ConstPtr detections,
                        const GOT::tracking::Hypothesis &hypo, int frame_of_association,
                        const po::variables_map &parameters);

            }
        }
    }
}


#endif //GOT_DATA_ASSOCIATION_H
