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

#ifndef GOT_HYPO_EXPORT_H
#define GOT_HYPO_EXPORT_H

// tracking lib
#include <tracking/hypothesis.h>

namespace GOT {
    namespace tracking {
        namespace CAMOT_tracker {

            // Export track info on per-frame level
            bool SerializeHyposPerFrameJson(const char *filename,
                                            int frame,
                                            const std::vector<GOT::tracking::Hypothesis> &hypos_to_export);

            // Export whole tracks; per sequence.
            bool SerializeHyposJson(const char *filename,
                                    const std::vector<GOT::tracking::Hypothesis> &hypos_to_export,
                                    const std::map<int, std::string> &label_mapper);
        }
    }
}


#endif //GOT_HYPO_EXPORT_H
