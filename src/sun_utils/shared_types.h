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

#ifndef GOT_SHARED_TYPES_H
#define GOT_SHARED_TYPES_H

#include "external/maskApi.h"

#include <cassert>
#include <vector>
#include <map>

namespace SUN {
    namespace shared_types {

        enum CategoryTypeKITTI {
            CAR = 0,
            VAN = 1,
            TRUCK = 2,
            PEDESTRIAN = 3,
            PERSON_SITTING = 4,
            CYCLIST = 5,
            TRAM = 6,
            MISC = 7,
            DONT_CARE = 8,
            UNKNOWN_TYPE = 9,
            STATIC_BACKGROUND = 10,
            NOISE = 11,
        };

        namespace category_maps {
            const std::map<int, std::string> kitti_map = {
                    {0,  "car"},
                    {1,  "van"},
                    {2,  "truck"},
                    {3,  "pedestrian"},
                    {4,  "person_sitting"},
                    {5,  "cyclist"},
                    {6,  "tram"},
                    {7,  "misc"},
                    {8,  "dont_care"},
                    {9,  "unknown_type"},
                    {10, "static_background"},
                    {11, "noise"},
            };

            const std::map<int, std::string> coco_map = {
                    {0,  "background"},
                    {1,  "person"},
                    {2,  "bicycle"},
                    {3,  "car"},
                    {4,  "motorcycle"},
                    {5,  "airplane"},
                    {6,  "bus"},
                    {7,  "train"},
                    {8,  "truck"},
                    {9,  "boat"},
                    {10, "traffic light"},
                    {11, "fire hydrant"},
                    {12, "stop sign"},
                    {13, "parking meter"},
                    {14, "bench"},
                    {15, "bird"},
                    {16, "cat"},
                    {17, "dog"},
                    {18, "horse"},
                    {19, "sheep"},
                    {20, "cow"},
                    {21, "elephant"},
                    {22, "bear"},
                    {23, "zebra"},
                    {24, "giraffe"},
                    {25, "backpack"},
                    {26, "umbrella"},
                    {27, "handbag"},
                    {28, "tie"},
                    {29, "suitcase"},
                    {30, "frisbee"},
                    {31, "skis"},
                    {32, "snowboard"},
                    {33, "sports ball"},
                    {34, "kite"},
                    {35, "baseball bat"},
                    {36, "baseball glove"},
                    {37, "skateboard"},
                    {38, "surfboard"},
                    {39, "tennis racket"},
                    {40, "bottle"},
                    {41, "wine glass"},
                    {42, "cup"},
                    {43, "fork"},
                    {44, "knife"},
                    {45, "spoon"},
                    {46, "bowl"},
                    {47, "banana"},
                    {48, "apple"},
                    {49, "sandwich"},
                    {50, "orange"},
                    {51, "broccoli"},
                    {52, "carrot"},
                    {53, "hot dog"},
                    {54, "pizza"},
                    {55, "donut"},
                    {56, "cake"},
                    {57, "chair"},
                    {58, "couch"},
                    {59, "potted plant"},
                    {60, "bed"},
                    {61, "dining table"},
                    {62, "toilet"},
                    {63, "tv"},
                    {64, "laptop"},
                    {65, "mouse"},
                    {66, "remote"},
                    {67, "keyboard"},
                    {68, "cell phone"},
                    {69, "microwave"},
                    {70, "oven"},
                    {71, "toaster"},
                    {72, "sink"},
                    {73, "refrigerator"},
                    {74, "book"},
                    {75, "clock"},
                    {76, "vase"},
                    {77, "scissors"},
                    {78, "teddy bear"},
                    {79, "hair drier"},
                    {80, "toothbrush"},

            };
        }

        struct CompressedMask {
            std::string rle_string_;
            int w_;
            int h_;

            CompressedMask(const std::vector<int> &inds, int image_width, int image_height) {
                FromIndices(inds, image_width, image_height);
            }

            CompressedMask(const std::string &rle_str, int width, int height) {
                this->rle_string_ = rle_str;
                this->w_ = width;
                this->h_ = height;
            }

            CompressedMask() {
                w_ = h_ = 0;
            }

            std::vector<int> SegmentationMaskToIndices(byte *mask, int width, int height) const {
                assert(mask != nullptr);
                std::vector<int> inds;
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        byte val = mask[height * x + y];
                        if (val > 0) {
                            int ind = y * width + x;
                            inds.push_back(ind);
                        }
                    }
                }

                return inds;
            }

            void FromIndices(const std::vector<int> &inds, int image_width, int image_height) {
                w_ = image_width;
                h_ = image_height;

                RLE *R = new RLE;
                siz w = image_width, h = image_height; //size.at(1), h = size.at(0);

                // Inds -> mask
                const siz imsize = w * h;
                auto M = new byte[imsize];

                // Init with 0
                for (int i = 0; i < imsize; i++) {
                    int x = i % image_width;
                    int y = i / image_width;
                    if (x >= 0 && y >= 0 && x < image_width && y < image_height)
                        M[image_height * x + y] = static_cast<byte>(0);
                }

                // Switch pixels on
                for (auto ind:inds) {
                    int x = ind % image_width;
                    int y = ind / image_width;
                    if (x >= 0 && y >= 0 && x < image_width && y < image_height)
                        M[image_height * x + y] = static_cast<byte>(255); // 255 or 1?
                }

                // Encode!
                rleEncode(R, M, h, w, 1); //w, h, 1);

                //rle
                char *rle_str = rleToString(R);
                this->rle_string_ = std::string(rle_str);

                delete[] M;
                rleFree(R);
            }

            std::vector<int> GetIndices() const {
                /// Uncompress the mask data
                RLE *R = new RLE;
                auto rle_copy = rle_string_;
                char *compressed_str = &rle_copy[0u];
                rleFrString(R, compressed_str, (siz) h_, (siz) w_);
                const siz imsize = R->w * R->h;
                auto M = new byte[imsize];
                rleDecode(R, M, 1);

                /// Turn binary segm. mask to indices
                std::vector<int> inds = SegmentationMaskToIndices(M, w_, h_);

                delete[] M;
                M = nullptr;
                rleFree(R);

                return inds;
            }

            RLE *GetRLERepresentation() const {
                RLE *R = new RLE;
                auto rle_str_copy = this->rle_string_;
                char *compressed_str = &rle_str_copy[0u];
                rleFrString(R, compressed_str, (siz) h_, (siz) w_);
                return R;
            }

            void GetBoundingBox(int &u, int &v, int &w, int &h) const {
                auto R = this->GetRLERepresentation();
                BB bbox = new double[4];
                rleToBbox(R, bbox, (siz) 1);
                u = bbox[0];
                v = bbox[1];
                w = bbox[2];
                h = bbox[3];
                delete[] bbox;
                rleFree(R);
            }

            double IoU(const CompressedMask &the_other_compressed_mask) const {
                auto thisRLE = GetRLERepresentation();
                auto otherRLE = the_other_compressed_mask.GetRLERepresentation();
                double iou;
                rleIou(thisRLE, otherRLE, (siz) 1, (siz) 1, (byte *) 0, &iou);
                rleFree(thisRLE);
                rleFree(otherRLE);
                return iou;
            }

            double IoM(const CompressedMask &the_other_compressed_mask) const {
                auto thisRLE = GetRLERepresentation();
                auto otherRLE = the_other_compressed_mask.GetRLERepresentation();

                uint int_area;
                uint area_this;
                uint area_other;

                rleArea(thisRLE, (siz) 1, &area_this);
                rleArea(otherRLE, (siz) 1, &area_other);

                rleMerge(thisRLE, otherRLE, (siz) 1, 1); // I assume the result gets stored to the otherRLE

                rleArea(thisRLE, (siz) 1, &int_area);

                printf("Area 1: %d, area 2: %d, area int: %d\r\n", area_this, area_other, int_area);

                rleFree(thisRLE);
                rleFree(otherRLE);

                return int_area /
                       std::max(1.0, std::min(static_cast<double>(area_this), static_cast<double>(area_other)));
            }

            CompressedMask GetIntersectionMask(const CompressedMask &the_other_compressed_mask) const {
                auto thisRLE = GetRLERepresentation();
                auto otherRLE = the_other_compressed_mask.GetRLERepresentation();
                rleMerge(thisRLE, otherRLE, (siz) 1, 1);
                char *rle_str = rleToString(otherRLE);
                CompressedMask mask_out;
                mask_out.rle_string_ = std::string(rle_str);
                mask_out.w_ = this->w_;
                mask_out.h_ = this->h_;

                rleFree(thisRLE);
                rleFree(otherRLE);
                return mask_out;
            }
        };

    }
}

#endif //GOT_SHARED_TYPES_H
