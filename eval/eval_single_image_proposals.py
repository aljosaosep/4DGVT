#!/usr/bin/env python3

import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.mask import toBbox
import matplotlib
import argparse
import datetime
import time
import re
import os

IOU_THRESHOLD = 0.5


def load_gt(exclude_classes=(), ignored_sequences=(), prefix_dir_name='oxford_labels',
            dist_thresh=1000.0, area_thresh=10*10):
    gt_jsons = glob.glob("%s/*/*.json"%prefix_dir_name)

    gt = {}
    for gt_json in gt_jsons:
        # Exclude from eval
        matching = [s for s in ignored_sequences if s in gt_json]
        if len(matching) > 0: continue
        anns = json.load(open(gt_json))
        bboxes = []
        for ann in anns:

            if ann["category"] in exclude_classes:
                continue

            extr = ann["extreme_points"]
            assert len(extr) == 4
            x0 = min([c[0] for c in extr])
            y0 = min([c[1] for c in extr])
            x1 = max([c[0] for c in extr])
            y1 = max([c[1] for c in extr])

            # -----------------------------------------------------
            # w = x1 - x0
            # h = y1 - y0
            # if w*h < area_thresh:
            #         continue
            # -----------------------------------------------------

            bboxes.append((x0, y0, x1, y1))
        gt[gt_json] = bboxes
    n_boxes = sum([len(x) for x in gt.values()], 0)
    print("number of gt boxes", n_boxes)
    return gt, n_boxes


def load_gt_categories(exclude_classes=(), ignored_sequences=(), prefix_dir_name='oxford_labels'):
    gt_jsons = glob.glob("%s/*/*.json"%prefix_dir_name)

    gt_cat = {}
    gt_cat_map = {}

    cat_idx = 0
    for gt_json in gt_jsons:

        # Exclude from eval
        matching = [s for s in ignored_sequences if s in gt_json]
        if len(matching) > 0: continue

        anns = json.load(open(gt_json))

        categories = []
        for ann in anns:
            cat_str = ann["category"]
            if cat_str in exclude_classes:
                continue
            categories.append(cat_str)

            if cat_str not in gt_cat_map:
                gt_cat_map[cat_str] = cat_idx
                cat_idx += 1

        gt_cat[gt_json] = categories
    n_boxes = sum([len(x) for x in gt_cat.values()], 0)
    print("number of gt boxes", n_boxes)
    return gt_cat, n_boxes, gt_cat_map


def load_proposals(folder, gt, ignored_sequences=(), score_fnc=lambda prop: prop["score"]):
    proposals = {}
    for filename in gt.keys():
        prop_filename = os.path.join(folder, "/".join(filename.split("/")[-2:]))

        # Exclude from eval
        matching = [s for s in ignored_sequences if s in filename]
        if len(matching) > 0:
            continue

        # Load proposals
        try:
            props = json.load(open(prop_filename))
        except ValueError:
            print ("Error loading json: %s"%prop_filename)
            quit()

        if props is None:
            continue

        props = sorted(props, key=score_fnc, reverse=True)

        if "bbox" in props[0]:
            bboxes = [prop["bbox"] for prop in props]
        else:
            bboxes = [toBbox(prop["segmentation"]) for prop in props]

        # convert from [x0, y0, w, h] (?) to [x0, y0, x1, y1]
        bboxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bboxes]
        proposals[filename] = bboxes

    return proposals


def calculate_ious(bboxes1, bboxes2):
    """
    :param bboxes1: Kx4 matrix, assume layout (x0, y0, x1, y1)
    :param bboxes2: Nx$ matrix, assume layout (x0, y0, x1, y1)
    :return: KxN matrix of IoUs
    """
    min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
    I = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    U = area1[:, np.newaxis] + area2[np.newaxis, :] - I
    assert (U > 0).all()
    IOUs = I / U
    assert (IOUs >= 0).all()
    assert (IOUs <= 1).all()
    return IOUs


def evaluate_proposals(gt, props, n_max_proposals=1000):
    all_ious = [] # ious for all frames
    for img, img_gt in gt.items():
        if len(img_gt) == 0:
            continue
        img_props = props[img]
        gt_bboxes = np.array(img_gt)
        prop_bboxes = np.array(img_props)
        ious = calculate_ious(gt_bboxes, prop_bboxes)

        # pad to n_max_proposals
        ious_padded = np.zeros((ious.shape[0], n_max_proposals))
        ious_padded[:, :ious.shape[1]] = ious[:, :n_max_proposals]
        all_ious.append(ious_padded)
    all_ious = np.concatenate(all_ious)
    if IOU_THRESHOLD is None:
        iou_curve = [0.0 if n_max == 0 else all_ious[:, :n_max].max(axis=1).mean() for n_max in range(0, n_max_proposals + 1)]
    else:
        assert 0 <= IOU_THRESHOLD <= 1
        iou_curve = [0.0 if n_max == 0 else (all_ious[:, :n_max].max(axis=1) > IOU_THRESHOLD).mean() for n_max in
                     range(0, n_max_proposals + 1)]
    return iou_curve


def evaluate_folder(gt, folder, ignored_sequences=(), score_fnc=lambda prop: prop["score"]):
    props = load_proposals(folder, gt, ignored_sequences=ignored_sequences, score_fnc=score_fnc)
    iou_curve = evaluate_proposals(gt, props)

    iou_50 = iou_curve[50]
    iou_100 = iou_curve[100]
    iou_150 = iou_curve[150]
    iou_200 = iou_curve[200]
    iou_700 = iou_curve[700]
    end_iou = iou_curve[-1]

    method_name = os.path.basename(os.path.dirname(folder+"/"))

    print("%s: R50: %1.2f, R100: %1.2f, R150: %1.2f, R200: %1.2f, R700: %1.2f, R_total: %1.2f" %
          (method_name,
           iou_50,
           iou_100,
           iou_150,
           iou_200,
           iou_700,
           end_iou))

    return iou_curve


def title_to_filename(plot_title):
    filtered_title = re.sub("[\(\[].*?[\)\]]", "", plot_title) # Remove the content within the brackets
    filtered_title = filtered_title.replace("_", "").replace(" ", "").replace(",", "_")
    return filtered_title


def make_plot(export_dict, plot_title, x_vals, linewidth=5):
    plt.figure()

    itm = export_dict.items()
    itm = sorted(itm, reverse=True)
    for idx, item in enumerate(itm):
        curve_label = item[0].replace('.', '')
        plt.plot(x_vals[0:700], item[1][0:700], label=curve_label, linewidth=linewidth)

    ax = plt.gca()
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_xticks(np.asarray([25, 100, 200, 300, 500, 700]))
    plt.xlabel("$\#$ proposals")
    plt.ylabel("Recall")
    ax.set_ylim([0.0, 1.0])
    plt.legend()
    plt.grid()
    plt.title(plot_title)


def export_figs(export_dict, plot_title, output_dir, x_vals):
    # Export figs, csv
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, title_to_filename(plot_title) + ".pdf"), bbox_inches='tight')

        # Save to csv
        np.savetxt(os.path.join(output_dir, 'num_objects.csv'), np.array(x_vals), delimiter=',', fmt='%d')
        for item in export_dict.items():
            np.savetxt(os.path.join(output_dir, item[0] + '.csv'), item[1], delimiter=',', fmt='%1.4f')


def evaluate_all_folders_oxford(gt, plot_title, user_specified_result_dir=None, output_dir=None):

    print("----------- Evaluate Oxford Recall -----------")

    # Export dict
    export_dict = {

    }

    # +++ User-specified +++
    user_specified_results = None
    if user_specified_result_dir is not None:
        dirs = os.listdir(user_specified_result_dir)
        dirs.sort()
        for mydir in dirs:
            print("---Eval: %s ---"%mydir)
            user_specified_results = evaluate_folder(gt, os.path.join(user_specified_result_dir, mydir))
            export_dict[mydir] = user_specified_results

    x_vals = range(1001)

    # Plot everything specified via export_dict
    make_plot(export_dict, plot_title, x_vals)

    # Export figs, csv
    export_figs(export_dict, plot_title, output_dir, x_vals)


def eval_recall_oxford(output_dir):

    # +++ Most common categories +++
    print("evaluating car, bike, person, bus:")
    exclude_classes = ("other",)
    gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=FLAGS.labels)

    evaluate_all_folders_oxford(gt, "car, bike, person, and bus (" + str(n_gt_boxes) + " bounding boxes)",
                                output_dir=output_dir,
                                user_specified_result_dir=FLAGS.evaluate_dir)

    # +++ "other" categories +++
    print("evaluating others:")
    exclude_classes = ("car", "bike", "person", "bus")
    gt, n_gt_boxes = load_gt(exclude_classes, prefix_dir_name=FLAGS.labels)

    evaluate_all_folders_oxford(gt, "others (" + str(n_gt_boxes) + " bounding boxes)",
                                output_dir=output_dir,
                                user_specified_result_dir=FLAGS.evaluate_dir)


def main():

    # Matplotlib params
    matplotlib.rcParams.update({'font.size':15})
    matplotlib.rcParams.update({'font.family':'sans-serif'})
    matplotlib.rcParams['text.usetex'] = True

    # Prep output dir (if specified)
    output_dir = None
    if FLAGS.plot_output_dir is not None:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d__%H_%M_%S')

        if FLAGS.do_not_timestamp:
            timestamp = ""

        output_dir = os.path.join(FLAGS.plot_output_dir, timestamp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        eval_recall_oxford(output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Args
    parser.add_argument('--plot_output_dir', type=str, help='Plots output dir.')
    parser.add_argument('--evaluate_dir', type=str, help='Dir containing result files that you want to evaluate')
    parser.add_argument('--labels', type=str, default = 'oxford_labels',
                        help='Specify dir containing the labels')
    parser.add_argument('--do_not_timestamp', action='store_true', help='Dont timestamp output dirs')

    FLAGS = parser.parse_args()
    main()
