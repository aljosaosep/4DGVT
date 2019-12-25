import os
import evaluation_tools
import range_analysis_tools as range_eval
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime, time
from tabulate import tabulate

classes = ['car', 'pedestrian']

hook_data = []


def evaluate_folder(c, tr_path):
    # ---------------------------------------------------------------
    #     +++ STANDARD KITTI CLEARMOT +++
    # ---------------------------------------------------------------
    print(' --- Running evaluation: %s --- '%c)
    (hook_initialize, hook_finalize, hook_track) = evaluation_tools.range_analysis()
    tracking_eval = evaluation_tools.trackingEvaluation(t_sha='dummy_sha', cls=c,
                                                        filename_test_mapping=FLAGS.seqmap_path,
                                                        hook_initialize=hook_initialize,
                                                        hook_finalize=hook_finalize,
                                                        hook_track=hook_track)

    # Load ground truth
    try:
        tracking_eval._loadData(FLAGS.labels_path, c, loading_groundtruth=True)
    except IOError:
        raise Exception('Failed to load GT data.')

    # Load tracker
    try:
        tracking_eval._loadData(tr_path, cls=c, loading_groundtruth=False)
    except IOError:
        raise Exception('Failed to load tracker data: %s' % tr_path)

    # Evaluate CLEARMOT
    print('Running CLEARMOT evaluation ...')
    if tracking_eval.compute3rdPartyMetrics():
        print('MOTA: %2.3f' % tracking_eval.MOTA)
        print('MOTP: %2.3f' % tracking_eval.MOTP)
        print('Recall: %2.3f' % tracking_eval.recall)
        print('Precision: %2.3f' % tracking_eval.precision)
        print('Mostly-lost: %2.3f' % tracking_eval.ML)
        print('Partially-lost: %2.3f' % tracking_eval.PT)
        print('Mostly-tracked: %2.3f' % tracking_eval.MT)
        print('ID switches: %d' % tracking_eval.id_switches)
        print('Fragments: %d' % tracking_eval.fragments)

        cmot_dict = {
            "mota": tracking_eval.MOTA,
            "motp": tracking_eval.MOTP,
            "recall": tracking_eval.recall,
            "precision": tracking_eval.precision,
            "mostly-lost": tracking_eval.ML,
            "partially-lost": tracking_eval.PT,
            "mostly-tracked": tracking_eval.MT,
            "ids": tracking_eval.id_switches,
            "fragments": tracking_eval.fragments,
        }

        # Append 'hooked' data, needed for range evaluation.
        #hook_data.append(tracking_eval.hook_data)

    else:
        raise('Evaluation failed')

    # ---------------------------------------------------------------
    #     +++ RANGE ANALYSIS +++
    # ---------------------------------------------------------------
    print(' --- Running range analysis for: %s --- ' % c)
    range_eval_data = tracking_eval.hook_data

    (ranges, moda_range) = range_eval.ComputeMODARange(range_eval_data)
    (ranges, recall_range) = range_eval.ComputeRecallRange(range_eval_data)
    (ranges, precision_range) = range_eval.ComputePrecisionRange(range_eval_data)
    (ranges, loc3d) = range_eval.Compute3Dloc(range_eval_data)

    print ("------------ LOC3D -----------")
    print (loc3d)

    return {
        "ranges":ranges,
        "moda": moda_range,
        "recall": recall_range,
        "precision": precision_range,
        "loc3d": loc3d,
        "cmot_dict": cmot_dict
    }


def format_plot():
    start_distance = 5.0
    end_distance = 60
    plt.grid(True)
    plt.xlim((start_distance, end_distance))


def plot_metric(export_dict, metric_name):
    colors = ['tab:green',  'tab:cyan', 'tab:blue' , 'tab:orange', 'tab:red'] * 10
    back_idx = 1
    dictkeys = export_dict.keys()
    dictkeys.sort(reverse=True)
    for idx, key in enumerate(dictkeys):
        # -------------- PLOT -----------------
        eval_item = export_dict[key]
        ranges = eval_item["ranges"]
        vals = eval_item[metric_name]

        plt.plot(ranges[0:-back_idx], vals[0:-back_idx],
                 linestyle='-',
                 color=colors[idx],
                 linewidth=4,
                 label=key)


def plot_cat(dict, cat, do_x_title=False, output_dir=None):
    fig_size = (4, 4)
    fig = plt.figure(figsize=fig_size)

    ax1 = plt.subplot(211)
    ax1.set_ylabel('Recall')
    plt.title(cat)
    plot_metric(dict, 'recall')
    format_plot()
    ax1.set_yticks(np.arange(0.0, 1.2, 0.2))
    plt.legend()

    ax2 = plt.subplot(212)
    ax2.set_ylabel('Loc-3D (m)')
    plot_metric(dict, 'loc3d')
    format_plot()
    ax2.set_yticks(np.arange(0.0, 10.2, 2.))
    ax2.set_ylim((0.0, 10.0))

    if do_x_title:
        ax2.set_xlabel('Camera Range (m)')

    if output_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, "%s_%s.pdf" % (cat, 'loc_and_recall')), bbox_inches='tight')


def export_table(export_dict, categ_name="NA", output_dir=None):

    if output_dir is not None:
        table = []
        header = []
        title = "%s_clearmot" % categ_name

        for idx, key in enumerate(export_dict.keys()):
            eval_item = export_dict[key]
            cmot_dict = eval_item["cmot_dict"]

            table.append([key] + [cmot_dict[x] for x in cmot_dict.keys()])
            header = [x for x in cmot_dict.keys()]

        table.sort(key=lambda x: x[2], reverse=True) # 2 ... MOTA
        tab_latex = tabulate(table, headers=header, tablefmt="latex", floatfmt=".2f")
        print("------------------------")
        print("Table: %s" % title)
        print(tab_latex)

        with open(os.path.join(output_dir, title + '.tex'), "w") as f:
            f.write(tab_latex)


def evaluate_category(dest_dir, category, dir_prefix='', output_dir=None):
    export_dict = {

    }

    user_specified_results = None
    if dest_dir is not None:
        dirs = os.listdir(dest_dir)
        #dirs.sort(key=MyFn)
        for result_dir in dirs:
            user_specified_results = evaluate_folder(c=category, tr_path=os.path.join(dest_dir, result_dir, dir_prefix))
            export_dict[result_dir] = user_specified_results

    export_table(export_dict, categ_name=category, output_dir=output_dir)
    return export_dict


def main(_):

    # Matplotlib params
    matplotlib.rcParams.update({'font.size':8})
    matplotlib.rcParams.update({'font.family':'sans-serif'})
    matplotlib.rcParams['text.usetex'] = True

    # Make sure output dir exits + add a timestamp dir
    output_dir = None
    if FLAGS.plot_output_dir is not None:
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d__%H_%M_%S')

        if FLAGS.do_not_timestamp:
            timestamp = ""

        output_dir = os.path.join(FLAGS.plot_output_dir, timestamp)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    exp_dict_car = evaluate_category(FLAGS.evaluate_dir, 'car', dir_prefix=FLAGS.result_dir_prefix, output_dir=output_dir)
    exp_dict_ped = evaluate_category(FLAGS.evaluate_dir, 'pedestrian', dir_prefix=FLAGS.result_dir_prefix, output_dir=output_dir)

    plot_cat(exp_dict_ped, 'Pedestrian', do_x_title=True, output_dir=output_dir)
    plot_cat(exp_dict_car, 'Car', do_x_title=True, output_dir=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_output_dir', type=str, help='Plots output dir.')
    parser.add_argument('--evaluate_dir', type=str, help='Dir, containing result files that you want to evaluate')
    parser.add_argument('--result_dir_prefix', type=str, default='', help='Result dir prefix.')
    parser.add_argument('--labels_path', type=str, default='label_02',
                        help='Labels path (default assumed in this repo)')
    parser.add_argument('--seqmap_path', type=str, default='evaluate_tracking.seqmap',
                        help='Sequence map path (optional; default evals whole tracking train)')
    parser.add_argument('--do_not_timestamp', action='store_true', help='Dont timestamp out dirs')

    FLAGS = parser.parse_args()
    main(FLAGS)