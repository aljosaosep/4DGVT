#!/usr/bin/env python
# encoding: utf-8

"""
System
"""
import sys, os, copy, math
from collections import defaultdict
try:
	from ordereddict import OrderedDict  # can be installed using pip
except:
	from collections import OrderedDict  # only included from python 2.7 on


"""
Munkres - Hungarian alg.
"""
from munkres import Munkres


class tData:
	def __init__(self, frame=-1, obj_type="unset", truncation=-1, occlusion=-1, \
				 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, w=-1, h=-1, l=-1, \
				 X=-1000, Y=-1000, Z=-1000, yaw=-10, score=-1000, track_id=-1):
		# init object data
		self.frame = frame
		self.track_id = track_id
		self.obj_type = obj_type
		self.truncation = truncation
		self.occlusion = occlusion
		self.obs_angle = obs_angle
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.w = w
		self.h = h
		self.l = l
		self.X = X
		self.Y = Y
		self.Z = Z
		self.yaw = yaw
		self.score = score
		self.ignored = False
		self.valid = False
		self.tracker = -1

	def __str__(self):
		attrs = vars(self)
		return '\n'.join("%s: %s" % item for item in attrs.items())


class trackingEvaluation(object):
	""" tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
			 MOTA	- Multi-object tracking accuracy in [0,100]
			 MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
			 MOTAL	- Multi-object tracking accuracy in [0,100] with log10(id-switches)

			 id-switches - number of id switches
			 fragments   - number of fragmentations

			 MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories

			 recall	        - recall = percentage of detected targets
			 precision	    - precision = percentage of correctly detected targets
			 FAR		    - number of false alarms per frame
			 falsepositives - number of false positives (FP)
			 missed         - number of missed targets (FN)
	"""

	def __init__(self, t_sha, t_output_path="./results", t_path_base="./results", gt_path="./data/tracking",
				 min_overlap=0.5, max_truncation=0.15, mail=None, cls="car", \
				 filename_test_mapping="./data/tracking/evaluate_tracking.seqmap", \
				 hook_initialize=None, hook_finalize=None, hook_track=None, do_not_modif_t_path=False):

		# get number of sequences and
		# get number of frames per sequence from test mapping
		# (created while extractign the benchmark)
		self.n_frames = []
		self.sequence_name = []
		with open(filename_test_mapping, "r") as fh:
			for i, l in enumerate(fh):
				fields = l.split(" ")
				self.sequence_name.append("%04d" % int(fields[0]))
				self.n_frames.append(int(fields[3]) - int(fields[2]) + 1)
		fh.close()

		self.n_sequences = i + 1

		# mail object
		self.mail = mail

		# class to evaluate
		self.cls = cls

		# data and parameter
		self.dataset = "training"
		self.gt_path = os.path.join(gt_path, "label_02")
		self.t_output_path = t_output_path
		self.t_sha = t_sha
		self.t_path_base = t_path_base

		# Modified by Aljosa:
		# By default, you will find tracking results in path_base/sha_key/data
		# When this is not the case, set 'do_not_modif_t_path'.
		# Then, it is assumed that result files are directly in 'self.t_path'
		self.t_path = t_path_base
		if not do_not_modif_t_path:
			self.t_path = os.path.join(t_path_base, t_sha, "data")

		self.n_gt = 0
		self.n_gt_trajectories = 0
		self.n_gt_seq = []
		self.n_tr = 0
		self.n_tr_trajectories = 0
		self.n_tr_seq = []
		self.min_overlap = min_overlap  # minimum bounding box overlap for 3rd party metrics
		self.max_truncation = max_truncation  # maximum truncation of an object for evaluation
		self.n_sample_points = 500

		# figures for evaluation
		self.MOTA = 0
		self.MOTP = 0
		self.MOTAL = 0
		self.MODA = 0
		self.MODP = 0
		self.MODP_t = []
		self.recall = 0
		self.precision = 0
		self.F1 = 0
		self.FAR = 0
		self.total_cost = 0
		self.tp = 0
		self.fn = 0
		self.fp = 0
		self.mme = 0
		self.fragments = 0
		self.id_switches = 0
		self.MT = 0
		self.PT = 0
		self.ML = 0
		self.distance = []
		self.seq_res = []
		self.seq_output = []

		# Temporary!
		#self.tp_ = 0
		#self.fn_ = 0


		# this should be enough to hold all groundtruth trajectories
		# is expanded if necessary and reduced in any case
		self.gt_trajectories = [[] for x in xrange(self.n_sequences)]
		self.ign_trajectories = [[] for x in xrange(self.n_sequences)]

		# Wolle's code
		if hook_initialize is None:
			self.hook_initialize = lambda sha_key, obj_class: 1
		else:
			self.hook_initialize = hook_initialize
		if hook_finalize is None:
			self.hook_finalize = lambda sha_key, obj_class: 1
		else:
			self.hook_finalize = hook_finalize
		if hook_track is None:
			self.hook_track = lambda event, gt, track, ignored: 1
		else:
			self.hook_track = hook_track

		self.hook_data = None

	def createEvalDir(self):
		"""Creates directory to store evaluation results and data for visualization"""
		self.eval_dir = os.path.join(self.t_output_path, self.t_sha, "eval", self.cls)
		if not os.path.exists(self.eval_dir):
			print "create directory:", self.eval_dir,
			os.makedirs(self.eval_dir)
			print "done"

	def loadGroundtruth(self):
		"""Helper function to load ground truth"""
		try:
			self._loadData(self.gt_path, cls=self.cls, loading_groundtruth=True)
		except IOError:
			# print "Failed to load tracker data."
			return False

		print "Load labels - Success"
		return True


	def loadTracker(self, evaluating_clearmot=False):
		"""Helper function to load tracker data"""
		try:
			# DBG
			print "Loading: %s" % self.t_path

			if not self._loadData(self.t_path, cls=self.cls, loading_groundtruth=False):
				print "Failed to load tracker data: %s" % self.t_path
				return False
		except IOError:
			return False

		print "Load tracker - Success"
		return True

	def loadProposals(self):
		"""Helper function to load object-proposals data"""
		try:
			# print "Loading: %s"%self.t_path
			if not self._loadData(self.t_path, cls=self.cls, loading_groundtruth=False):
				# print "Failed to load tracker data."
				return False
		except IOError:
			return False

		print "Load proposals - Success"
		return True


	def _loadData(self, root_dir, cls, min_score=-1000, loading_groundtruth=False):
		"""
			Generic loader for ground truth and tracking data.
			Use loadGroundtruth() or loadTracker() to load this data.
			Loads detections in KITTI format from textfiles.
		"""


		t_data = tData()
		data = []
		eval_2d = True
		eval_3d = True

		seq_data = []
		n_trajectories = 0
		n_trajectories_seq = []


		# Load data for each subsequence
		for seq, s_name in enumerate(self.sequence_name):

			# print ('Loading seq: %s'%s_name)

			# Open the result file
			i = 0
			filename = os.path.join(root_dir, "%s.txt" % s_name)
			f = open(filename, "r")

			# For current sequence, initialize 'f_data_ list (size -> number of frames)
			f_data = [[] for x in xrange(self.n_frames[seq])]  # current set has only 1059 entries, sufficient length is checked anyway
			ids = []
			n_in_seq = 0
			id_frame_cache = []
			frame_counter = 1

			# Read the lines in the result file
			for line in f:
				# Parse the line
				line = line.strip()
				fields = line.split(" ")

				# classes that should be loaded (ignored neighboring classes)
				if "car" in cls.lower():
					classes = ["car", "van"]
				elif "pedestrian" in cls.lower():
					classes = ["pedestrian", "person_sitting"]
				else:
					classes = [cls.lower()]
				classes += ["dontcare"]


				if not any([s for s in classes if s in fields[2].lower()]):
					continue

				# Initialize t_data structure (all fields were parsed into 'fields' list)
				t_data.frame = int(float(fields[0]))  # frame
				t_data.track_id = int(float(fields[1]))  # id
				t_data.obj_type = fields[2].lower()  # object type [car, pedestrian, cyclist, ...]
				t_data.truncation = float(fields[3])  # truncation [0..1]
				t_data.occlusion = int(float(fields[4]))  # occlusion  [0,1,2]
				t_data.obs_angle = float(fields[5])  # observation angle [rad]
				t_data.x1 = float(fields[6])  # left   [px]
				t_data.y1 = float(fields[7])  # top    [px]
				t_data.x2 = float(fields[8])  # right  [px]
				t_data.y2 = float(fields[9])  # bottom [px]
				t_data.h = float(fields[10])  # height [m]
				t_data.w = float(fields[11])  # width  [m]
				t_data.l = float(fields[12])  # length [m]
				t_data.X = float(fields[13])  # X [m]
				t_data.Y = float(fields[14])  # Y [m]
				t_data.Z = float(fields[15])  # Z [m]
				t_data.yaw = float(fields[16])  # yaw angle [rad]

				# Some weird post-processing (?)
				if not loading_groundtruth:
					if len(fields) == 17:
						t_data.score = -1
					elif len(fields) == 18:
						t_data.score = float(fields[17])  # detection score
					else:
						# self.mail.msg("file is not in KITTI format")
						print "file is not in KITTI format"
						return


				idx = t_data.frame

				# Make sure we have same length for 'tracker' and 'groundtruth', otherwise evaluation will crash.
				if idx >= len(f_data):
					f_data += [[] for x in xrange(max(500, idx - len(f_data)))]

				if t_data.track_id not in ids and t_data.obj_type != "dontcare":
					ids.append(t_data.track_id)
					n_trajectories += 1
					n_in_seq += 1

				# check if uploaded data provides information for 2D and 3D evaluation
				if not loading_groundtruth and eval_2d is True and (
								t_data.x1 == -1 or t_data.x2 == -1 or t_data.y1 == -1 or t_data.y2 == -1):
					eval_2d = False
				if not loading_groundtruth and eval_3d is True and (
							t_data.X == -1000 or t_data.Y == -1000 or t_data.Z == -1000):
					eval_3d = False

			# only add existing frames
			n_trajectories_seq.append(n_in_seq)
			seq_data.append(f_data)

			f.close()

		if not loading_groundtruth:
			self.tracker = seq_data
			self.n_tr_trajectories = n_trajectories
			self.eval_2d = eval_2d
			self.eval_3d = eval_3d
			self.n_tr_seq = n_trajectories_seq
			if self.n_tr_trajectories == 0:
				print "Error: self.n_tr_trajectories==0"
				return False
		else:
			# split ground truth and DontCare areas
			self.dcareas = []
			self.groundtruth = []
			for seq_idx in range(len(seq_data)):
				seq_gt = seq_data[seq_idx]
				s_g, s_dc = [], []
				for f in range(len(seq_gt)):
					all_gt = seq_gt[f]
					g, dc = [], []
					for gg in all_gt:
						if gg.obj_type == "dontcare":
							dc.append(gg)
						else:
							g.append(gg)
					s_g.append(g)
					s_dc.append(dc)
				self.dcareas.append(s_dc)
				self.groundtruth.append(s_g)
			self.n_gt_seq = n_trajectories_seq
			self.n_gt_trajectories = n_trajectories
		return True

	def boxoverlap(self, a, b, criterion="union"):
		"""
			boxoverlap computes intersection over union for bbox a and b in KITTI format.
			If the criterion is 'union', overlap = (a inter b) / a union b).
			If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
		"""
		x1 = max(a.x1, b.x1)
		y1 = max(a.y1, b.y1)
		x2 = min(a.x2, b.x2)
		y2 = min(a.y2, b.y2)

		w = x2 - x1
		h = y2 - y1

		if w <= 0. or h <= 0.:
			return 0.
		inter = w * h
		aarea = (a.x2 - a.x1) * (a.y2 - a.y1)
		barea = (b.x2 - b.x1) * (b.y2 - b.y1)
		# intersection over union overlap
		if criterion.lower() == "union":
			o = inter / float(aarea + barea - inter)
		elif criterion.lower() == "a":
			o = float(inter) / float(aarea)
		else:
			raise TypeError("Unkown type for criterion")
		return o


	# Wolle's version
	def compute3rdPartyMetrics(self):
		"""
			Computes the metrics defined in
				- Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
				  MOTA, MOTAL, MOTP
				- Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows
				  MT/PT/ML
		"""

		self.hook_initialize(self.t_sha, self.cls)

		# construct Munkres object for Hungarian Method association
		hm = Munkres()
		max_cost = 1e9

		# go through all frames and associate ground truth and tracker results
		# groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
		fr, ids = 0, 0

		# -------------------------------------------------------------------------------
		# +++ FOR EVERY SEQUENCE +++
		# -------------------------------------------------------------------------------
		for seq_idx in range(len(self.groundtruth)):
			seq_gt = self.groundtruth[seq_idx]
			seq_dc = self.dcareas[seq_idx]
			seq_tracker = self.tracker[seq_idx]
			seq_trajectories = defaultdict(list)
			seq_ignored = defaultdict(list)
			seqtp = 0
			seqfn = 0
			seqfp = 0
			seqcost = 0

			last_ids = [[], []]
			tmp_frags = 0

			# -------------------------------------------------------------------------------
			# +++ FOR EVERY FRAME +++
			# -------------------------------------------------------------------------------
			#for f in range(len(seq_gt)):
			n_frames_this_seq = self.n_frames[seq_idx]
			for f in range(n_frames_this_seq):	# ALJOSA 2017_04_28: POSSIBLY BUGFIX BUT NOT SURE			
				g = seq_gt[f]  # GT objects for this frame
				dc = seq_dc[f]  # DontCares for this frame
				t = seq_tracker[f]  # Tracker hypotheses for this frame

				# Total number of ground truth and tracker objects
				self.n_gt += len(g)  # Total GT objects
				self.n_tr += len(t)  # Total tracker objects

				# Use hungarian method to associate, using IOU boxoverlap 0..1 as cost
				# (1) Build the cost matrix
				cost_matrix = []
				this_ids = [[], []]

				# Loop over all GT objects in this frame
				for gg in g:  # This is a loop over all GT objects, 'gg' is of a type 'tData'
					# save current ids
					this_ids[0].append(gg.track_id)
					this_ids[1].append(-1)
					gg.tracker = -1
					gg.id_switch = 0
					gg.fragmentation = 0
					cost_row = []

					# Loop over all tracker hypotheses ('tt' is of a type 'tData')
					# Note: one could use here different matching fnc. that 'boxoverlap' (which is IOU in the image domain)
					for tt in t:
						# Overlap == 1 is cost == 0
						c = 1 - self.boxoverlap(gg, tt)

						# Gating for boxoverlap
						if c <= self.min_overlap:
							cost_row.append(c)
						else:
							cost_row.append(max_cost)

					cost_matrix.append(cost_row)

					# All ground truth trajectories are initially not associated
					# Extend groundtruth trajectories lists (merge lists)
					seq_trajectories[gg.track_id].append(-1)
					seq_ignored[gg.track_id].append(False)

				if len(g) is 0:
					cost_matrix = [[]]

				# Compute association (Hungarian alg.)
				association_matrix = hm.compute(cost_matrix)

				idx_g2t = [-1] * len(g)  # Map ground-truth index to associated track index
				idx_t2g = [-1] * len(t)  # Vice versa

				for idx_g, idx_t in association_matrix:
					c = cost_matrix[idx_g][idx_t]
					if c < max_cost:
						idx_g2t[idx_g] = idx_t
						idx_t2g[idx_t] = idx_g

				# Mapping for tracker ids and ground truth ids
				tmptp = 0
				tmpfp = 0
				tmpfn = 0
				tmpc = 0
				this_cost = [-1] * len(g)

				# Ignored FN/TP (truncation or neighboring object class)
				# Ignored trackers in neighboring classes
				nignoredfn = 0
				nignoredtp = 0
				nignoredtracker = 0

				for idx_g, gg in enumerate(g):
					this_tp = False
					this_fn = False

					idx_t = idx_g2t[idx_g]

					if idx_t != -1:
						tt = t[idx_t]
						c = cost_matrix[idx_g][idx_t]

						this_tp = True

						g[idx_g].tracker = t[idx_t].track_id
						g[idx_g].distance = c
						t[idx_t].valid = True

						this_ids[1][idx_g] = t[idx_t].track_id
						seq_trajectories[g[idx_g].track_id][-1] = t[idx_t].track_id

						self.total_cost += 1 - c
						seqcost += 1 - c
						tmpc += 1 - c
						this_cost.append(c)

						g[idx_g].distance3d = math.sqrt((gg.X - tt.X) ** 2 + (gg.Z - tt.Z) ** 2)

						# ignored TP due to truncation
						# or due neighboring object class
						if gg.truncation > self.max_truncation \
								or (self.cls == "car" and gg.obj_type == "van") \
								or (self.cls == "pedestrian" and gg.obj_type == "person_sitting") \
								or (self.cls == "pedestrian" and gg.obj_type == "cyclist") \
								or (gg.occlusion > 2): # NEW!
							seq_ignored[gg.track_id][-1] = True
							gg.ignored = True
							nignoredtp += 1
							this_tp = False

						if this_tp:
							self.tp += 1
							tmptp += 1
							self.hook_track('tp', gg, tt, False)

						else:
							self.tp += 1
							self.hook_track('tp', gg, tt, True)
					else:
						this_fn = True

						# no tracker associated
						g[idx_g].tracker = -1

						# ignored FN due to truncation
						# or due to neighboring object class
						if gg.truncation > self.max_truncation \
								or (self.cls == "car" and gg.obj_type == "van") \
								or (self.cls == "pedestrian" and gg.obj_type == "person_sitting") \
								or (self.cls == "pedestrian" and gg.obj_type == "cyclist") \
								or (gg.occlusion > 2): # NEW!
							seq_ignored[gg.track_id][-1] = True
							gg.ignored = True
							nignoredfn += 1
							this_fn = False

						if this_fn:
							self.fn += 1
							tmpfn += 1
							self.hook_track('fn', gg, None, False)
						else:
							self.hook_track('fn', gg, None, True)

				for idx_t, tt in enumerate(t):
					this_fp = False

					idx_g = idx_t2g[idx_t]

					if idx_g < 0:
						this_fp = True

					# ignore tracker in neighboring classes
					# associate tracker and DontCare areas
					if (self.cls == "car" and tt.obj_type == "van") or (self.cls == "pedestrian" and tt.obj_type == "person_sitting"):
						nignoredtracker += 1
						tt.ignored = True
					elif not tt.valid:
						for d in dc:
							overlap = self.boxoverlap(tt, d, "a")
							if overlap > 0.5:
								nignoredtracker += 1
								tt.ignored = False
								break

					if this_fp:
						tmpfp += 1
						self.fp += 1
						self.hook_track('fp', None, tt, False)
					else:
						self.hook_track('fp', None, tt, True)

				# append single distance values
				self.distance.append(this_cost)

				# save current index
				last_ids = this_ids
				# compute MOTP_t
				MODP_t = 0
				if tmptp != 0:
					MODP_t = tmpc / float(tmptp)
				self.MODP_t.append(MODP_t)

				self.n_gt -= (nignoredfn + nignoredtp)

				# update sequence data
				seqtp += tmptp
				seqfp += tmpfp
				seqfn += tmpfn

			# -------------------------------------------------------------------------------
			# --- FOR EVERY FRAME ---
			# -------------------------------------------------------------------------------

			# remove empty lists for current gt trajectories
			self.gt_trajectories[seq_idx] = seq_trajectories
			self.ign_trajectories[seq_idx] = seq_ignored

		# compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
		n_ignored_tr_total = 0
		for seq_idx, (seq_trajectories, seq_ignored) in enumerate(zip(self.gt_trajectories, self.ign_trajectories)):
			if len(seq_trajectories) == 0:
				continue
			tmpMT, tmpML, tmpPT, tmpId_switches, tmpFragments = [0] * 5
			n_ignored_tr = 0
			for g, ign_g in zip(seq_trajectories.values(), seq_ignored.values()):
				# all frames of this gt trajectory are ignored
				if all(ign_g):
					n_ignored_tr += 1
					n_ignored_tr_total += 1
					continue
				if all([this == -1 for this in g]):
					tmpML += 1
					self.ML += 1
					continue
				# compute tracked frames in trajectory
				last_id = g[0]
				# first detection (necessary to be in gt_trajectories) is always tracked
				tracked = 1 if g[0] >= 0 else 0
				lgt = 0 if ign_g[0] else 1
				for f in range(1, len(g)):
					if ign_g[f]:
						last_id = -1
						continue
					lgt += 1
					if last_id != g[f] and last_id != -1 and g[f] != -1 and g[f - 1] != -1:
						tmpId_switches += 1
						self.id_switches += 1
					if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
						tmpFragments += 1
						self.fragments += 1
					if g[f] != -1:
						tracked += 1
						last_id = g[f]
				# handle last frame; tracked state is handeled in for loop (g[f]!=-1)
				if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and not ign_g[f]:
					tmpFragments += 1
					self.fragments += 1

				# compute MT/PT/ML
				tracking_ratio = tracked / float(len(g) - sum(ign_g))
				if tracking_ratio > 0.8:
					tmpMT += 1
					self.MT += 1
				elif tracking_ratio < 0.2:
					tmpML += 1
					self.ML += 1
				else:  # 0.2 <= tracking_ratio <= 0.8
					tmpPT += 1
					self.PT += 1

		if (self.n_gt_trajectories - n_ignored_tr_total) == 0:
			self.MT = 0.
			self.PT = 0.
			self.ML = 0.
		else:
			self.MT /= float(self.n_gt_trajectories - n_ignored_tr_total)
			self.PT /= float(self.n_gt_trajectories - n_ignored_tr_total)
			self.ML /= float(self.n_gt_trajectories - n_ignored_tr_total)

		# precision/recall etc.
		if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
			self.recall = 0.
			self.precision = 0.
		else:
			self.recall = self.tp / float(self.tp + self.fn)
			self.precision = self.tp / float(self.fp + self.tp)
		if (self.recall + self.precision) == 0:
			self.F1 = 0.
		else:
			self.F1 = 2. * (self.precision * self.recall) / (self.precision + self.recall)
		if sum(self.n_frames) == 0:
			self.FAR = "n/a"
		else:
			self.FAR = self.fp / float(sum(self.n_frames))

		# compute CLEARMOT
		if self.n_gt == 0:
			self.MOTA = -float("inf")
			self.MODA = -float("inf")
		else:
			self.MOTA = 1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)
			self.MODA = 1 - (self.fn + self.fp) / float(self.n_gt)
		if self.tp == 0:
			self.MOTP = float("inf")
		else:
			self.MOTP = self.total_cost / float(self.tp)
		if self.n_gt != 0:
			if self.id_switches == 0:
				self.MOTAL = 1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)
			else:
				self.MOTAL = 1 - (self.fn + self.fp + math.log10(self.id_switches)) / float(self.n_gt)
		else:
			self.MOTAL = -float("inf")
		if sum(self.n_frames) == 0:
			self.MODP = "n/a"
		else:
			self.MODP = sum(self.MODP_t) / float(sum(self.n_frames))

		self.hook_data = self.hook_finalize(self.t_sha, self.cls)

		return True


# By Wolle
def range_analysis():

	"""
	:return:
	"""

	data = {'obj_class': '', 'bins': [], 'meta': {}}

	meta_std = {}

	meta_std['range_min'] = 0.0  # in m
	meta_std['range_max'] = 50.0  # in m
	meta_std['range_len'] = meta_std['range_max'] - meta_std['range_min']
	meta_std['n_bins'] = 10 + 2

	data['meta'] = meta_std

	def init(sha_key, obj_class):

		if obj_class == 'car':
			meta = {}

			meta['range_min'] = 0.0  # in m
			meta['range_max'] = 70.0  # in m
			meta['range_len'] = meta['range_max'] - meta['range_min']
			meta['n_bins'] = 14 + 2

			data['meta'] = meta
		else:
			data['meta'] = meta_std

		meta = data['meta']
		n_bins = meta['n_bins']

		data['obj_class'] = obj_class
		data['bins'] = []
		for idx in range(n_bins):
			rec = {'tp': 0, 'fn': 0, 'fp': 0, 'dist2d': 0.0, 'dist3d': 0.0, 'Z': 0.0, 'tp_bboxes':{}, 'fp_bboxes':{}, 'fn_bboxes':{},}
			rec['Z'] = (idx - 1.0) * meta['range_len'] / (n_bins - 2.0) + meta['range_min']
			data['bins'].append(rec)

	def finalize(sha_key, obj_class):
		fout = open('{0}_{1}_ranges.csv'.format(sha_key, obj_class), 'w')
		fout.write('Class {0}\n'.format(obj_class))
		fout.write('{0:6} {1:6} {2:6} {3:6} {4:6} {5:6}\n'.format('Z', 'tp', 'fn', 'fp', 'dist2d', 'dist3d'))
		for rec in data['bins']:
			fout.write('{0:6} {1:6} {2:6} {3:6} {4:6} {5:6}\n'.format(rec['Z'], rec['tp'], rec['fn'], rec['fp'], rec['dist2d'], rec['dist3d']))

		return data['bins']

	def add_bbox(data, idx, obj, bbox_type):
		bboxes = data['bins'][idx][bbox_type]
		if obj.frame in bboxes:
			bboxes[obj.frame].append([obj.x1, obj.y1, obj.x2, obj.y2])
		else:
			bboxes[obj.frame] = [[obj.x1, obj.y1, obj.x2, obj.y2]]

	def track(event, gt, track, ignored):
		if ignored:
			return

		meta = data['meta']
		n_bins = meta['n_bins']

		if event == 'tp' or event == 'fn':
			Z = gt.Z
		elif event == 'fp':
			Z = track.Z

		if Z < meta['range_min']:
			data['bins'][0][event] += 1
		elif Z >= meta['range_max']:
			data['bins'][n_bins - 1][event] += 1
		else:
			idx = int(math.floor((Z - meta['range_min']) / meta['range_len'] * (n_bins - 2)) + 1)
			data['bins'][idx][event] += 1
			if event == 'tp':
				data['bins'][idx]['dist2d'] += gt.distance
				data['bins'][idx]['dist3d'] += gt.distance3d

				# TODO: ADD BBOXES
				add_bbox(data, idx, track, 'tp_bboxes')
				# tp_bboxes = data['bins'][idx]['tp_bboxes']
				# if track.frame in tp_bboxes:
				# 	tp_bboxes[track.frame].append([track.x1, track.y1, track.x2, track.y2])
				# else:
				# 	tp_bboxes[track.frame] = [[track.x1, track.y1, track.x2, track.y2]]

			elif event == 'fp':
				add_bbox(data, idx, track, 'fp_bboxes')
			elif event == 'fn':
				add_bbox(data, idx, gt, 'fn_bboxes')

	return (init, finalize, track)