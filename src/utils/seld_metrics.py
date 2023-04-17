import os
from weakref import ref
import numpy as np
eps = np.finfo(np.float).eps

from scipy import stats
from scipy.optimize import linear_sum_assignment


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])

def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 5: #polar coordinates 
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # cartesian coordinates
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict

def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
            _fid.write('{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3])))
    _fid.close()

def convert_output_format_polar_to_cartesian(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:

                ele_rad = tmp_val[3]*np.pi/180.
                azi_rad = tmp_val[2]*np.pi/180

                tmp_label = np.cos(ele_rad)
                x = np.cos(azi_rad) * tmp_label
                y = np.sin(azi_rad) * tmp_label
                z = np.sin(ele_rad)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
    return out_dict

def convert_output_format_cartesian_to_polar(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                # in degrees
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                r = np.sqrt(x**2 + y**2 + z**2)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], azimuth, elevation])
    return out_dict

def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth Cartesian or Polar coordinates in Radians
        :param pred_list_xyz: list of predicted Carteisan or Polar coordinates in Radians
        :return: cost - distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        if len(gt_list[0]) == 3: #Cartesian
            x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
        else:
            az1, ele1, az2, ele2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval


class SELDMetrics(object):
    def __init__(self, doa_threshold=20, nb_classes=11, average='macro'):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.
        '''
        self._nb_classes = nb_classes

        # Variables for Location-senstive detection performance
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes)

        self._spatial_T = doa_threshold

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_DE = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        self._average = average

    def early_stopping_metric(self, _er, _f, _le, _lr):
        """
        Compute early stopping metric from sed and doa errors.

        :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
        :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
        :return: early stopping metric result
        """
        seld_metric = np.mean([
            _er,
            1 - _f,
            _le / 180,
            1 - _lr
        ], 0)
        return seld_metric

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''
        ER = (self._S + self._D + self._I) / (self._Nref.sum() + eps)
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            LE = self._total_DE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else 180
            LR = self._DE_TP.sum() / (eps + self._DE_TP.sum() + self._DE_FN.sum())

            SELD_scr = self.early_stopping_metric(ER, F, LE, LR)

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            LE = self._total_DE / (self._DE_TP + eps)
            LE[self._DE_TP==0] = 180.0
            LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)

            SELD_scr = self.early_stopping_metric(np.repeat(ER, self._nb_classes), F, LE, LR)
            classwise_results = np.array([np.repeat(ER, self._nb_classes), F, LE, LR, SELD_scr])
            F, LE, LR, SELD_scr = F.mean(), LE.mean(), LR.mean(), SELD_scr.mean()
        return ER, F, LE, LR, SELD_scr, classwise_results

    def update_seld_scores(self, pred, gt):
        '''
        Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
        Adds the multitrack extensions proposed in paper [2]

        The input pred/gt can either both be Cartesian or Degrees

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''
        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of referece tracks for each class in the segment
                nb_gt_doas = max([len(val) for val in gt[block_cnt][class_cnt][0][1]]) if class_cnt in gt[block_cnt] else None
                nb_pred_doas = max([len(val) for val in pred[block_cnt][class_cnt][0][1]]) if class_cnt in pred[block_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False positive case

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                    # the associated reference-predicted tracks.

                    # Reference and predicted track matching
                    matched_track_dist = {}
                    matched_track_cnt = {}
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind])
                            gt_ids = np.arange(len(gt_arr[:, -1])) #TODO if the reference has track IDS use here - gt_arr[:, -1]
                            gt_doas = gt_arr[:, 1:]

                            pred_ind = pred_ind_list.index(gt_val)
                            pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])
                            pred_doas = pred_arr[:, 1:]

                            if gt_doas.shape[-1] == 2: # convert DOAs to radians, if the input is in degrees
                                gt_doas = gt_doas * np.pi / 180.
                                pred_doas = pred_doas * np.pi / 180.

                            dist_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas)

                            # Collect the frame-wise distance between matched ref-pred DOA pairs
                            for dist_cnt, dist_val in enumerate(dist_list):
                                matched_gt_track = gt_ids[row_inds[dist_cnt]]
                                if matched_gt_track not in matched_track_dist:
                                    matched_track_dist[matched_gt_track], matched_track_cnt[matched_gt_track] = [], []
                                matched_track_dist[matched_gt_track].append(dist_val)
                                matched_track_cnt[matched_gt_track].append(pred_ind)

                    # Update evaluation metrics based on the distance between ref-pred tracks
                    if len(matched_track_dist) == 0:
                        # if no tracks are found. This occurs when the predicted DOAs are not aligned frame-wise to the reference DOAs
                        loc_FN += nb_pred_doas
                        self._FN[class_cnt] += nb_pred_doas
                        self._DE_FN[class_cnt] += nb_pred_doas
                    else:
                        # for the associated ref-pred tracks compute the metrics
                        for track_id in matched_track_dist:
                            total_spatial_dist = sum(matched_track_dist[track_id])
                            total_framewise_matching_doa = len(matched_track_cnt[track_id])
                            avg_spatial_dist = total_spatial_dist / total_framewise_matching_doa

                            # Class-sensitive localization performance
                            self._total_DE[class_cnt] += avg_spatial_dist
                            self._DE_TP[class_cnt] += 1

                            # Location-sensitive detection performance
                            if avg_spatial_dist <= self._spatial_T:
                                self._TP[class_cnt] += 1
                            else:
                                loc_FP += 1
                                self._FP_spatial[class_cnt] += 1
                        # in the multi-instance of same class scenario, if the number of predicted tracks are greater
                        # than reference tracks count as FP, if it less than reference count as FN
                        if nb_pred_doas > nb_gt_doas:
                            # False positive
                            loc_FP += (nb_pred_doas-nb_gt_doas)
                            self._FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                            self._DE_FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                        elif nb_pred_doas < nb_gt_doas:
                            # False negative
                            loc_FN += (nb_gt_doas-nb_pred_doas)
                            self._FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                            self._DE_FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return
    

class ComputeSELDResults(object):
    def __init__(
            self, params, ref_files_folder=None, use_polar_format=True, overlap_test=False, classwise_overlap_test=False
    ):
        self._use_polar_format = use_polar_format
        self._desc_dir         = ref_files_folder
        self._doa_thresh       = 20
        self._nb_classes       = params['data_config']['nb_classes']
        self._label_hop_len_s  = params['data_config']['label_hop_len_s']
        self._nb_label_frames_1s = int(params['data_config']['sr'] / float(int(params['data_config']['sr'] * self._label_hop_len_s)))
        
        # collect reference files
        self._ref_labels = {}
        for ref_file in os.listdir(self._desc_dir):
            # Load reference description file
            gt_dict = load_output_format_file(os.path.join(self._desc_dir, ref_file))
            if not self._use_polar_format:
                gt_dict = convert_output_format_polar_to_cartesian(gt_dict)
            nb_ref_frames = max(list(gt_dict.keys()))
            self._ref_labels[ref_file] = [self.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]
                
        self._nb_ref_files = len(self._ref_labels)
        self._average = 'macro'

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        pred_labels_dict = {}
        eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = load_output_format_file(os.path.join(pred_files_path, pred_file))
            if self._use_polar_format:
                pred_dict = convert_output_format_cartesian_to_polar(pred_dict)
            pred_labels = self.segment_labels(pred_dict, self._ref_labels[pred_file][1])
            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels
        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
            global_values = [ER, F, LE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_files:
                leave_one_out_list = pred_files[:]
                leave_one_out_list.remove(leave_file)
                eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0])
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                leave_one_out_est = [ER, F, LE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)
                    
            estimate, bias, std_err, conf_interval = [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                           global_value=global_values[i],
                           partial_estimates=partial_estimates[:, i],
                           significance_level=0.05
                           )
            return [ER, conf_interval[0]], [F, conf_interval[1]], [LE, conf_interval[2]], [LR, conf_interval[3]], [seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5,self._nb_classes,2) if len(classwise_results) else []]
      
        else:      
            return ER, F, LE, LR, seld_scr, classwise_results
        
    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames/float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt+self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict
        

class ComputeSELDResultsFromEventOverlap(object):
    def __init__(
            self, params, ref_files_folder=None, use_polar_format=True, classwise_overlap_test=False
    ):
        self._use_polar_format = use_polar_format
        self._desc_dir         = ref_files_folder
        self._doa_thresh       = 20
        self._nb_classes       = params['data_config']['nb_classes']
        self._label_hop_len_s  = params['data_config']['label_hop_len_s']
        self._nb_label_frames_1s = int(params['data_config']['sr'] / float(int(params['data_config']['sr'] * self._label_hop_len_s)))
        self._classwise_overlap_test = classwise_overlap_test
        
        # collect reference files
        self._ref_labels = {}
        self._ref_ov_frame_keys = {}

        for ref_file in os.listdir(self._desc_dir):
            gt_dict = load_output_format_file(os.path.join(self._desc_dir, ref_file))

            if not self._use_polar_format:
                gt_dict = convert_output_format_polar_to_cartesian(gt_dict)
                
            nb_ref_frames = max(list(gt_dict.keys()))
            
            #########################################################################################
            self._ref_ov_frame_keys[ref_file] = []
            overlap_gt_dict = {}
            for frame_idx, event_list in gt_dict.items():
                
                if self._classwise_overlap_test:
                    class_cnt = np.zeros(self._nb_classes)
                    for event in event_list:
                        class_cnt[event[0]] += 1
                    if class_cnt.max() > 1:
                        self._ref_ov_frame_keys[ref_file].append(frame_idx)
                        overlap_gt_dict[frame_idx] = event_list
                else:
                    if len(event_list) > 1:
                        self._ref_ov_frame_keys[ref_file].append(frame_idx)
                        overlap_gt_dict[frame_idx] = event_list
                        
            gt_dict = overlap_gt_dict
            
            if len(gt_dict):
                self._ref_labels[ref_file] = [self.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]
            #########################################################################################
                
        self._nb_ref_files = len(self._ref_labels)
        self._average = 'macro'
        
        print('{} files have sound-overlapping events...'.format(self._nb_ref_files))
        print('a total of {} frames comprise the overlapping events...'.format(sum([len(v) for i, v in self._ref_ov_frame_keys.items()])))

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        pred_labels_dict = {}

        eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            #########################################################################################
            if pred_file not in self._ref_labels.keys():
                continue
            #########################################################################################
            
            # Load predicted output format file
            pred_dict = load_output_format_file(os.path.join(pred_files_path, pred_file))

            if self._use_polar_format:
                pred_dict = convert_output_format_cartesian_to_polar(pred_dict)
                
            #########################################################################################
            overlap_pred_dict = {}
            for ov_frame_idx in self._ref_ov_frame_keys[pred_file]:
                if pred_dict.get(ov_frame_idx) is None:
                    continue
                
                overlap_pred_dict[ov_frame_idx] = pred_dict[ov_frame_idx]
            pred_dict = overlap_pred_dict
            #########################################################################################
            
            pred_labels = self.segment_labels(pred_dict, self._ref_labels[pred_file][1])

            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels


        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
            global_values = [ER, F, LE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_files:
                leave_one_out_list = pred_files[:]
                leave_one_out_list.remove(leave_file)

                eval = SELDMetrics(nb_classes=self._nb_classes, doa_threshold=self._doa_thresh, average=self._average)

                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0])
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                leave_one_out_est = [ER, F, LE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)
                    
            estimate, bias, std_err, conf_interval = [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                           global_value=global_values[i],
                           partial_estimates=partial_estimates[:, i],
                           significance_level=0.05
                           )
            return [ER, conf_interval[0]], [F, conf_interval[1]], [LE, conf_interval[2]], [LR, conf_interval[3]], [seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5,self._nb_classes,2) if len(classwise_results) else []]
      
        else:      
            return ER, F, LE, LR, seld_scr, classwise_results
        
    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames/float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt+self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict