from genericpath import exists
from random import sample
from typing import Dict, Tuple
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
matplotlib.use('agg')

from pathlib import Path
from geometry import *
import numpy as np
import pandas as pd
import seaborn as sns

from evo.core import metrics
from evo.core.metrics import PoseRelation, StatisticsType, Result
class PostRunOperation():
    def __init__(self):
        pass
    def visit_model_results(self, model_results_path):
        for d in sorted(model_results_path.iterdir()):
            if d.is_dir():
                self.visit_sample_results(model_results_path, d)
        self.on_end_model_visit(model_results_path)
    def visit_sample_results(self, model_results_path, sample_path):
        pass
    def on_end_model_visit(self, model_results_path):
        pass
    def visit_multiple_model_results(self, models_root):
        for path in sorted(models_root.iterdir()):
            print(path)
            if path.is_dir():
                self.visit_model_results(path)
        self.on_end_multiple_models_visit(models_root)
    def on_end_multiple_models_visit(self, models_root):
        pass

class GeneratePoseErrorStats(PostRunOperation):
    ET_NAME = 'error_translation.txt'
    ER_NAME = 'error_rotation.txt'
    VIS_NAME = 'initial_overlap.txt'
    LOOK_AT_NAME = 'point_look_at.txt'
    LOOK_AT_DESIRED_NAME = 'point_look_at_desired.txt'
    VELOCITIES_NAME = 'vc.txt.npz'
    CDRC_NAME = 'cdrc.txt.npz'

    def __init__(self, path_to_name_dict, global_results_path):
        self.errors = {}
        self.path_to_name_dict = path_to_name_dict
        self.visibilities = {}
        self.points_look_at = {}
        self.look_at_dist_to_conv_rate_dicts = {}
        self.converged_velocities = {}
        self.curvatures = {}
        self.global_results_path = global_results_path
        self.global_results_path.mkdir(exist_ok=True)
    @staticmethod
    def has_converged(set, ser, eet, eer):
        return eet / max(set, 0.0001) < 0.1 and eer / max(ser, 0.0001) < 0.1 or eet < 0.01 and eer < 1.0
    @staticmethod
    def has_converged_on_velocities(velocities):
        end_vcs = velocities[-10:]
        end_vcs_t = end_vcs[:, :3]
        end_vcs_r = end_vcs[:, 3:]
        tnorm_mm = np.linalg.norm(end_vcs_t, axis=-1) * 1000.0
        rnorm_deg = np.degrees(np.linalg.norm(end_vcs_r, axis=-1))
        conv_t = np.all(tnorm_mm < 0.1)
        conv_r = np.all(rnorm_deg < 0.01)
        return conv_t and conv_r

    @staticmethod
    def compute_curvature_mean(traj):
        vcs = traj[1:] - traj[:-1]
        accs = vcs[1:] - vcs[:-1]

        vcs_pow = np.linalg.norm(vcs[1:], axis=-1) ** 3
        cross_prod = np.cross(accs, vcs[1:])
        cross_norm = np.linalg.norm(cross_prod, axis=-1)
        curvatures = cross_norm / vcs_pow

        weights = np.linalg.norm(vcs[1:], axis=-1)
        weights /= np.sum(weights)
        curvature = np.sum(curvatures * weights)

        return curvature, curvatures


    def visit_model_results(self, model_results_path):
        if model_results_path not in self.path_to_name_dict:
            pass
        else:
            super().visit_model_results(model_results_path)

    def visit_sample_results(self, model_results_path, sample_path):
        if not str(sample_path.name).isdigit():
            return
        if model_results_path not in self.path_to_name_dict:
            return
        if model_results_path not in self.errors:
            self.errors[model_results_path] = [[] for _ in range(5)]
            self.visibilities[model_results_path] = []
            self.points_look_at[model_results_path] = []
            self.converged_velocities[model_results_path] = []
            self.curvatures[model_results_path] = []
        if (sample_path / GeneratePoseErrorStats.ET_NAME).exists():
            with open(str(sample_path / GeneratePoseErrorStats.ET_NAME), 'r') as etf:
                set = float(etf.readline())
                for last_line in etf:
                    pass
                eet = float(last_line)
        else:
            d = np.load(sample_path / (GeneratePoseErrorStats.ET_NAME + '.npz'))['arr_0']
            set = d[0]
            eet = d[-1]
        if (sample_path / GeneratePoseErrorStats.ET_NAME).exists():
            with open(str(sample_path / GeneratePoseErrorStats.ER_NAME), 'r') as erf:
                ser = float(erf.readline())
                for last_line in erf:
                    pass
                eer = float(last_line)
        else:
            d = np.load(sample_path / (GeneratePoseErrorStats.ER_NAME + '.npz'))['arr_0']
            ser = d[0]
            eer = d[-1]
        vis = 0.0
        if (sample_path / GeneratePoseErrorStats.VIS_NAME).exists():
            with open(str(sample_path / GeneratePoseErrorStats.VIS_NAME), 'r') as vf:
                vis = float(vf.readline())
        path_look_at = sample_path / GeneratePoseErrorStats.LOOK_AT_NAME
        if path_look_at.exists():
            with open(str(sample_path / GeneratePoseErrorStats.LOOK_AT_NAME), 'r') as plf:
                p = np.array([float(v) for v in plf])
            with open(str(sample_path / GeneratePoseErrorStats.LOOK_AT_DESIRED_NAME), 'r') as pdlf:
                pd = np.array([float(v) for v in pdlf])
        else:
            p = [0, 0, 0]
            pd = [0, 0, 0]
        velocities = np.load(sample_path / GeneratePoseErrorStats.VELOCITIES_NAME)['arr_0']
        
        cdrc = np.load(sample_path / GeneratePoseErrorStats.CDRC_NAME)['arr_0']
        curvature, curvatures = GeneratePoseErrorStats.compute_curvature_mean(cdrc[:, :3])
        self.curvatures[model_results_path].append(curvature)
        self.visibilities[model_results_path].append(vis)
        self.points_look_at[model_results_path].append((p, pd))
        self._add_erors(model_results_path, set, ser, eet, eer)
        self.converged_velocities[model_results_path].append(GeneratePoseErrorStats.has_converged_on_velocities(velocities))
        np.savetxt(sample_path / 'curvature.txt', [curvature])
        np.savetxt(sample_path / 'curvatures.txt', curvatures)
        

    def _add_erors(self, key, set, ser, eet, eer):
        data = self.errors[key]
        c = GeneratePoseErrorStats.has_converged(set, ser, eet, eer)
        for i, e in enumerate([set, ser, eet, eer, c]):
            data[i].append(e)
    def on_end_model_visit(self, model_results_path):
        print(model_results_path)
        def make_sorted_error_plot_converged(eidx, conv_idx, plt_name, transform_data=lambda x: x, ylabel='Error'):
            ax = plt.gca()
            end_errors = []
            model_errors = self.errors[model_results_path]
            for eer, has_conved in zip(model_errors[eidx], model_errors[conv_idx]):
                if has_conved:
                    end_errors.append(transform_data(eer))
            end_errors = sorted(end_errors)
            plt.ylabel(ylabel)

            plt.plot(end_errors)
            plt.autoscale()
            plt.legend(loc='lower right')
            plt.savefig(str(model_results_path / plt_name))
            plt.clf()
        matplotlib.rcParams['text.usetex'] = False
        ps = np.array(self.points_look_at[model_results_path])
        pc = ps[:, 0]
        pd = ps[:, 1]
        distances = np.linalg.norm(pc - pd, axis=-1)
        converged = self.errors[model_results_path][4]
        dist_to_conv = {}
        eps = 1e-5
        for i in range(len(ps)):
            d = distances[i]
            c = converged[i]
            matching_dist = -1
            for k in dist_to_conv:
                if abs(d - k) < eps:
                    matching_dist = k
                    break
            if matching_dist == -1:
                matching_dist = d
            if matching_dist not in dist_to_conv:
                dist_to_conv[matching_dist] = [c]
            else:
                dist_to_conv[matching_dist].append(c)
        dist_to_conv_rate = {k: sum(v) / len(v) for k, v in dist_to_conv.items()}

        dist_groups = dist_to_conv_rate.keys()
        plt.figure()
        plt.plot(dist_groups, [v for v in dist_to_conv_rate.values()])
        plt.savefig(model_results_path / 'conv_noise_look_at.png')
        plt.close()
        self.look_at_dist_to_conv_rate_dicts[model_results_path] = dist_to_conv_rate

        make_sorted_error_plot_converged(2, 4, 'sorted_error_translation.png', lambda x: x * 1000.0, ylabel='Error, mm')
        make_sorted_error_plot_converged(3, 4, 'sorted_error_rotation.png', lambda x: np.degrees(x), ylabel='Error, degrees')

    def on_end_multiple_models_visit(self, models_root):
        def make_threshold_plot(sidx, eidx, plt_name, transform_data=lambda x: x, xlabel='Error', insert_initial=True):
            ax = plt.gca()
            max_x = 0.0
            for i, k in enumerate(sorted(self.errors)):
                if i == 0 and insert_initial:
                    sets = sorted(transform_data(self.errors[k][sidx]))
                    max_x = max(max_x, sets[-1])
                    plt.ylim([0, len(sets)])
                    plt.plot(sets, range(1, len(sets) + 1), label='Initial')
                    ax.axhline(len(sets), c='k')
                eets = sorted(transform_data(self.errors[k][eidx]))
                max_x = max(max_x, eets[-1])
                plt.plot(eets, range(1, len(eets) + 1), label=self.path_to_name_dict[k])
                plt.ylabel('# of samples with error below x')
                plt.xlabel(xlabel)
            # plt.xlim([0, max_x])
            plt.autoscale()
            plt.legend(loc='lower right')
            plt.grid()
            plt.savefig(str(self.global_results_path / plt_name))
            plt.clf()

        make_threshold_plot(0, 2, 'threshold_end_error_translation.pdf', xlabel='Error (in m)', insert_initial=True)
        make_threshold_plot(1, 3, 'threshold_end_error_rotation.pdf', lambda x: np.degrees(x), xlabel='Error (in °)', insert_initial=True)
        make_threshold_plot(0, 2, 'threshold_end_error_translation_no_initial.pdf', xlabel='Error (in m)', insert_initial=False)
        make_threshold_plot(1, 3, 'threshold_end_error_rotation_no_initial.pdf', lambda x: np.degrees(x), xlabel='Error (in °)', insert_initial=False)
        
        for i, k in enumerate(sorted(self.errors)):
            converged = self.errors[k][4]
            cy = np.empty(len(converged))
            ctr = 0
            for j, has_converged in enumerate(converged):
                if has_converged:
                    ctr += 1
                cy[j] = ctr
            plt.plot(range(len(converged)), cy, label=self.path_to_name_dict[k])
            plt.ylabel('# of samples converged')
            plt.xlabel('# of samples seen')
        plt.legend(loc='lower right')
        plt.savefig(str(self.global_results_path / 'convergence.pdf'))
        plt.clf()
        xx = np.arange(10)
        step = 0.1

        for i, k in enumerate(sorted(self.errors)):
            converged = np.array(self.errors[k][4])
            visibilities = np.array(self.visibilities[k])
            cy = []
            for j in range(len(xx)):
                lo, hi = step * j, step * (j + 1)
                converged_bin = converged[(visibilities >= lo) & (visibilities < hi)]
                percentage_converged = (converged_bin.sum() / len(converged_bin)) * 100.0
                cy.append(percentage_converged)
            plt.plot(xx, cy, label=self.path_to_name_dict[k], marker='x')
        matplotlib.rcParams.update({'font.size': 11})
        plt.xlabel('Batch index, as a function of overlap', fontsize=11)
        plt.ylim([0.0, 110.0])
        plt.grid(linestyle='-')
        plt.hlines(100.0, -1, 11)
        plt.xlim([0, 9])
        plt.ylabel('Convergence rate, %', fontsize=11)
        plt.tight_layout()
        ax = plt.gca()
        # plt.xticks(xx + bar_width + bar_width/2, ['{}%-{}%'.format(int(i * step * 100), int((i + 1) * step * 100)) for i in range(len(xx))])
        leg = plt.legend(loc='upper left', ncol=1)
        plt.draw()
        bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
        xOffset = 0.01
        bb.x0 += xOffset
        bb.x1 += xOffset
        leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
        plt.savefig(str(models_root / 'overlap.pdf'))
        plt.clf()
        plt.close()
        plt.figure()

        for m in self.look_at_dist_to_conv_rate_dicts.keys():
            name = self.path_to_name_dict[m]
            name = name.replace('__look_at_with_noise', '')
            print('name', name)
            data = self.look_at_dist_to_conv_rate_dicts[m]

            d = dict(sorted(data.items(), key=lambda item: item[0]))

            plt.plot(d.keys(), d.values(), label=name, marker='x')

        plt.grid()
        plt.xlabel('Distance between focal points of starting and desired cameras, m')
        plt.ylim([0, 1.0])
        plt.ylabel('Convergence rate, %')
        plt.legend(loc='lower left')
        plt.savefig(str(self.global_results_path / 'look_at_dist_vs_convergence.pdf'))
        plt.clf()
        plt.close()
        # print(self.errors)
        columns = ['Start Error translation Mean, Cm', 'Start Error translation Std, Cm',
                    'Start Error rotation Mean, Deg', 'Start Error rotation Std, Deg',
                    'End Error Translation Mean, Cm', 'End Error translation Std, Cm',
                    'End Error rotation Mean, Deg', 'End Error rotation Std, Deg',
                    'Converged, % of samples', 'Error_mean_converged, cm',
                    'Error_std_converged, cm', 'Error_mean_converged, degrees',
                    'Error_std_converged, degrees',
                    'Velocity convergence, %', 'Mean curvature']
        error_df = pd.DataFrame(index=[str(k) for k in self.errors.keys()], columns=columns)
        # print(error_df)
        for i, k in enumerate(sorted(self.errors)):
            ks = str(k)
            error_df.loc[ks, columns[0]] = np.mean(self.errors[k][0]) * 100.0
            error_df.loc[ks, columns[1]] = np.std(self.errors[k][0]) * 100.0
            error_df.loc[ks, columns[2]] = np.degrees(np.mean(self.errors[k][1]))
            error_df.loc[ks, columns[3]] = np.degrees(np.std(self.errors[k][1]))
            error_df.loc[ks, columns[4]] = np.mean(self.errors[k][2]) * 100.0
            error_df.loc[ks, columns[5]] = np.std(self.errors[k][2]) * 100.0
            error_df.loc[ks, columns[6]] = np.degrees(np.mean(self.errors[k][3]))
            error_df.loc[ks, columns[7]] = np.degrees(np.std(self.errors[k][3]))
            error_df.loc[ks, columns[8]] = np.sum(self.errors[k][4]) / len(self.errors[k][4])

            c = np.array(self.errors[k][4])
            dt = np.array(self.errors[k][2])
            dtc = dt[c]
            drc = np.array(self.errors[k][3])[c]
            error_df.loc[ks, columns[9]] = np.mean(dtc) * 100.0
            error_df.loc[ks, columns[10]] = np.std(dtc) * 100.0

            error_df.loc[ks, columns[11]] = np.degrees(np.mean(drc))
            error_df.loc[ks, columns[12]] = np.degrees(np.std(drc))
            error_df.loc[ks, columns[13]] = np.sum(self.converged_velocities[k]) / len(self.converged_velocities[k])
            error_df.loc[ks, columns[14]] = np.sum(self.curvatures[k]) / len(self.curvatures[k])
            


        pd.set_option('display.max_colwidth', 200)
        print(error_df[[columns[8], columns[13], columns[14]]])
        error_df.to_csv(self.global_results_path / 'stats_error.csv', sep=';')
        # def save_txt(name, model_index, error_index):
        #     np.savetxt(str(models_root /   / 'index'))

class ComputeTrajectoryStats(PostRunOperation):
    statistics_and_name = {
        StatisticsType.max: 'Max',
        StatisticsType.min: 'Min',
        StatisticsType.mean: 'Mean',
        StatisticsType.rmse: 'rmse'
    }
    def __init__(self, path_to_name_dict, reference_dir, global_results_path, plot_per_sample):
        self.errors = {}
        self.evo_results = {}
        self.converged = {}
        self.path_to_name_dict = path_to_name_dict
        self.reference_dir = reference_dir
        self.trajectory_lengths = {}
        self.plot_per_sample = plot_per_sample
        self.global_results_path = global_results_path
    def visit_model_results(self, model_results_path):
        if model_results_path not in self.path_to_name_dict or model_results_path == self.reference_dir:
            pass
        else:
            super().visit_model_results(model_results_path)

        
    def get_pose_file(self, dir):
        import glob
        return glob.glob(str(dir) + '/*.kitti')[0]
    def compute_area(self, traj_curr, traj_ref):
        res = 0.0
        traj_curr_p = traj_curr[:, :3, 3]
        traj_ref_p = traj_ref[:, :3, 3]
        np.seterr(all='raise')
        for i in range(len(traj_curr) - 1):
            p11, p12 = traj_curr_p[i].copy(), traj_curr_p[i + 1].copy()
            p21, p22 = traj_ref_p[i].copy(), traj_ref_p[i + 1].copy()


            if np.dot(p12 - p11, p22 - p21) < 0:
                temp = p12.copy()
                p12 = p11
                p11 = temp

            p11_p12 = p12 - p11
            p11_p21 = p21 - p11
            p21_p22 = p22 - p21
            p22_p21 = p21 - p22
            p22_p12 = p12 - p22
            n11_12 = np.linalg.norm(p11_p12)
            n11_21 = np.linalg.norm(p11_p21)
            n22_12 = np.linalg.norm(p22_p12)
            n22_21 = np.linalg.norm(p22_p21)

            if n11_12 == 0.0 or n11_21 == 0:
                a1 = 0.0
            else:
                # a1 = 0.0
                # try:
                #     a1 = np.arccos(np.clip(np.dot(p11_p21, p11_p12) / (n11_21 * n11_12), -1.0, 1.0))
                # except:
                #     print(n11_12, n11_21)
                #     print(p11_p12, p11_p21)
                #     print(np.dot(p11_p21, p11_p12))
                #     print(np.dot(p11_p21, p11_p12) / (n11_21 * n11_12))
                a1 = 0.5 * np.linalg.norm(np.cross(p11_p21, p11_p12))

            if n22_12 == 0.0 or n22_21 == 0.0:
                a2 = 0.0
            else:
                # a2 = np.arccos(np.clip(np.dot(p22_p12, p22_p21) / (n22_21 * n22_12), -1.0, 1.0))
                a2 = 0.5 * np.linalg.norm(np.cross(p22_p12, p22_p12))
            # res += 0.5 * np.sin(a1) * n11_21 * n11_12 + 0.5 * np.sin(a2) * n22_21 * n22_12
            res += a1 + a2
        return res

    def read_pose_file(self, f):
        traj_T = []

        for l in f:
            traj_T.append(np.array([float(v) for v in l.split(' ')] + [0.0, 0.0, 0.0, 1.0]).reshape((4, 4)))
        return np.array(traj_T)
    
    # def trajectory_length(self, wTcurr, currTw):
    #     traj_length_meters = 0.0
    #     traj_length_degrees = 0.0
        
    #     for i in range(1, len(wTcurr)):
    #         iTw = currTw[i]
    #         wTprev = wTcurr[i - 1]
    #         iTprev = iTw @ wTprev
    #         t_norm = np.linalg.norm(iTprev[:3, 3], ord=2)
    #         traj_length_meters += t_norm
    #         tu = batch_rotation_matrix_to_axis_angle(iTprev[None, :3, :3])[0]
    #         traj_length_degrees += np.degrees(np.linalg.norm(tu, ord=2))
    #     return traj_length_meters, traj_length_degrees
    # def get_traj_stats(self, curr_traj, ref_traj):
    #     # curr_traj = self.read_pose_file(curr_file)
    #     # ref_traj = self.read_pose_file(ref_file)
    #     wTcurr = curr_traj
    #     wTref = ref_traj
    #     currTw = batch_homogeneous_inverse(wTcurr)
    #     currTref = np.matmul(currTw, wTref)
    #     t = currTref[:, :3, 3]

    #     R = currTref[:, :3, :3]
    #     curr_r_ref = batch_rotation_matrix_to_axis_angle(R)
    #     dists_r = np.linalg.norm(curr_r_ref, ord=2, axis=-1)
    #     # diff = curr_traj - ref_traj

    #     dists_t = np.linalg.norm(t, ord=2, axis=-1)
    #     curr_l_m, curr_l_deg = self.trajectory_length(wTcurr, currTw)
    #     ref_l_m, ref_l_deg = self.trajectory_length(wTref, batch_homogeneous_inverse(wTref))

    #     ratio_length_m = curr_l_m / ref_l_m if ref_l_m > 0 else 0.0
    #     ratio_length_deg = curr_l_deg / ref_l_deg if ref_l_deg > 0 else 0.0         

    #     return np.mean(dists_t), np.std(dists_t), np.mean(dists_r), np.std(dists_r), ratio_length_m, ratio_length_deg


    def visit_sample_results(self, model_results_path, sample_path):
        from evo.core import metrics
        from evo.tools import file_interface
        ape_t = metrics.APE(metrics.PoseRelation.translation_part)
        ape_r = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
        
        if model_results_path not in self.evo_results:
            # self.errors[model_results_path] = [[] for i in range(7)]
            self.evo_results[model_results_path] = {}
            self.converged[model_results_path] = {}
            self.trajectory_lengths[model_results_path] = {}
        
        ref_sample_dir = self.reference_dir / sample_path.name
        
        traj_curr_file = self.get_pose_file(sample_path)
        traj_ref_file = self.get_pose_file(ref_sample_dir)
        traj_ref = file_interface.read_kitti_poses_file(traj_ref_file)
        traj_curr = file_interface.read_kitti_poses_file(traj_curr_file)
        
        ape_t.process_data((traj_ref, traj_curr))
        ape_r.process_data((traj_ref, traj_curr))

        self.evo_results[model_results_path][sample_path.name] = (ape_t.get_result(), ape_r.get_result())
        
        # print(sample_path)
        if (sample_path / GeneratePoseErrorStats.ET_NAME).exists():
            with open(str(sample_path / GeneratePoseErrorStats.ET_NAME), 'r') as etf:
                set = float(etf.readline())
                for last_line in etf:
                    pass
                eet = float(last_line)
        else:
            d = np.load(sample_path / (GeneratePoseErrorStats.ET_NAME + '.npz'))['arr_0']
            set = d[0]
            eet = d[-1]
        if (sample_path / GeneratePoseErrorStats.ET_NAME).exists():
            with open(str(sample_path / GeneratePoseErrorStats.ER_NAME), 'r') as erf:
                ser = float(erf.readline())
                for last_line in erf:
                    pass
                eer = float(last_line)
        else:
            d = np.load(sample_path / (GeneratePoseErrorStats.ER_NAME + '.npz'))['arr_0']
            ser = d[0]
            eer = d[-1]

        stats_t = {}
        stats_t.update(ape_t.get_all_statistics())
        stats_t['Length'] = traj_curr.path_length
        if traj_ref.path_length > 0:
            stats_t['Length Ratio'] = traj_curr.path_length / traj_ref.path_length
        stats_r = ape_r.get_all_statistics()
        self.trajectory_lengths[model_results_path][sample_path.name] = (stats_t['Length'], stats_t['Length Ratio'])
        import yaml
        with open(str(sample_path / 'stats_translation.yaml'), 'w') as stats_t_f:
            yaml.dump(stats_t, stats_t_f)
        with open(str(sample_path / 'stats_rotation.yaml'), 'w') as stats_r_f:
            yaml.dump(stats_r, stats_r_f)
        from evo.tools import settings
        if self.plot_per_sample:
            settings.SETTINGS.plot_backend = 'Agg'
            from evo.tools import plot
            plot_mode = plot.PlotMode.xyz
            fig = plt.figure()
            ax = plot.prepare_axis(fig, plot_mode)
            plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
            plot.traj_colormap(ax, traj_curr, ape_t.error,
                            plot_mode, min_map=stats_t["min"], max_map=stats_t["max"])
            ax.legend()
            plt.savefig(sample_path / 'traj_comparison.pdf')
            plt.close()
            violin_translation = sns.violinplot(x=ape_t.error, cut=0)
            violin_translation.set(xlabel='APE (m)')
            plt.savefig(sample_path / 'violin_translation.png')
            plt.close()
            violin_rotation = sns.violinplot(x=ape_r.error, cut=0)
            violin_rotation.set(xlabel='APE (°)')
            plt.savefig(sample_path / 'violin_rotation.png')
            plt.close()
        
        self.converged[model_results_path][sample_path.name] = GeneratePoseErrorStats.has_converged(set, ser, eet, eer)
            


    def on_end_multiple_models_visit(self, models_root):

        
        results_path = self.global_results_path
        results_path.mkdir(exist_ok=True)
        from evo.core.result import merge_results
        aggregated_results_t = {}
        aggregated_results_r = {}
        aggregated_results_t_c = {}
        aggregated_results_r_c = {}
        length_ratios_c = {}
        
        for i, m in enumerate(self.evo_results):
            tl = [self.evo_results[m][a][0] for a in self.evo_results[m]]
            for v in tl:
                v.np_arrays['error_array'] = np.array(v.np_arrays['error_array'])
            tl_c = [self.evo_results[m][a][0] for a in self.evo_results[m] if self.converged[m][a]]
            aggregated_results_t[m] = merge_results(tl)
            
            aggregated_results_r[m] = merge_results([self.evo_results[m][a][1] for a in self.evo_results[m]])
            aggregated_results_t_c[m] = merge_results(tl_c)
            aggregated_results_r_c[m] = merge_results([self.evo_results[m][a][1] for a in self.evo_results[m] if self.converged[m][a]])
            length_ratios_c[m] = []

            for s in self.evo_results[m]:
                if i == 0: # Generate violin plot per sample
                    pass
                    # ms: Dict[str, Tuple[Result, Result]] = {mm.name.split('__')[0]: self.evo_results[mm][s] for mm in self.evo_results}
                    # ms_t = {msk: v[0].np_arrays['error_array'] for msk, v in ms.items()}
                    # df = pd.DataFrame(columns=['Method', 'APE(m)'])
                    # for k in ms_t:
                    #     vals = [{'Method': k,'APE(m)' :ms_t[k][j]} for j in range(len(ms_t[k]))]
                    #     df = df.append(vals)

                    # violin_plot = sns.violinplot(data=df, x='Method', y='APE(m)', hue='Method', cut=0)

                    # plt.savefig(results_path / f'APE_translation_{s}.pdf')
                    # plt.close()
                    # ms_r = {msk: v[1].np_arrays['error_array'] for msk, v in ms.items()}
                    # df = pd.DataFrame(columns=['Method', 'APE(°)'])
                    # for k in ms_r:
                    #     vals = [{'Method': k,'APE(°)' :ms_r[k][j]} for j in range(len(ms_r[k]))]
                    #     df = df.append(vals)

                    # violin_plot = sns.violinplot(data=df, x='Method', y='APE(°)', hue='Method', cut=0)

                    # plt.savefig(results_path / f'APE_rot_{s}.pdf')
                    # plt.close()
                    
                if self.converged[m][s]:
                    length_ratios_c[m].append(self.trajectory_lengths[m][s][1])



        df_t_all = pd.DataFrame(columns=['Method', 'APE(m)'])
        df_r_all = pd.DataFrame(columns=['Method', 'APE(°)'])
        for m in aggregated_results_t:
            key = m.name.split('__')[0]
            vals = [{'Method': key, 'APE(m)': v} for v in aggregated_results_t[m].np_arrays['error_array']]
            df_t_all = df_t_all.append(vals)
            vals = [{'Method': key, 'APE(°)': v} for v in aggregated_results_r[m].np_arrays['error_array']]
            df_r_all = df_r_all.append(vals)
        violin_plot = sns.violinplot(data=df_t_all, x='Method', y='APE(m)', hue='Method', cut=0)
        plt.savefig(results_path / f'APE_translation_global.pdf')
        plt.close()
        violin_plot = sns.violinplot(data=df_r_all, x='Method', y='APE(°)', hue='Method', cut=0)
        plt.savefig(results_path / f'APE_rotation_global.pdf')
        plt.close()
        df_t_all_c = pd.DataFrame(columns=['Method', 'APE(m)'])
        df_r_all_c = pd.DataFrame(columns=['Method', 'APE(°)'])
        for m in aggregated_results_t_c:
            key = m.name.split('__')[0]
            vals = [{'Method': key, 'APE(m)': v} for v in aggregated_results_t_c[m].np_arrays['error_array']]
            df_t_all_c = df_t_all_c.append(vals)
            vals = [{'Method': key, 'APE(°)': v} for v in aggregated_results_r_c[m].np_arrays['error_array']]
            df_r_all_c = df_r_all_c.append(vals)
        violin_plot = sns.violinplot(data=df_t_all_c, x='Method', y='APE(m)', hue='Method', cut=0)
        plt.savefig(results_path / f'APE_translation_converged.pdf')
        plt.close()
        violin_plot = sns.violinplot(data=df_r_all_c, x='Method', y='APE(°)', hue='Method', cut=0)
        plt.savefig(results_path / f'APE_rotation_converged.pdf')
        plt.close()

        

        indices = [str(k) for k in self.path_to_name_dict.keys()]
        cols = ['mean APE, cm', 'std APE, cm', 'mean APE, degrees', 'std APE, degrees', 'mean length ratio', 'std length ratio']
        stat_df = pd.DataFrame(index=indices, columns=cols)
        print(length_ratios_c)
        for k in self.evo_results:
            ks = str(k)
            stats_t = aggregated_results_t_c[k].stats
            stats_r = aggregated_results_r_c[k].stats
            print(stats_t, stats_r)
            stat_df.loc[ks, cols[0]] = stats_t['mean'] * 100.0
            stat_df.loc[ks, cols[1]] = stats_t['std'] * 100.0
            stat_df.loc[ks, cols[2]] = stats_r['mean']
            stat_df.loc[ks, cols[3]] = stats_r['std']
            stat_df.loc[ks, cols[4]] = np.mean(length_ratios_c[k])
            stat_df.loc[ks, cols[5]] = np.std(length_ratios_c[k])
            


            
            

        stat_df.to_csv(results_path / 'trajectory_stats.csv', sep=';')


        

        # fig = plt.figure()
        # for k in self.errors:
        #     data = self.errors[k][1]
        #     data = sorted(np.array(data) * 100.0) 
        #     plt.plot(data, label=k.name)

        
            
        # plt.legend()
        # plt.ylabel('APE, cm')
        # plt.savefig(models_root / 'APE_translation.pdf')
        # plt.close()
        # fig = plt.figure()
        # for k in self.errors:
        #     data = self.errors[k][3]
        #     data = np.degrees(sorted(data))
        #     plt.plot(data, label=k.name)            
        # plt.legend()
        # plt.ylabel('APE, degrees')
        # plt.savefig(models_root / 'APE_rotation.pdf')
        # plt.close()
        # fig = plt.figure()
        # for k in self.errors:
        #     data = self.errors[k][0]
        #     data = sorted(np.array(data)) 
        #     plt.plot(data, label=k.name)            
        # plt.legend()
        # plt.ylabel('Area, m2')
        # plt.savefig(models_root / 'area.pdf')
        # plt.close()
        # fig = plt.figure()
        # for k in self.errors:
        #     data = self.errors[k][5]
        #     data = sorted(data) 
        #     plt.plot(data, label=k.name)           
        # plt.legend()
        # plt.ylabel('Translation length ratios')
        # plt.savefig(models_root / 'length_ratio_translation.pdf')
        # plt.close()
        
        # indices = [str(k) for k in self.path_to_name_dict.keys()]
        # cols = ['mean Area, cm²', 'Area std, cm²', 'mean APE, cm', 'APE std, cm',
        #         'mean ARE, degrees', 'ARE std, degrees', 'mean length ratio translation', 'mean length ratio, deg']
        # stat_df = pd.DataFrame(index=indices, columns=cols)
        # for k in self.errors:
        #     ks = str(k)
        #     stat_df.loc[ks, cols[0]] = np.mean(self.errors[k][0]) * 100 ** 2
        #     stat_df.loc[ks, cols[1]] = np.std(self.errors[k][0]) * 100 ** 2
        #     stat_df.loc[ks, cols[2]] = np.mean(self.errors[k][1]) * 100
        #     stat_df.loc[ks, cols[3]] = np.std(self.errors[k][1]) * 100
        #     stat_df.loc[ks, cols[4]] = np.degrees(np.mean(self.errors[k][3]))
        #     stat_df.loc[ks, cols[5]] = np.degrees(np.std(self.errors[k][3]))
        #     stat_df.loc[ks, cols[6]] = np.mean(self.errors[k][5])
        #     stat_df.loc[ks, cols[7]] = np.mean(self.errors[k][6])
            

        # stat_df.to_csv(models_root / 'trajectory_stats.csv', sep=';')
