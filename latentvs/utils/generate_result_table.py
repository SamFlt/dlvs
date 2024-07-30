'''
Generate Latex table from multiple csv files, output by the inference script
'''


from pathlib import Path
import pandas as pd
import numpy as np
class MethodData():
    ETM = 'Error_mean_converged, cm'
    ETSTD = 'Error_std_converged, cm'
    
    ERM = 'Error_mean_converged, degrees'
    ERSTD = 'Error_std_converged, cm'
    CONV = 'Converged, % of samples'
    MAPET ='mean APE, cm'
    SAPET = 'std APE, cm'
    MAPER = 'mean APE, degrees'
    SAPER = 'std APE, degrees'
    MAL =  'mean length ratio'
    SAL =  'std length ratio'
    
    def __init__(self, method_name, global_results_path_clean, method_str, global_results_path_perturbed=None):
        self.method_name = method_name
        self.gp = global_results_path_clean
        self.gp_p = global_results_path_perturbed
        self.sm = method_str
    def get(self, perturbed):
        p = self.gp if not perturbed else self.gp_p
        pose_stats = pd.read_csv(p / 'stats_error.csv', delimiter=';')
        traj_stats = pd.read_csv(p / 'trajectory_stats.csv', delimiter=';')
        data = pose_stats[pose_stats['Unnamed: 0'].str.contains(self.sm)]
        traj_data = traj_stats[traj_stats['Unnamed: 0'].str.contains(self.sm)]
        d = data.to_dict()
        td = traj_data.to_dict()
        l = [MethodData.ETM, MethodData.ETSTD, MethodData.ERM, MethodData.ERSTD, MethodData.CONV]
        tl = [MethodData.MAPET, MethodData.SAPET, MethodData.MAPER, MethodData.SAPER, MethodData.MAL, MethodData.SAL]
        print(self.method_name)
        for k in l:
            print(k, list(d[k].values()))
        values = [list(d[k].values())[0] for k in l]
        print(td)
        for k in tl:
            print(k, list(td[k].values()))

        tl_values = [list(td[k].values())[0] for k in tl]
        values[0] *= 10 # To Mm
        values[1] *= 10
        values[4] *= 100 # To percentage

        # tl_values[0] *= 10
        # tl_values[1] *= 10
        
        return values, tl_values
            
        

if __name__ == '__main__':

    datas = [
        ('DVS', r'DVS \cite{Collewet08c}', 1, None),
        ('aevs', r'AEVS \cite{Felton21b}', 1, None),
        ('pose_regressor', r'Pose estimator, e.g. \cite{Bateux18a}', 0, 0),
        ('MLVS_k=1_oversampling', 'Ours, $K=1$', 0, 0),
        ('MLVS_k=50_oversampling', 'Ours, $K=50$', 0, 0),
    ]
    strs_method = [s[0] + '__look_at_with_noise_rz' for s in datas]
    print(strs_method)

    g1c = Path('~/mlvs_global_results/clean_300').expanduser()
    g2c = Path('~/mlvs_global_results/clean_1500').expanduser()
    g1p = Path('~/mlvs_global_results/augmented_300').expanduser()
    g2p = Path('~/mlvs_global_results/augmented_1500').expanduser()
    
    
    gclean = [g1c, g2c]
    gperturbed = [g1p, g2p]
    gs = [gclean[data[2]] for data in datas]
    gps = [gperturbed[data[3]] if data[3] is not None else None for data in datas]
    method_names = [s[1] for s in datas]
    method_datas = [MethodData(m, gp, sm, gpp) for m,gp,sm, gpp in zip(method_names, gs, strs_method, gps)]
    res = []
    final_str = ''
    round_count = 2
    for md in method_datas:
        has_perturbed_data = md.gp_p is not None
        values, traj_values = md.get(False)

        values[:4] = np.round(values[:4], round_count)
        values[4] = np.round(values[4], 1)
        traj_values = np.round(traj_values, round_count)
        etm, etstd, erm, erstd, conv = values
        clean_separator = r'\hline' if not has_perturbed_data else r'\cline{2-8}'
        if has_perturbed_data:
            final_str += '\multirow{2}{*}{' + md.method_name + '} &'
        else:
            final_str += fr'{md.method_name} &'
        final_str += fr'\cmark & {conv} & {etm} & {etstd} & {erm} & {erstd} &'
        mapet, sapet, maper, saper, mal, sal = traj_values
        final_str += fr'{mapet} &{sapet} & {maper} & {saper} & {mal} & {sal}\\ {clean_separator}' + '\n'
        if has_perturbed_data:
            values, traj_values = md.get(True)
            values[:4] = np.round(values[:4], round_count)
            values[4] = np.round(values[4], 1)
            traj_values = np.round(traj_values, round_count)
            etm, etstd, erm, erstd, conv = values
            final_str += fr'&\xmark & {conv} & {etm} & {etstd} & {erm} & {erstd} &'
            mapet, sapet, maper, saper, mal, sal = traj_values
            final_str += fr'{mapet} & {sapet} & {maper} & {saper} & {mal} & {sal}\\ \hline' + '\n'
    print(final_str)