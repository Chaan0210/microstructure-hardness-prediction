"""Feature-group ablation files for the 044 slot (u1dmavgH2Z-anghcor): one npz per dropped group (noise column always kept)."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np, json
F = np.load(_repo_io('../push-800/feat-u1dmavgH2Z-anghcor-train.npz'), allow_pickle=True); keys = [str(k) for k in F['keys']]
G = {'size': 'aw_eqd_mean aw_eqd_q50 aw_eqd_q90 aw_inv_sqrt_d eqd_q10 eqd_q90 grain_eqd_mean grain_eqd_med grain_eqd_q90_q10 grain_inv_sqrt_d grain_logA_std grain_mean_area_j grain_density_j obj_per_kpx logA_bimodality logA_skew coarse_area_frac fine_area_frac',
     'shape': 'ar_med ar_q90 ar_wmean solidity_wmean neighbors_mean neighbors_std orient_coherence boundary_coherence',
     'intensity': 'grain_I_std_mean obj_I_cv obj_I_spread_vs_noise dark_span_vs_noise img_std coverage',
     'pore': 'pore_d_mean pore_frac pore_n pore_per_grain pore_to_grain_size px_pore_d px_pore_ecc px_pore_frac px_pore_n',
     'het': 'sizefield_plane_r2 sizefield_range_s32 sizefield_ratio sizefield_std_s16 sizefield_std_s32 tile8_logA_std tile_logA_std',
     'sp': 'sp_ar sp_contiguity sp_field_std_s32 sp_to_matrix_size sp_z_mean tile_spfrac_std',
     'vstd': ' '.join(k for k in keys if k.startswith('vstd_')), 'stcoh': ' '.join(k for k in keys if k.startswith('stcoh_')),
     'unet_ph': 'u_ph u_ph_hard u_ph_f32 m_ph_hard het4_u het4_u_mx', 'unet_m': 'm_el_glob m_el_loc16 m_el_pix m_el_coh m_la_std m_la_f16 m_la_f32 m_la_tile8 m_la_mean',
     'z': 'z_f25 z_f50', 'ang': 'ang_c4 ang_s4', 'xsh': 'het4_u_x_m_el_pix sizefield_std_s32_x_m_el_pix sh_cor15'}
allg = [c for v in G.values() for c in v.split()]; assert len(allg) == len(set(allg)) and all(c in keys for c in allg), set(allg) - set(keys)
print('ungrouped', [k for k in keys if k not in allg])
for g, cols in G.items():
    keep = [j for j, k in enumerate(keys) if k not in cols.split()]
    np.savez(_repo_io(f'abl/feat-drop-{g}-train.npz'), ids=F['ids'], X=F['X'][:, keep], keys=np.array(keys)[keep]); print(g, len(cols.split()), '->', len(keep))
np.savez(_repo_io('abl/feat-drop-none-train.npz'), ids=F['ids'], X=F['X'], keys=np.array(keys))
json.dump(G, open(_repo_io('abl/groups.json'), 'w'), indent=1)
