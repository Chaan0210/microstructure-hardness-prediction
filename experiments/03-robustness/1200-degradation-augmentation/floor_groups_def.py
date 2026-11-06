import re
def grp(k):
    if k == 'noise': return 'noise'
    if re.search(r'(u_ph|m_ph|z_f|^sp_|spfrac|dark_span|f_dark|sp_frac)', k): return 'phase'
    if re.search(r'(het4|sizefield|tile_logA|tile8|m_la_std|m_la_f|m_la_tile|logA_std|eqd_q90_q10|logA_skew|logA_bimod|vstd_sizefield|vstd_tile8|vstd_grain_logA)', k): return 'het'
    if re.search(r'(m_el|ar_|orient|stcoh|solidity|boundary_coh|vstd_ar|vstd_orient|vstd_solid)', k): return 'elong'
    if re.search(r'(pore)', k): return 'pore'
    if re.search(r'(m_la_mean|grain_|n_obj|obj_per|aw_|eqd|coarse|fine|neighbors|density|mean_area|vstd_)', k): return 'size'
    return 'other'
