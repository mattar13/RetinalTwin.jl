# ============================================================
# retinal_column.jl — Column assembly and initial conditions
# Spec §7.6
# ============================================================

"""
    dark_adapted_state(col::RetinalColumn, sidx::StateIndex)

Compute initial conditions for a dark-adapted retina.
Returns flat state vector u0.
"""
function dark_adapted_state(col::RetinalColumn, sidx::StateIndex)
    u0 = zeros(sidx.total)
    p = col.pop

    # Rods: biophysical dark state from docs/rod_photoreceptor_model.md
    for i in 1:p.n_rod
        offset = sidx.rod[1] + (i - 1) * ROD_STATE_VARS
        u0[offset]      = -36.186   # V
        u0[offset + 1]  = 0.430     # mKv
        u0[offset + 2]  = 0.999     # hKv
        u0[offset + 3]  = 0.436     # mCa
        u0[offset + 4]  = 0.642     # mKCa
        u0[offset + 5]  = 0.0966    # Ca_s
        u0[offset + 6]  = 0.0966    # Ca_f
        u0[offset + 7]  = 80.929    # Cab_ls
        u0[offset + 8]  = 29.068    # Cab_hs
        u0[offset + 9]  = 80.929    # Cab_lf
        u0[offset + 10] = 29.068    # Cab_hf
        u0[offset + 11] = 0.0       # Rh
        u0[offset + 12] = 0.0       # Rhi
        u0[offset + 13] = 0.0       # Tr
        u0[offset + 14] = 0.0       # PDE
        u0[offset + 15] = 0.3       # Ca_photo
        u0[offset + 16] = 34.88     # Cab_photo
        u0[offset + 17] = 2.0       # cGMP
        u0[offset + 18] = rod_glutamate_release(u0[offset], col.rod_params)  # Glu
    end

    # Cones: same pattern as rods with cone dark values
    G_ss_cone = col.cone_params.alpha_G / col.cone_params.beta_G
    for i in 1:p.n_cone
        offset = sidx.cone[1] + (i - 1) * CONE_STATE_VARS
        u0[offset]     = 0.0
        u0[offset + 1] = G_ss_cone               # G = steady-state cGMP
        u0[offset + 2] = col.cone_params.Ca_dark
        u0[offset + 3] = -40.0
        u0[offset + 4] = 0.0
        u0[offset + CONE_GLU_INDEX - 1] = 0.5
    end

    # Horizontal cells: partially depolarized by tonic glutamate
    for i in 1:p.n_hc
        offset = sidx.hc[1] + (i - 1) * 3
        u0[offset]     = -50.0   # V (depolarized by PR glutamate)
        u0[offset + 1] = 0.1     # w
        u0[offset + 2] = 0.5     # s_Glu (tracking PR glutamate)
    end

    # ON-Bipolar: hyperpolarized in dark (high Glu → mGluR6 active → TRPM1 closed)
    for i in 1:p.n_on
        offset = sidx.on_bc[1] + (i - 1) * 4
        u0[offset]     = -60.0   # V (hyperpolarized)
        u0[offset + 1] = 0.0     # w
        u0[offset + 2] = 0.8     # S_mGluR6 (high, Glu is high)
        u0[offset + 3] = 0.1     # Glu release (low, cell hyperpolarized)
    end

    # OFF-Bipolar: depolarized in dark (receiving glutamate directly)
    for i in 1:p.n_off
        offset = sidx.off_bc[1] + (i - 1) * 4
        u0[offset]     = -40.0   # V (somewhat depolarized)
        u0[offset + 1] = 0.2     # w
        u0[offset + 2] = 0.5     # s_Glu (tracking PR glutamate)
        u0[offset + 3] = 0.3     # Glu release
    end

    # A2 Amacrines: near resting in dark
    for i in 1:p.n_a2
        offset = sidx.a2[1] + (i - 1) * 3
        u0[offset]     = -60.0   # V
        u0[offset + 1] = 0.0     # w
        u0[offset + 2] = 0.0     # Gly
    end

    # GABAergic Amacrines
    for i in 1:p.n_gaba
        offset = sidx.gaba_ac[1] + (i - 1) * 3
        u0[offset]     = -60.0   # V
        u0[offset + 1] = 0.0     # w
        u0[offset + 2] = 0.0     # GABA
    end

    # DA Amacrine
    for i in 1:p.n_dopa
        offset = sidx.da_ac[1] + (i - 1) * 3
        u0[offset]     = -60.0   # V
        u0[offset + 1] = 0.0     # w
        u0[offset + 2] = 0.0     # DA
    end

    # Ganglion cells
    for i in 1:p.n_gc
        offset = sidx.gc[1] + (i - 1) * 2
        u0[offset]     = -65.0   # V
        u0[offset + 1] = 0.0     # w
    end

    # Müller glia: resting K+ levels
    for i in 1:p.n_muller
        offset = sidx.muller[1] + (i - 1) * 4
        u0[offset]     = -80.0                       # V_M (highly K+ permeable)
        u0[offset + 1] = col.muller_params.K_o_rest  # K+_o endfoot
        u0[offset + 2] = col.muller_params.K_o_rest  # K+_o stalk
        u0[offset + 3] = 0.0                         # Glu_o
    end

    # RPE: resting
    for i in 1:p.n_rpe
        offset = sidx.rpe[1] + (i - 1) * 2
        u0[offset]     = -60.0                     # V_RPE
        u0[offset + 1] = col.rpe_params.K_sub_rest # K+_sub
    end

    return u0
end
