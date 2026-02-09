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

    # Rods: dark state — depolarized, tonic glutamate release
    for i in 1:p.n_rod
        offset = sidx.rod[1] + (i - 1) * 6
        u0[offset]     = 0.0                    # R* = 0 (no light)
        u0[offset + 1] = col.rod_params.G_dark   # G = G_dark
        u0[offset + 2] = col.rod_params.Ca_dark  # Ca = Ca_dark
        u0[offset + 3] = -40.0                  # V ≈ -40 mV (dark)
        u0[offset + 4] = 0.0                    # h (I_H gate, low in dark)
        u0[offset + 5] = 0.5                    # Glu ≈ tonic release
    end

    # Cones: same pattern as rods with cone dark values
    for i in 1:p.n_cone
        offset = sidx.cone[1] + (i - 1) * 6
        u0[offset]     = 0.0
        u0[offset + 1] = col.cone_params.G_dark
        u0[offset + 2] = col.cone_params.Ca_dark
        u0[offset + 3] = -40.0
        u0[offset + 4] = 0.0
        u0[offset + 5] = 0.5
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
