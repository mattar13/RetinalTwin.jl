# ============================================================
# ode_system.jl — Full ODE right-hand side for the retinal column
# Spec §7.4
# ============================================================

"""
    zipper_rhs!(du, u, p, t)

Zipper the global state/params into per-cell RHS calls.
`p = (col::RetinalColumn, sidx::StateIndex, connections::Vector{ConnectionDef})`.
"""
function zipper_rhs!(du, u, p, t)
    col, sidx, conns = p
    pop = col.pop

    # --- Stimulus ---
    Phi = compute_stimulus(col.stimulus, t)

    # --- Phase 1: Read neurotransmitter concentrations ---
    # Photoreceptor glutamate
    glu_rod_mean  = mean_nt(u, sidx.rod, ROD_GLU_INDEX, ROD_STATE_VARS, pop.n_rod)
    glu_cone_mean = mean_nt(u, sidx.cone, CONE_GLU_INDEX, CONE_STATE_VARS, pop.n_cone)
    glu_pr_mean   = weighted_mean(glu_rod_mean, pop.n_rod, glu_cone_mean, pop.n_cone)

    # ON-bipolar glutamate (var 4 in ON-BC state)
    glu_on_mean = mean_nt(u, sidx.on_bc, 4, 4, pop.n_on)

    # OFF-bipolar glutamate (var 4 in OFF-BC state)
    glu_off_mean = mean_nt(u, sidx.off_bc, 4, 4, pop.n_off)

    # Amacrine NTs
    gly_a2_mean = mean_nt(u, sidx.a2, 3, 3, pop.n_a2)        # glycine
    gaba_mean   = mean_nt(u, sidx.gaba_ac, 3, 3, pop.n_gaba)  # GABA
    da_mean     = mean_nt(u, sidx.da_ac, 3, 3, pop.n_dopa)    # dopamine

    # Mean voltages for gap junctions and feedback
    hc_V_mean = mean_voltage(u, sidx.hc, 1, 3, pop.n_hc)

    # HC feedback to photoreceptors (simple additive current)
    I_hc_fb = hc_feedback(hc_V_mean)

    # --- Phase 2: Compute K+ currents for Müller/RPE ---
    total_I_K = 0.0
    for i in 1:pop.n_rod
        offset = sidx.rod[1] + (i - 1) * ROD_STATE_VARS
        total_I_K += photoreceptor_K_current(view(u, offset:offset + ROD_STATE_VARS - 1), col.rod_params)
    end
    for i in 1:pop.n_cone
        offset = sidx.cone[1] + (i - 1) * CONE_STATE_VARS
        total_I_K += photoreceptor_K_current(view(u, offset:offset + CONE_STATE_VARS - 1), col.cone_params)
    end

    # Total glutamate released (for Müller cell sensing)
    total_glu_release = glu_pr_mean * (pop.n_rod + pop.n_cone) +
                        glu_on_mean * pop.n_on +
                        glu_off_mean * pop.n_off

    # --- Phase 3: Update each cell population ---

    # Photoreceptors (rods)
    for i in 1:pop.n_rod
        offset = sidx.rod[1] + (i - 1) * ROD_STATE_VARS
        rod_cell_rhs!(view(du, offset:offset + ROD_STATE_VARS - 1),
                      view(u, offset:offset + ROD_STATE_VARS - 1),
                      (col.rod_params, Phi, I_hc_fb), t)
    end

    # Photoreceptors (cones)
    for i in 1:pop.n_cone
        offset = sidx.cone[1] + (i - 1) * CONE_STATE_VARS
        cone_cell_rhs!(view(du, offset:offset + CONE_STATE_VARS - 1),
                       view(u, offset:offset + CONE_STATE_VARS - 1),
                       (col.cone_params, Phi, I_hc_fb), t)
    end

    # Horizontal cells
    for i in 1:pop.n_hc
        offset = sidx.hc[1] + (i - 1) * 3
        V_hc = u[offset]

        # Excitatory input from PR glutamate
        s_glu = u[offset + 2]
        I_exc = synaptic_current(5.0, s_glu, V_hc, 0.0)  # g=5 nS, E=0 mV

        # Gap junction with other HCs
        I_gap = 0.0
        g_gap = 0.5  # nS
        for j in 1:pop.n_hc
            if j != i
                other_offset = sidx.hc[1] + (j - 1) * 3
                I_gap += g_gap * (u[other_offset] - V_hc)
            end
        end

        horizontal_cell_rhs!(view(du, offset:offset+2),
                             view(u, offset:offset+2),
                             (col.hc_params, I_exc, I_gap, glu_pr_mean), t)
    end

    # ON-Bipolar cells
    for i in 1:pop.n_on
        offset = sidx.on_bc[1] + (i - 1) * 4
        V_on = u[offset]

        # Inhibitory: GABA from GABAergic amacrines
        I_inh = synaptic_current(3.0, gaba_mean, V_on, -70.0)

        # Modulatory: dopamine gain
        I_mod = 0.0  # dopamine modulates TRPM1 gain inside update_on_bipolar

        on_bipolar_cell_rhs!(view(du, offset:offset+3),
                             view(u, offset:offset+3),
                             (col.on_params, col.mglur6_params, glu_pr_mean, I_inh, I_mod), t)
    end

    # OFF-Bipolar cells
    for i in 1:pop.n_off
        offset = sidx.off_bc[1] + (i - 1) * 4
        V_off = u[offset]

        # Inhibitory: glycine from A2 + GABA from GABAergic
        I_inh = synaptic_current(5.0, gly_a2_mean, V_off, -80.0) +
                synaptic_current(3.0, gaba_mean, V_off, -70.0)

        I_mod = 0.0

        off_bipolar_cell_rhs!(view(du, offset:offset+3),
                              view(u, offset:offset+3),
                              (col.off_params, glu_pr_mean, I_inh, I_mod), t)
    end

    # A2 Amacrine cells
    for i in 1:pop.n_a2
        offset = sidx.a2[1] + (i - 1) * 3
        V_a2 = u[offset]

        # Excitatory: glutamate from ON-bipolars
        I_exc = synaptic_current(8.0, glu_on_mean, V_a2, 0.0)

        # Inhibitory: GABA from GABAergic amacrines (reciprocal — OP oscillator)
        I_inh = synaptic_current(10.0, gaba_mean, V_a2, -70.0)

        # Modulatory: dopamine
        I_mod_a2 = -da_mean * 1.0 * (V_a2 - (-60.0))  # reduces excitability

        a2_cell_rhs!(view(du, offset:offset+2),
                     view(u, offset:offset+2),
                     (col.a2_params, I_exc, I_inh, I_mod_a2), t)
    end

    # GABAergic Amacrine cells
    for i in 1:pop.n_gaba
        offset = sidx.gaba_ac[1] + (i - 1) * 3
        V_gaba = u[offset]

        # Excitatory: glutamate from ON-bipolars
        I_exc = synaptic_current(6.0, glu_on_mean, V_gaba, 0.0)

        # Inhibitory: glycine from A2 (reciprocal — OP oscillator)
        I_inh = synaptic_current(10.0, gly_a2_mean, V_gaba, -80.0)

        # Modulatory: dopamine
        I_mod_gaba = -da_mean * 1.0 * (V_gaba - (-60.0))

        gaba_cell_rhs!(view(du, offset:offset+2),
                       view(u, offset:offset+2),
                       (col.gaba_params, I_exc, I_inh, I_mod_gaba), t)
    end

    # DA Amacrine cells
    for i in 1:pop.n_dopa
        offset = sidx.da_ac[1] + (i - 1) * 3
        V_da = u[offset]

        # Excitatory: glutamate from ON-bipolars
        I_exc = synaptic_current(3.0, glu_on_mean, V_da, 0.0)

        da_cell_rhs!(view(du, offset:offset+2),
                     view(u, offset:offset+2),
                     (col.da_params, I_exc), t)
    end

    # Ganglion cells
    for i in 1:pop.n_gc
        offset = sidx.gc[1] + (i - 1) * 2
        V_gc = u[offset]

        # Excitatory: glutamate from ON and OFF bipolars
        I_exc = synaptic_current(5.0, glu_on_mean, V_gc, 0.0) +
                synaptic_current(5.0, glu_off_mean, V_gc, 0.0)

        # Inhibitory: glycine from A2 + GABA
        I_inh = synaptic_current(3.0, gly_a2_mean, V_gc, -80.0) +
                synaptic_current(3.0, gaba_mean, V_gc, -70.0)

        ganglion_cell_rhs!(view(du, offset:offset+1),
                           view(u, offset:offset+1),
                           (col.gc_params, I_exc, I_inh), t)
    end

    # Müller glia
    for i in 1:pop.n_muller
        offset = sidx.muller[1] + (i - 1) * 4
        muller_cell_rhs!(view(du, offset:offset+3),
                         view(u, offset:offset+3),
                         (col.muller_params, total_I_K, total_glu_release * 0.01), t)
    end

    # RPE
    for i in 1:pop.n_rpe
        offset = sidx.rpe[1] + (i - 1) * 2
        rpe_cell_rhs!(view(du, offset:offset+1),
                      view(u, offset:offset+1),
                      (col.rpe_params, total_I_K), t)
    end

    return nothing
end

"""
    retinal_column_rhs!(du, u, p, t)

In-place ODE right-hand side for the full retinal column.
`p = (col::RetinalColumn, sidx::StateIndex, connections::Vector{ConnectionDef})`.

This function:
1. Computes stimulus
2. Reads population-averaged neurotransmitter concentrations
3. Computes synaptic currents for each postsynaptic cell
4. Updates each cell type's derivatives
"""
function retinal_column_rhs!(du, u, p, t)
    zipper_rhs!(du, u, p, t)
    return nothing
end
