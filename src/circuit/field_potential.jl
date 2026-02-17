# ============================================================
# field_potential.jl — ERG computation and OP extraction
# Spec §5, §7.7, Appendix B
# This will be a major rework of course...
# ============================================================

using DSP
using FFTW

"""
    compute_erg(sol, col::RetinalColumn, sidx::StateIndex)

Compute ERG field potential and per-component decomposition from ODE solution.
Returns (erg::Vector{Float64}, components::Dict{Symbol, Vector{Float64}}).
"""
function compute_erg(sol, col::RetinalColumn, sidx::StateIndex)
    n_t = length(sol.t)
    p = col.pop
    w = col.erg_weights

    erg = zeros(n_t)
    components = Dict{Symbol, Vector{Float64}}(
        :a_wave   => zeros(n_t),
        :b_wave   => zeros(n_t),
        :d_wave   => zeros(n_t),
        :OPs      => zeros(n_t),
        :P3       => zeros(n_t),
        :c_wave   => zeros(n_t),
        :ganglion => zeros(n_t),
    )

    for ti in 2:n_t
        u      = sol.u[ti]
        u_prev = sol.u[ti - 1]
        dt     = sol.t[ti] - sol.t[ti - 1]

        if dt <= 0.0
            continue
        end

        # Transmembrane current ≈ C_m * dV/dt for each cell type

        # --- Photoreceptors (a-wave) ---
        I_rod = 0.0
        for i in 1:p.n_rod
            offset = sidx.rod[1] + (i - 1) * ROD_STATE_VARS + (ROD_V_INDEX - 1)
            I_rod += col.rod_params.C_m * (u[offset] - u_prev[offset]) / dt
        end
        I_cone = 0.0
        for i in 1:p.n_cone
            offset = sidx.cone[1] + (i - 1) * CONE_STATE_VARS + (CONE_V_INDEX - 1)
            I_cone += col.cone_params.C_m * (u[offset] - u_prev[offset]) / dt
        end
        pr_contrib = w.rod * I_rod + w.cone * I_cone
        components[:a_wave][ti] = pr_contrib

        # --- ON-Bipolar (b-wave) ---
        I_on = 0.0
        for i in 1:p.n_on
            offset = sidx.on_bc[1] + (i - 1) * 4  # V is var 1 (0-indexed: +0)
            I_on += col.on_params.C_m * (u[offset] - u_prev[offset]) / dt
        end
        b_contrib = w.on_bc * I_on
        components[:b_wave][ti] = b_contrib

        # --- OFF-Bipolar (d-wave) ---
        I_off = 0.0
        for i in 1:p.n_off
            offset = sidx.off_bc[1] + (i - 1) * 4
            I_off += col.off_params.C_m * (u[offset] - u_prev[offset]) / dt
        end
        d_contrib = w.off_bc * I_off
        components[:d_wave][ti] = d_contrib

        # --- Amacrine OPs ---
        I_a2 = 0.0
        for i in 1:p.n_a2
            offset = sidx.a2[1] + (i - 1) * 3
            I_a2 += col.a2_params.C_m * (u[offset] - u_prev[offset]) / dt
        end
        I_gaba = 0.0
        for i in 1:p.n_gaba
            offset = sidx.gaba_ac[1] + (i - 1) * 3
            I_gaba += col.gaba_params.C_m * (u[offset] - u_prev[offset]) / dt
        end
        op_contrib = w.a2 * I_a2 + w.gaba * I_gaba
        components[:OPs][ti] = op_contrib

        # --- Müller glia (P3) ---
        I_muller = 0.0
        for i in 1:p.n_muller
            offset = sidx.muller[1] + (i - 1) * 4
            I_muller += col.muller_params.C_m * (u[offset] - u_prev[offset]) / dt
        end
        p3_contrib = w.muller * I_muller
        components[:P3][ti] = p3_contrib

        # --- RPE (c-wave) ---
        I_rpe = 0.0
        for i in 1:p.n_rpe
            offset = sidx.rpe[1] + (i - 1) * 2
            # RPE tau is very large, so dV/dt is small; use the actual RPE current instead
            rpe_u = view(u, offset:offset+1)
            I_rpe += rpe_transmembrane_current(rpe_u, col.rpe_params)
        end
        c_contrib = w.rpe * I_rpe
        components[:c_wave][ti] = c_contrib

        # --- Ganglion ---
        I_gc = 0.0
        for i in 1:p.n_gc
            offset = sidx.gc[1] + (i - 1) * 2
            I_gc += col.gc_params.C_m * (u[offset] - u_prev[offset]) / dt
        end
        gc_contrib = w.gc * I_gc
        components[:ganglion][ti] = gc_contrib

        # Total ERG
        erg[ti] = pr_contrib + b_contrib + d_contrib + op_contrib +
                  p3_contrib + c_contrib + gc_contrib
    end

    # Copy first sample from second to avoid initial spike
    if n_t >= 2
        erg[1] = erg[2]
        for (k, v) in components
            v[1] = v[2]
        end
    end

    return erg, components
end

"""
    extract_ops(erg, t; low=75.0, high=300.0)

Extract oscillatory potentials via bandpass filtering.
Returns (ops_filtered, t) for the full time range.
"""
function extract_ops(erg::Vector{Float64}, t::Vector{Float64};
                     low::Float64=75.0, high::Float64=300.0)
    dt = t[2] - t[1]  # ms
    fs = 1000.0 / dt   # Hz (convert from ms to seconds)

    # Bandpass filter 75-300 Hz, 4th order Butterworth
    responsetype = Bandpass(low, high; fs=fs)
    designmethod = Butterworth(4)
    bp = digitalfilter(responsetype, designmethod)

    # Apply zero-phase filter
    ops = filtfilt(bp, erg)

    return ops
end
