# ============================================================
# run.jl — Simulation driver
# Spec §7.6
# ============================================================

using DifferentialEquations
using Setfield

"""
    simulate_flash(; intensity, duration, t_total, dt_save, regime, kwargs...)

Main simulation entry point. Returns a NamedTuple with:
- `t`: time vector (ms)
- `erg`: ERG trace
- `erg_components`: Dict of per-cell-type ERG contributions
- `cell_voltages`: Dict of voltage traces by cell type
- `solution`: raw ODE solution
- `col`: the RetinalColumn used
- `sidx`: the StateIndex used
"""
function simulate_flash(; intensity::Float64=1000.0, duration::Float64=10.0,
                         t_total::Float64=5000.0, dt_save::Float64=0.1,
                         regime::Symbol=:scotopic, kwargs...)

    col = build_retinal_column(; regime=regime, kwargs...)
    col = @set col.stimulus.I_0 = intensity
    col = @set col.stimulus.t_dur = duration

    sidx = StateIndex(col)
    conns = default_connections()

    # Initial conditions (dark-adapted)
    u0 = dark_adapted_state(col, sidx)

    # ODE problem
    tspan = (0.0, t_total)
    p = (col, sidx, conns)
    prob = ODEProblem(retinal_column_rhs!, u0, tspan, p)

    # Solve with auto-stiffness detection (handles fast OPs + slow RPE)
    sol = solve(prob, AutoTsit5(Rosenbrock23());
                saveat=dt_save,
                abstol=1e-8, reltol=1e-6,
                maxiters=1_000_000)

    # Compute ERG
    erg, components = compute_erg(sol, col, sidx)

    # Extract voltages
    voltages = extract_voltages(sol, col, sidx)

    return (t=sol.t, erg=erg, erg_components=components,
            cell_voltages=voltages, solution=sol, col=col, sidx=sidx)
end

"""
    extract_voltages(sol, col, sidx)

Pull voltage traces from ODE solution for each cell type.
Returns Dict{Symbol, Matrix{Float64}} where each value is (n_timepoints × n_cells).
"""
function extract_voltages(sol, col::RetinalColumn, sidx::StateIndex)
    n_t = length(sol.t)
    p = col.pop
    voltages = Dict{Symbol, Matrix{Float64}}()

    # Helper: extract voltage for populations with given vars_per_cell and V at offset v_idx
    function extract_pop!(name, range, vars_per_cell, v_idx, n_cells)
        if n_cells == 0
            voltages[name] = zeros(n_t, 0)
            return
        end
        V = zeros(n_t, n_cells)
        for ti in 1:n_t
            u = sol.u[ti]
            for ci in 1:n_cells
                offset = range[1] + (ci - 1) * vars_per_cell + (v_idx - 1)
                V[ti, ci] = u[offset]
            end
        end
        voltages[name] = V
    end

    extract_pop!(:rod,     sidx.rod,     ROD_STATE_VARS, ROD_V_INDEX, p.n_rod)
    extract_pop!(:cone,    sidx.cone,    CONE_STATE_VARS, CONE_V_INDEX, p.n_cone)
    extract_pop!(:hc,      sidx.hc,      3, 1, p.n_hc)     # V is var 1
    extract_pop!(:on_bc,   sidx.on_bc,   4, 1, p.n_on)
    extract_pop!(:off_bc,  sidx.off_bc,  4, 1, p.n_off)
    extract_pop!(:a2,      sidx.a2,      3, 1, p.n_a2)
    extract_pop!(:gaba_ac, sidx.gaba_ac, 3, 1, p.n_gaba)
    extract_pop!(:da_ac,   sidx.da_ac,   3, 1, p.n_dopa)
    extract_pop!(:gc,      sidx.gc,      2, 1, p.n_gc)
    extract_pop!(:muller,  sidx.muller,  4, 1, p.n_muller)
    extract_pop!(:rpe,     sidx.rpe,     2, 1, p.n_rpe)

    return voltages
end

"""
    extract_neurotransmitters(sol, col, sidx)

Pull neurotransmitter concentration traces from ODE solution.
Returns Dict{Symbol, Matrix{Float64}}.
"""
function extract_neurotransmitters(sol, col::RetinalColumn, sidx::StateIndex)
    n_t = length(sol.t)
    p = col.pop
    nts = Dict{Symbol, Matrix{Float64}}()

    function extract_nt!(name, range, vars_per_cell, nt_idx, n_cells)
        if n_cells == 0
            nts[name] = zeros(n_t, 0)
            return
        end
        NT = zeros(n_t, n_cells)
        for ti in 1:n_t
            u = sol.u[ti]
            for ci in 1:n_cells
                offset = range[1] + (ci - 1) * vars_per_cell + (nt_idx - 1)
                NT[ti, ci] = u[offset]
            end
        end
        nts[name] = NT
    end

    extract_nt!(:glu_rod,  sidx.rod,     ROD_STATE_VARS, ROD_GLU_INDEX, p.n_rod)
    extract_nt!(:glu_cone, sidx.cone,    CONE_STATE_VARS, CONE_GLU_INDEX, p.n_cone)
    extract_nt!(:glu_on,   sidx.on_bc,   4, 4, p.n_on)     # Glu is var 4
    extract_nt!(:glu_off,  sidx.off_bc,  4, 4, p.n_off)
    extract_nt!(:gly_a2,   sidx.a2,      3, 3, p.n_a2)     # Gly is var 3
    extract_nt!(:gaba,     sidx.gaba_ac, 3, 3, p.n_gaba)   # GABA is var 3
    extract_nt!(:da,       sidx.da_ac,   3, 3, p.n_dopa)   # DA is var 3

    return nts
end
