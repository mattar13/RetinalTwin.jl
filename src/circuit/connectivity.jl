# ============================================================
# connectivity.jl — Retinal column connection matrix
# Spec §4.1 — 19 synaptic connections
# ============================================================

"""
    ConnectionDef

Defines one synaptic connection between cell populations.
"""
struct ConnectionDef
    pre::Symbol         # presynaptic cell type
    post::Symbol        # postsynaptic cell type
    nt_type::Symbol     # :E, :I, or :M
    receptor::Symbol    # :iGluR, :mGluR6, :GlyR, :GABA_A, :D1R, :feedback
    g_max::Float64      # nS
    E_rev::Float64      # mV (NaN for modulatory)
    tau_s::Float64      # ms
end

"""
    default_connections()

Return the full retinal column connection table from spec §4.1.
"""
function default_connections()
    return ConnectionDef[
        # Photoreceptors → Horizontal cells
        ConnectionDef(:rod,     :hc,      :E, :iGluR,    5.0,    0.0,   3.0),
        ConnectionDef(:cone,    :hc,      :E, :iGluR,    5.0,    0.0,   3.0),
        # Photoreceptors → ON-Bipolar (sign-inverting, handled separately)
        ConnectionDef(:rod,     :on_bc,   :E, :mGluR6,   NaN,    NaN,  30.0),
        ConnectionDef(:cone,    :on_bc,   :E, :mGluR6,   NaN,    NaN,  30.0),
        # Photoreceptors → OFF-Bipolar (ionotropic)
        ConnectionDef(:rod,     :off_bc,  :E, :iGluR,    4.0,    0.0,   3.0),
        ConnectionDef(:cone,    :off_bc,  :E, :iGluR,    4.0,    0.0,   3.0),
        # HC → Photoreceptors (modulatory feedback)
        ConnectionDef(:hc,      :rod,     :M, :feedback,  1.0,   NaN,  20.0),
        ConnectionDef(:hc,      :cone,    :M, :feedback,  1.0,   NaN,  20.0),
        # ON-Bipolar → downstream
        ConnectionDef(:on_bc,   :a2,      :E, :iGluR,    8.0,    0.0,   2.0),
        ConnectionDef(:on_bc,   :gaba_ac, :E, :iGluR,    6.0,    0.0,   2.0),
        ConnectionDef(:on_bc,   :da_ac,   :E, :iGluR,    3.0,    0.0,   5.0),
        ConnectionDef(:on_bc,   :gc,      :E, :iGluR,    5.0,    0.0,   3.0),
        # OFF-Bipolar → Ganglion
        ConnectionDef(:off_bc,  :gc,      :E, :iGluR,    5.0,    0.0,   3.0),
        # A2 Amacrine (glycinergic inhibition)
        ConnectionDef(:a2,      :gaba_ac, :I, :GlyR,    10.0,  -80.0,   4.0),
        ConnectionDef(:a2,      :off_bc,  :I, :GlyR,     5.0,  -80.0,   4.0),
        ConnectionDef(:a2,      :gc,      :I, :GlyR,     3.0,  -80.0,   4.0),
        # GABAergic Amacrine
        ConnectionDef(:gaba_ac, :a2,      :I, :GABA_A,  10.0,  -70.0,   8.0),
        ConnectionDef(:gaba_ac, :on_bc,   :I, :GABA_A,   3.0,  -70.0,   8.0),
        ConnectionDef(:gaba_ac, :gc,      :I, :GABA_A,   3.0,  -70.0,   8.0),
        # Dopaminergic modulation
        ConnectionDef(:da_ac,   :a2,      :M, :D1R,      1.0,   NaN, 200.0),
        ConnectionDef(:da_ac,   :gaba_ac, :M, :D1R,      1.0,   NaN, 200.0),
        ConnectionDef(:da_ac,   :hc,      :M, :D1R,      0.5,   NaN, 200.0),
    ]
end

"""
    get_connections(connections, post_type, nt_type)

Filter connections by postsynaptic target and NT type.
"""
function get_connections(connections::Vector{ConnectionDef}, post::Symbol, nt::Symbol)
    return filter(c -> c.post == post && c.nt_type == nt, connections)
end
