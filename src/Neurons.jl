# Core Morris-Lecar based neuron with E/I/M neurotransmitter systems
# Based on Tarchick et al. 2023 (Scientific Reports) starburst amacrine cell model
# Extended for full visual pathway digital twin

module Neurons

using Parameters
using StaticArrays
using LinearAlgebra

export AbstractCellType, NeuronState, NeuronParams
export BipolarCell, GanglionCell, ThalamicRelay, ThalamicInhibitory
export CorticalPyramidal, CorticalInhibitory, CorticalModulatory
export morris_lecar_derivatives!, gating_steady_state, sigmoidal_release
export logistic_growth, second_messenger_dynamics

#=============================================================================
    ABSTRACT TYPES AND CELL TYPE DEFINITIONS
=============================================================================#

abstract type AbstractCellType end

# Retinal cell types
struct BipolarCell <: AbstractCellType end
struct GanglionCell <: AbstractCellType end

# Thalamic (LGN) cell types  
struct ThalamicRelay <: AbstractCellType end
struct ThalamicInhibitory <: AbstractCellType end

# Cortical (V1) cell types
struct CorticalPyramidal <: AbstractCellType end
struct CorticalInhibitory <: AbstractCellType end
struct CorticalModulatory <: AbstractCellType end

#=============================================================================
    NEURON STATE STRUCTURE
    
    Each neuron has:
    - V: membrane voltage (mV)
    - N: K+ channel gating variable (dimensionless, 0-1)
    - M_Na: Na+ channel activation (dimensionless, 0-1)
    - H_Na: Na+ channel inactivation (dimensionless, 0-1)
    - Ca: intracellular calcium concentration (mM)
    - A: first-order logistic growth for modulatory cascade (dimensionless)
    - B: second messenger / modulatory channel activation (dimensionless)
    - E: excitatory neurotransmitter in synaptic cleft (μM)
    - I: inhibitory neurotransmitter in synaptic cleft (μM)
    - M_mod: modulatory neurotransmitter in synaptic cleft (μM)
=============================================================================#

@with_kw mutable struct NeuronState{T<:Real}
    V::T = -65.0          # Membrane voltage (mV)
    N::T = 0.0            # K+ channel gating
    M_Na::T = 0.0         # Na+ activation
    H_Na::T = 1.0         # Na+ inactivation
    Ca::T = 0.0001        # Calcium concentration (mM)
    A::T = 0.0            # Modulatory cascade - logistic growth
    B::T = 0.0            # Second messenger / iMod gating
    E::T = 0.0            # Excitatory NT (glutamate/ACh) in synapse
    I::T = 0.0            # Inhibitory NT (GABA) in synapse
    M_mod::T = 0.0        # Modulatory NT in synapse
    W::T = 0.0            # Noise state (Ornstein-Uhlenbeck)
end

# Convert to/from StaticArray for efficient computation
function state_to_vector(s::NeuronState)
    SVector(s.V, s.N, s.M_Na, s.H_Na, s.Ca, s.A, s.B, s.E, s.I, s.M_mod, s.W)
end

function vector_to_state!(s::NeuronState, v::AbstractVector)
    s.V, s.N, s.M_Na, s.H_Na, s.Ca, s.A, s.B, s.E, s.I, s.M_mod, s.W = v
    return s
end

#=============================================================================
    NEURON PARAMETERS
    
    Based on Tarchick et al. 2023, extended with:
    - Modulatory neurotransmitter system
    - Tunable E_MOD reversal potential for modulatory ion channel
    - Cell-type specific parameter sets
=============================================================================#

@with_kw struct NeuronParams{T<:Real}
    # Membrane properties
    C_m::T = 13.6         # Membrane capacitance (pF)
    
    # Leak channel
    g_leak::T = 2.0       # Leak conductance (nS)
    E_leak::T = -70.0     # Leak reversal (mV)
    
    # Calcium channels (L-type / T-type)
    g_Ca::T = 4.0         # Ca conductance (nS)
    E_Ca::T = 120.0       # Ca reversal (mV)
    V1::T = -1.2          # Ca activation half-max (mV)
    V2::T = 18.0          # Ca activation slope (mV)
    
    # Voltage-gated K+ channels
    g_K::T = 8.0          # K conductance (nS)
    E_K::T = -90.0        # K reversal (mV)
    V3::T = 12.0          # K activation half-max (mV)
    V4::T = 17.4          # K activation slope (mV)
    τ_N::T = 5.0          # K gating time constant (ms)
    
    # Voltage-gated Na+ channels
    g_Na::T = 20.0        # Na conductance (nS)
    E_Na::T = 50.0        # Na reversal (mV)
    V7::T = 0.1           # Na activation parameters
    V8::T = -40.0
    V9::T = 10.0
    V10::T = 4.0
    V11::T = -65.0
    V12::T = 18.0
    V13::T = 0.07
    V14::T = -65.0
    V15::T = 20.0
    V16::T = 1.0
    V17::T = -35.0
    V18::T = 10.0
    
    # TREK1 / Modulatory ion channel (sAHP mechanism)
    g_TREK::T = 0.5       # TREK1 conductance (nS) - uses E_K
    g_MOD::T = 0.3        # Modulatory channel conductance (nS)
    E_MOD::T = -70.0      # Modulatory channel reversal - TUNABLE
    
    # Calcium dynamics
    Ca_0::T = 0.0001      # Baseline calcium (mM)
    δ::T = 0.01           # Ca influx rate (mM/mV)
    λ::T = 0.01           # Ca decay rate (1/ms)
    τ_Ca::T = 500.0       # Ca time constant (ms)
    
    # Modulatory cascade (At → Bt → iMod)
    # At: first-order logistic growth activated by modulatory NT
    α_A::T = 625.0        # At activation rate
    τ_A::T = 8300.0       # At time constant (ms)
    
    # Bt: second messenger dynamics
    β_B::T = 34.0         # Bt activation rate  
    τ_B::T = 10000.0      # Bt time constant (ms)
    
    # Excitatory NT (glutamate/ACh) parameters
    g_E::T = 0.215        # Excitatory synaptic conductance (μS)
    E_E::T = 0.0          # Excitatory reversal (mV) - AMPA/nAChR
    ρ_E::T = 6.0          # Release rate (μM/mV)
    κ_E::T = 1.0          # Half-saturation for receptor activation (μM)
    τ_E::T = 10.0         # NT decay time constant (ms)
    V_sE::T = 0.1         # Release sigmoid slope
    V_0E::T = -40.0       # Release sigmoid midpoint (mV)
    
    # Inhibitory NT (GABA) parameters
    g_I::T = 0.9          # Inhibitory synaptic conductance (μS)
    E_Cl::T = -65.0       # Chloride reversal (mV) - can be depolarizing early dev
    ρ_I::T = 5.0          # Release rate (μM/mV)
    κ_I::T = 1.0          # Half-saturation (μM)
    τ_I::T = 15.0         # NT decay time constant (ms)
    V_sI::T = 0.1         # Release sigmoid slope
    V_0I::T = -40.0       # Release sigmoid midpoint (mV)
    
    # Modulatory NT parameters (e.g., dopamine, ACh muscarinic, neuropeptides)
    g_M::T = 0.1          # Modulatory synaptic conductance (μS)
    ρ_M::T = 3.0          # Release rate (μM/mV)
    κ_M::T = 0.5          # Half-saturation (μM)
    τ_M::T = 100.0        # Slow decay for modulatory NT (ms)
    V_sM::T = 0.1         # Release sigmoid slope
    V_0M::T = -30.0       # Release sigmoid midpoint (mV)
    
    # Noise parameters (Ornstein-Uhlenbeck)
    σ_noise::T = 0.1      # Noise amplitude (pA)
    τ_W::T = 800.0        # Noise correlation time (ms)
    
    # Applied current
    I_app::T = 0.0        # External applied current (pA)
end

#=============================================================================
    GATING FUNCTIONS
=============================================================================#

"""
    gating_steady_state(V, V_half, slope)

Boltzmann steady-state gating function
Returns fraction of channels open (0-1)
"""
function gating_steady_state(V::T, V_half::T, slope::T) where T<:Real
    return 0.5 * (1.0 + tanh((V - V_half) / slope))
end

"""
    gating_rate(V, V_half, slope)

Voltage-dependent gating rate for K+ channels
"""
function gating_rate(V::T, V_half::T, slope::T) where T<:Real
    return cosh((V - V_half) / (2.0 * slope))
end

"""
    sigmoidal_release(V, slope, midpoint)

Sigmoidal neurotransmitter release function
"""
function sigmoidal_release(V::T, slope::T, midpoint::T) where T<:Real
    return 1.0 / (1.0 + exp(-slope * (V - midpoint)))
end

"""
    receptor_activation(NT, κ)

Hill-type receptor activation (fractional occupancy)
"""
function receptor_activation(NT::T, κ::T) where T<:Real
    return NT^2 / (NT^2 + κ^2)
end

"""
    logistic_growth(x, rate, capacity)

First-order logistic growth equation for modulatory cascade
dx/dt = rate * x * (1 - x/capacity)
For normalized x (capacity=1): dx/dt = rate * x * (1 - x)
"""
function logistic_growth(x::T, rate::T) where T<:Real
    return rate * x * (1.0 - x)
end

"""
    second_messenger_dynamics(A, B, β, τ)

Second messenger (Bt) activation driven by At
Based on TREK1 dephosphorylation cascade
"""
function second_messenger_dynamics(A::T, B::T, β::T, τ::T) where T<:Real
    return (β * A^4 * (1.0 - B) - B) / τ
end

#=============================================================================
    IONIC CURRENTS
=============================================================================#

"""
    I_leak(V, p)

Leak current
"""
function I_leak(V::T, p::NeuronParams{T}) where T<:Real
    return -p.g_leak * (V - p.E_leak)
end

"""
    I_Ca(V, p)

Calcium current (L-type/T-type)
"""
function I_Ca(V::T, p::NeuronParams{T}) where T<:Real
    M_inf = gating_steady_state(V, p.V1, p.V2)
    return -p.g_Ca * M_inf * (V - p.E_Ca)
end

"""
    I_K(V, N, p)

Voltage-gated potassium current
"""
function I_K(V::T, N::T, p::NeuronParams{T}) where T<:Real
    return -p.g_K * N * (V - p.E_K)
end

"""
    I_Na(V, M, H, p)

Voltage-gated sodium current (Hodgkin-Huxley style)
"""
function I_Na(V::T, M::T, H::T, p::NeuronParams{T}) where T<:Real
    return -p.g_Na * M^3 * H * (V - p.E_Na)
end

"""
    I_TREK(V, B, p)

TREK1 potassium current (sAHP mechanism)
Activated by second messenger cascade
"""
function I_TREK(V::T, B::T, p::NeuronParams{T}) where T<:Real
    return -p.g_TREK * B * (V - p.E_K)
end

"""
    I_MOD(V, B, p)

Modulatory ion channel current
Reversal potential E_MOD is tunable
Can be excitatory, inhibitory, or shunting depending on E_MOD
"""
function I_MOD(V::T, B::T, p::NeuronParams{T}) where T<:Real
    return -p.g_MOD * B * (V - p.E_MOD)
end

"""
    I_exc(V, E, p)

Excitatory synaptic current (AMPA/nAChR)
"""
function I_exc(V::T, E::T, p::NeuronParams{T}) where T<:Real
    H_E = receptor_activation(E, p.κ_E)
    return -p.g_E * H_E * (V - p.E_E)
end

"""
    I_inh(V, I_nt, p)

Inhibitory synaptic current (GABA-A)
"""
function I_inh(V::T, I_nt::T, p::NeuronParams{T}) where T<:Real
    H_I = receptor_activation(I_nt, p.κ_I)
    return -p.g_I * H_I * (V - p.E_Cl)
end

"""
    I_mod_synaptic(V, M_nt, A, p)

Modulatory synaptic effect - triggers At cascade
Rather than direct current, modulates At dynamics
Returns contribution to dA/dt
"""
function modulatory_drive(M_nt::T, p::NeuronParams{T}) where T<:Real
    H_M = receptor_activation(M_nt, p.κ_M)
    return p.α_A * H_M
end

#=============================================================================
    SODIUM CHANNEL GATING (Hodgkin-Huxley)
=============================================================================#

function alpha_m(V::T, p::NeuronParams{T}) where T<:Real
    denom = p.V7 * (exp(-(V - p.V8) / p.V9) - 1.0)
    if abs(denom) < 1e-7
        return 1.0  # L'Hopital limit
    end
    return -(V - p.V8) / denom
end

function beta_m(V::T, p::NeuronParams{T}) where T<:Real
    return p.V10 * exp(-(V - p.V11) / p.V12)
end

function alpha_h(V::T, p::NeuronParams{T}) where T<:Real
    return p.V13 * exp(-(V - p.V14) / p.V15)
end

function beta_h(V::T, p::NeuronParams{T}) where T<:Real
    return 1.0 / (p.V16 * exp(-(V - p.V17) / p.V18) + 1.0)
end

#=============================================================================
    MAIN DERIVATIVE FUNCTION
=============================================================================#

"""
    morris_lecar_derivatives!(du, u, p, t; E_pre=0, I_pre=0, M_pre=0, noise=0)

Compute derivatives for Morris-Lecar neuron with E/I/M system

Arguments:
- du: derivative vector (output)
- u: state vector [V, N, M_Na, H_Na, Ca, A, B, E, I, M_mod, W]
- p: NeuronParams
- t: time
- E_pre: presynaptic excitatory NT received from neighbors
- I_pre: presynaptic inhibitory NT received from neighbors  
- M_pre: presynaptic modulatory NT received from neighbors
- noise: stochastic noise term
"""
function morris_lecar_derivatives!(
    du::AbstractVector{T},
    u::AbstractVector{T},
    p::NeuronParams{T},
    t::T;
    E_pre::T = zero(T),
    I_pre::T = zero(T),
    M_pre::T = zero(T),
    noise::T = zero(T)
) where T<:Real
    
    # Unpack state
    V, N, M_Na, H_Na, Ca, A, B, E, I_nt, M_mod, W = u
    
    # Compute all ionic currents
    I_total = I_leak(V, p) +
              I_Ca(V, p) +
              I_K(V, N, p) +
              I_Na(V, M_Na, H_Na, p) +
              I_TREK(V, B, p) +
              I_MOD(V, B, p) +
              I_exc(V, E + E_pre, p) +       # Own + received excitatory
              I_inh(V, I_nt + I_pre, p) +     # Own + received inhibitory
              p.σ_noise * W +                 # Noise current
              p.I_app                         # Applied current
    
    # dV/dt - Voltage
    du[1] = I_total / p.C_m
    
    # dN/dt - K+ gating
    N_inf = gating_steady_state(V, p.V3, p.V4)
    Λ = gating_rate(V, p.V3, p.V4)
    du[2] = Λ * (N_inf - N) / p.τ_N
    
    # dM_Na/dt - Na+ activation
    αm = alpha_m(V, p)
    βm = beta_m(V, p)
    du[3] = αm * (1.0 - M_Na) - βm * M_Na
    
    # dH_Na/dt - Na+ inactivation
    αh = alpha_h(V, p)
    βh = beta_h(V, p)
    du[4] = αh * (1.0 - H_Na) - βh * H_Na
    
    # dCa/dt - Calcium dynamics
    I_Ca_val = I_Ca(V, p)
    du[5] = (p.Ca_0 + p.δ * abs(I_Ca_val) - p.λ * Ca) / p.τ_Ca
    
    # dA/dt - Modulatory cascade (logistic growth)
    # Driven by both Ca (intrinsic) and modulatory NT (synaptic)
    mod_drive = modulatory_drive(M_mod + M_pre, p)
    Ca_drive = p.α_A * Ca^4
    total_drive = Ca_drive + mod_drive
    du[6] = (total_drive * (1.0 - A) - A) / p.τ_A
    
    # dB/dt - Second messenger / iMod gating
    du[7] = (p.β_B * A^4 * (1.0 - B) - B) / p.τ_B
    
    # dE/dt - Excitatory NT release
    release_E = p.ρ_E * sigmoidal_release(V, p.V_sE, p.V_0E)
    du[8] = (release_E - E) / p.τ_E
    
    # dI/dt - Inhibitory NT release
    release_I = p.ρ_I * sigmoidal_release(V, p.V_sI, p.V_0I)
    du[9] = (release_I - I_nt) / p.τ_I
    
    # dM_mod/dt - Modulatory NT release (slower dynamics)
    release_M = p.ρ_M * sigmoidal_release(V, p.V_sM, p.V_0M)
    du[10] = (release_M - M_mod) / p.τ_M
    
    # dW/dt - Ornstein-Uhlenbeck noise
    du[11] = (-W + noise) / p.τ_W
    
    return nothing
end

end # module Neurons
