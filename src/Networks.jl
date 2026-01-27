# Network connectivity module
# Implements discrete synaptic connections (no PDEs - NT shared between neighbors only)

module Networks

using ..Neurons
using ..CellTypes
using Parameters
using LinearAlgebra
using Random
using SparseArrays

export NetworkLayer, VisualPathway, Synapse
export create_layer, connect_layers!, get_synaptic_input
export create_visual_pathway, step_network!

#=============================================================================
    SYNAPSE STRUCTURE
    
    Represents a connection between two neurons
    - NT shared directly (no diffusion PDEs)
    - Only immediate neighbors share NT values
=============================================================================#

@with_kw struct Synapse{T<:Real}
    # Indices
    pre_layer::Int                    # Index of presynaptic layer
    pre_idx::Int                      # Index of presynaptic neuron
    post_layer::Int                   # Index of postsynaptic layer
    post_idx::Int                     # Index of postsynaptic neuron
    
    # Synaptic properties
    weight::T = 1.0                   # Synaptic weight (multiplicative)
    delay::T = 1.0                    # Synaptic delay (ms)
    
    # Type of transmission
    excitatory::Bool = true           # Is this E, I, or M?
    inhibitory::Bool = false
    modulatory::Bool = false
end

#=============================================================================
    NETWORK LAYER
    
    A population of neurons of the same type arranged in 2D
    - Neurons are arranged in a grid
    - Local connections within layer (recurrent)
    - Feedforward/feedback between layers
=============================================================================#

@with_kw mutable struct NetworkLayer{T<:Real}
    # Layer identity
    name::String
    cell_type::Type{<:AbstractCellType}
    
    # Grid dimensions
    n_rows::Int
    n_cols::Int
    n_neurons::Int
    
    # State vectors (n_neurons × n_state_vars)
    # [V, N, M_Na, H_Na, Ca, A, B, E, I, M_mod, W]
    states::Matrix{T}
    
    # Parameter sets (can be heterogeneous)
    params::Vector{NeuronParams{T}}
    
    # Local connectivity radius (for within-layer connections)
    local_radius::Int = 1             # Direct neighbors only
    
    # Neurotransmitter pools received from presynaptic partners
    E_received::Vector{T}             # Excitatory NT from all inputs
    I_received::Vector{T}             # Inhibitory NT from all inputs
    M_received::Vector{T}             # Modulatory NT from all inputs
    
    # Output neurotransmitters (for downstream)
    E_output::Vector{T}
    I_output::Vector{T}
    M_output::Vector{T}
end

"""
    create_layer(name, cell_type, n_rows, n_cols; kwargs...)

Create a network layer with neurons of specified type
"""
function create_layer(
    name::String,
    cell_type::Type{<:AbstractCellType},
    n_rows::Int,
    n_cols::Int;
    heterogeneous::Bool = false,
    param_noise::Float64 = 0.1,
    T::Type = Float64
)
    n_neurons = n_rows * n_cols
    n_state_vars = 11  # V, N, M_Na, H_Na, Ca, A, B, E, I, M_mod, W
    
    # Initialize states
    states = zeros(T, n_neurons, n_state_vars)
    base_params = get_cell_params(cell_type)
    
    # Set initial conditions
    for i in 1:n_neurons
        states[i, 1] = base_params.E_leak + randn() * 2.0    # V
        states[i, 2] = 0.0                                     # N
        states[i, 3] = 0.0                                     # M_Na
        states[i, 4] = 1.0                                     # H_Na
        states[i, 5] = base_params.Ca_0                        # Ca
        states[i, 6] = 0.0                                     # A
        states[i, 7] = 0.0                                     # B
        states[i, 8] = 0.0                                     # E
        states[i, 9] = 0.0                                     # I
        states[i, 10] = 0.0                                    # M_mod
        states[i, 11] = 0.0                                    # W
    end
    
    # Create parameter sets
    if heterogeneous
        # Add noise to parameters for heterogeneity
        params = [add_param_noise(base_params, param_noise) for _ in 1:n_neurons]
    else
        params = [base_params for _ in 1:n_neurons]
    end
    
    # Initialize NT pools
    E_received = zeros(T, n_neurons)
    I_received = zeros(T, n_neurons)
    M_received = zeros(T, n_neurons)
    E_output = zeros(T, n_neurons)
    I_output = zeros(T, n_neurons)
    M_output = zeros(T, n_neurons)
    
    return NetworkLayer(
        name = name,
        cell_type = cell_type,
        n_rows = n_rows,
        n_cols = n_cols,
        n_neurons = n_neurons,
        states = states,
        params = params,
        E_received = E_received,
        I_received = I_received,
        M_received = M_received,
        E_output = E_output,
        I_output = I_output,
        M_output = M_output
    )
end

"""
    add_param_noise(params, noise_frac)

Add random noise to parameters for heterogeneity
"""
function add_param_noise(p::NeuronParams{T}, noise_frac::Float64) where T
    # Create new params with noisy values
    # Only vary certain parameters (conductances, time constants)
    NeuronParams{T}(
        C_m = p.C_m * (1.0 + noise_frac * randn()),
        g_leak = p.g_leak * (1.0 + noise_frac * randn()),
        E_leak = p.E_leak,
        g_Ca = p.g_Ca * (1.0 + noise_frac * randn()),
        E_Ca = p.E_Ca,
        V1 = p.V1,
        V2 = p.V2,
        g_K = p.g_K * (1.0 + noise_frac * randn()),
        E_K = p.E_K,
        V3 = p.V3,
        V4 = p.V4,
        τ_N = p.τ_N * (1.0 + noise_frac * randn()),
        g_Na = p.g_Na * (1.0 + noise_frac * randn()),
        E_Na = p.E_Na,
        V7 = p.V7, V8 = p.V8, V9 = p.V9,
        V10 = p.V10, V11 = p.V11, V12 = p.V12,
        V13 = p.V13, V14 = p.V14, V15 = p.V15,
        V16 = p.V16, V17 = p.V17, V18 = p.V18,
        g_TREK = p.g_TREK * (1.0 + noise_frac * randn()),
        g_MOD = p.g_MOD * (1.0 + noise_frac * randn()),
        E_MOD = p.E_MOD,
        Ca_0 = p.Ca_0,
        δ = p.δ,
        λ = p.λ,
        τ_Ca = p.τ_Ca,
        α_A = p.α_A,
        τ_A = p.τ_A * (1.0 + noise_frac * randn()),
        β_B = p.β_B,
        τ_B = p.τ_B * (1.0 + noise_frac * randn()),
        g_E = p.g_E * (1.0 + noise_frac * randn()),
        E_E = p.E_E,
        ρ_E = p.ρ_E,
        κ_E = p.κ_E,
        τ_E = p.τ_E,
        V_sE = p.V_sE,
        V_0E = p.V_0E,
        g_I = p.g_I * (1.0 + noise_frac * randn()),
        E_Cl = p.E_Cl,
        ρ_I = p.ρ_I,
        κ_I = p.κ_I,
        τ_I = p.τ_I,
        V_sI = p.V_sI,
        V_0I = p.V_0I,
        g_M = p.g_M * (1.0 + noise_frac * randn()),
        ρ_M = p.ρ_M,
        κ_M = p.κ_M,
        τ_M = p.τ_M,
        V_sM = p.V_sM,
        V_0M = p.V_0M,
        σ_noise = p.σ_noise,
        τ_W = p.τ_W,
        I_app = p.I_app
    )
end

#=============================================================================
    VISUAL PATHWAY NETWORK
    
    Complete pathway from retina to cortex:
    - Retina: Bipolar → Ganglion
    - LGN: Relay ↔ Inhibitory (with cortical feedback)
    - V1: Pyramidal ↔ Inhibitory ↔ Modulatory
=============================================================================#

@with_kw mutable struct VisualPathway{T<:Real}
    # Layers by region
    layers::Vector{NetworkLayer{T}}
    layer_names::Dict{String, Int}    # Name → index mapping
    
    # Inter-layer connectivity (sparse matrices for efficiency)
    # Each element is a sparse matrix: connectivity[pre_layer][post_layer]
    # Matrix entries are synaptic weights
    E_connectivity::Vector{Vector{SparseMatrixCSC{T, Int}}}
    I_connectivity::Vector{Vector{SparseMatrixCSC{T, Int}}}
    M_connectivity::Vector{Vector{SparseMatrixCSC{T, Int}}}
    
    # Simulation parameters
    dt::T = 0.1                       # Time step (ms)
    t::T = 0.0                        # Current time
end

"""
    create_visual_pathway(; kwargs...)

Create complete visual pathway network with default structure
"""
function create_visual_pathway(;
    retina_size::Tuple{Int,Int} = (16, 16),
    lgn_size::Tuple{Int,Int} = (12, 12),
    cortex_size::Tuple{Int,Int} = (24, 24),
    heterogeneous::Bool = true,
    T::Type = Float64
)
    layers = NetworkLayer{T}[]
    layer_names = Dict{String, Int}()
    
    # Create retinal layers
    push!(layers, create_layer("Bipolar", BipolarCell, retina_size...; 
                               heterogeneous=heterogeneous, T=T))
    layer_names["Bipolar"] = length(layers)
    
    push!(layers, create_layer("Ganglion", GanglionCell, retina_size...; 
                               heterogeneous=heterogeneous, T=T))
    layer_names["Ganglion"] = length(layers)
    
    # Create LGN layers
    push!(layers, create_layer("ThalamicRelay", ThalamicRelay, lgn_size...; 
                               heterogeneous=heterogeneous, T=T))
    layer_names["ThalamicRelay"] = length(layers)
    
    push!(layers, create_layer("ThalamicInhibitory", ThalamicInhibitory, lgn_size...; 
                               heterogeneous=heterogeneous, T=T))
    layer_names["ThalamicInhibitory"] = length(layers)
    
    # Create cortical layers
    push!(layers, create_layer("CorticalPyramidal", CorticalPyramidal, cortex_size...; 
                               heterogeneous=heterogeneous, T=T))
    layer_names["CorticalPyramidal"] = length(layers)
    
    push!(layers, create_layer("CorticalInhibitory", CorticalInhibitory, cortex_size...; 
                               heterogeneous=heterogeneous, T=T))
    layer_names["CorticalInhibitory"] = length(layers)
    
    push!(layers, create_layer("CorticalModulatory", CorticalModulatory, cortex_size...; 
                               heterogeneous=heterogeneous, T=T))
    layer_names["CorticalModulatory"] = length(layers)
    
    n_layers = length(layers)
    
    # Initialize connectivity matrices
    E_conn = [[spzeros(T, layers[j].n_neurons, layers[i].n_neurons) 
               for j in 1:n_layers] for i in 1:n_layers]
    I_conn = [[spzeros(T, layers[j].n_neurons, layers[i].n_neurons) 
               for j in 1:n_layers] for i in 1:n_layers]
    M_conn = [[spzeros(T, layers[j].n_neurons, layers[i].n_neurons) 
               for j in 1:n_layers] for i in 1:n_layers]
    
    pathway = VisualPathway{T}(
        layers = layers,
        layer_names = layer_names,
        E_connectivity = E_conn,
        I_connectivity = I_conn,
        M_connectivity = M_conn
    )
    
    # Set up default connectivity
    setup_default_connectivity!(pathway)
    
    return pathway
end

"""
    setup_default_connectivity!(pathway)

Set up biologically plausible connectivity for visual pathway
"""
function setup_default_connectivity!(pathway::VisualPathway{T}) where T
    ln = pathway.layer_names
    
    # Retina: Bipolar → Ganglion (excitatory, topographic)
    connect_topographic!(pathway, "Bipolar", "Ganglion", :E, 
                        radius=1, weight=1.0)
    
    # Retina → LGN: Ganglion → Relay (excitatory, convergent)
    connect_convergent!(pathway, "Ganglion", "ThalamicRelay", :E,
                       convergence=2, weight=0.8)
    
    # Ganglion → Thalamic Inhibitory (feedforward inhibition)
    connect_convergent!(pathway, "Ganglion", "ThalamicInhibitory", :E,
                       convergence=2, weight=0.5)
    
    # LGN local: Relay ↔ Inhibitory
    connect_topographic!(pathway, "ThalamicRelay", "ThalamicInhibitory", :E,
                        radius=1, weight=0.6)
    connect_topographic!(pathway, "ThalamicInhibitory", "ThalamicRelay", :I,
                        radius=1, weight=0.8)
    
    # LGN → V1: Relay → Pyramidal (excitatory, divergent)
    connect_divergent!(pathway, "ThalamicRelay", "CorticalPyramidal", :E,
                      divergence=2, weight=0.7)
    
    # Relay → Inhibitory (feedforward)
    connect_divergent!(pathway, "ThalamicRelay", "CorticalInhibitory", :E,
                      divergence=2, weight=0.5)
    
    # V1 local: Pyramidal ↔ Inhibitory ↔ Modulatory
    connect_topographic!(pathway, "CorticalPyramidal", "CorticalInhibitory", :E,
                        radius=2, weight=0.6)
    connect_topographic!(pathway, "CorticalInhibitory", "CorticalPyramidal", :I,
                        radius=2, weight=0.9)
    
    connect_topographic!(pathway, "CorticalPyramidal", "CorticalModulatory", :E,
                        radius=2, weight=0.4)
    connect_topographic!(pathway, "CorticalModulatory", "CorticalPyramidal", :M,
                        radius=3, weight=0.5)
    connect_topographic!(pathway, "CorticalModulatory", "CorticalInhibitory", :I,
                        radius=2, weight=0.6)  # Disinhibition
    
    # Recurrent pyramidal connections
    connect_topographic!(pathway, "CorticalPyramidal", "CorticalPyramidal", :E,
                        radius=2, weight=0.3)
    
    # Corticothalamic feedback: Pyramidal → Relay, Inhibitory
    connect_convergent!(pathway, "CorticalPyramidal", "ThalamicRelay", :E,
                       convergence=2, weight=0.4)
    connect_convergent!(pathway, "CorticalPyramidal", "ThalamicInhibitory", :E,
                       convergence=2, weight=0.3)
end

"""
    connect_topographic!(pathway, pre_name, post_name, nt_type; kwargs)

Create topographic (retinotopic) connections preserving spatial organization
"""
function connect_topographic!(
    pathway::VisualPathway{T},
    pre_name::String,
    post_name::String,
    nt_type::Symbol;
    radius::Int = 1,
    weight::T = 1.0
) where T
    pre_idx = pathway.layer_names[pre_name]
    post_idx = pathway.layer_names[post_name]
    pre_layer = pathway.layers[pre_idx]
    post_layer = pathway.layers[post_idx]
    
    # Select connectivity matrix
    conn_matrix = if nt_type == :E
        pathway.E_connectivity[pre_idx][post_idx]
    elseif nt_type == :I
        pathway.I_connectivity[pre_idx][post_idx]
    else
        pathway.M_connectivity[pre_idx][post_idx]
    end
    
    # Compute spatial mapping
    row_ratio = post_layer.n_rows / pre_layer.n_rows
    col_ratio = post_layer.n_cols / pre_layer.n_cols
    
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]
    
    for pre_i in 1:pre_layer.n_neurons
        pre_row = (pre_i - 1) ÷ pre_layer.n_cols + 1
        pre_col = (pre_i - 1) % pre_layer.n_cols + 1
        
        # Map to post coordinates
        post_row_center = round(Int, (pre_row - 0.5) * row_ratio + 0.5)
        post_col_center = round(Int, (pre_col - 0.5) * col_ratio + 0.5)
        
        # Connect within radius
        for dr in -radius:radius
            for dc in -radius:radius
                post_row = post_row_center + dr
                post_col = post_col_center + dc
                
                if 1 <= post_row <= post_layer.n_rows && 
                   1 <= post_col <= post_layer.n_cols
                    post_i = (post_row - 1) * post_layer.n_cols + post_col
                    
                    # Distance-dependent weight
                    dist = sqrt(dr^2 + dc^2)
                    w = weight * exp(-dist / (radius + 0.1))
                    
                    push!(I_idx, post_i)
                    push!(J_idx, pre_i)
                    push!(V_val, w)
                end
            end
        end
    end
    
    # Update connectivity matrix
    if nt_type == :E
        pathway.E_connectivity[pre_idx][post_idx] = sparse(I_idx, J_idx, V_val, 
            post_layer.n_neurons, pre_layer.n_neurons)
    elseif nt_type == :I
        pathway.I_connectivity[pre_idx][post_idx] = sparse(I_idx, J_idx, V_val,
            post_layer.n_neurons, pre_layer.n_neurons)
    else
        pathway.M_connectivity[pre_idx][post_idx] = sparse(I_idx, J_idx, V_val,
            post_layer.n_neurons, pre_layer.n_neurons)
    end
end

"""
    connect_convergent!(pathway, pre_name, post_name, nt_type; kwargs)

Create convergent connections (many pre → one post, with overlap)
"""
function connect_convergent!(
    pathway::VisualPathway{T},
    pre_name::String,
    post_name::String,
    nt_type::Symbol;
    convergence::Int = 2,
    weight::T = 1.0
) where T
    # Convergent is like topographic with larger radius
    connect_topographic!(pathway, pre_name, post_name, nt_type;
                        radius=convergence, weight=weight/convergence)
end

"""
    connect_divergent!(pathway, pre_name, post_name, nt_type; kwargs)

Create divergent connections (one pre → many post)
"""
function connect_divergent!(
    pathway::VisualPathway{T},
    pre_name::String,
    post_name::String,
    nt_type::Symbol;
    divergence::Int = 2,
    weight::T = 1.0
) where T
    # Similar to topographic but with larger radius in post space
    connect_topographic!(pathway, pre_name, post_name, nt_type;
                        radius=divergence, weight=weight/divergence)
end

#=============================================================================
    NETWORK SIMULATION
=============================================================================#

"""
    compute_synaptic_inputs!(pathway)

Compute total synaptic inputs for all neurons from all sources
"""
function compute_synaptic_inputs!(pathway::VisualPathway{T}) where T
    n_layers = length(pathway.layers)
    
    # Reset received NT
    for layer in pathway.layers
        fill!(layer.E_received, zero(T))
        fill!(layer.I_received, zero(T))
        fill!(layer.M_received, zero(T))
    end
    
    # Compute contributions from each presynaptic layer
    for pre_idx in 1:n_layers
        pre_layer = pathway.layers[pre_idx]
        
        # Extract output NT from presynaptic states
        @views pre_layer.E_output .= pre_layer.states[:, 8]  # E
        @views pre_layer.I_output .= pre_layer.states[:, 9]  # I
        @views pre_layer.M_output .= pre_layer.states[:, 10] # M_mod
        
        for post_idx in 1:n_layers
            post_layer = pathway.layers[post_idx]
            
            # Excitatory
            if nnz(pathway.E_connectivity[pre_idx][post_idx]) > 0
                mul!(post_layer.E_received, 
                     pathway.E_connectivity[pre_idx][post_idx],
                     pre_layer.E_output, 1.0, 1.0)
            end
            
            # Inhibitory
            if nnz(pathway.I_connectivity[pre_idx][post_idx]) > 0
                mul!(post_layer.I_received,
                     pathway.I_connectivity[pre_idx][post_idx],
                     pre_layer.I_output, 1.0, 1.0)
            end
            
            # Modulatory
            if nnz(pathway.M_connectivity[pre_idx][post_idx]) > 0
                mul!(post_layer.M_received,
                     pathway.M_connectivity[pre_idx][post_idx],
                     pre_layer.M_output, 1.0, 1.0)
            end
        end
    end
end

"""
    step_network!(pathway, dt; noise_amp=1.0)

Advance network by one time step using forward Euler
For production use, switch to adaptive RK methods via DifferentialEquations.jl
"""
function step_network!(pathway::VisualPathway{T}, dt::T; noise_amp::T = one(T)) where T
    # Compute synaptic inputs
    compute_synaptic_inputs!(pathway)
    
    du = zeros(T, 11)
    
    for layer in pathway.layers
        for i in 1:layer.n_neurons
            # Get state and params
            @views u = layer.states[i, :]
            p = layer.params[i]
            
            # Generate noise
            noise = noise_amp * randn() * sqrt(dt / p.τ_W)
            
            # Compute derivatives
            morris_lecar_derivatives!(du, u, p, pathway.t;
                E_pre = layer.E_received[i],
                I_pre = layer.I_received[i],
                M_pre = layer.M_received[i],
                noise = noise
            )
            
            # Forward Euler update
            @views layer.states[i, :] .+= du .* dt
        end
    end
    
    pathway.t += dt
end

"""
    simulate!(pathway, duration; dt=0.1, callback=nothing)

Simulate network for specified duration
"""
function simulate!(
    pathway::VisualPathway{T},
    duration::T;
    dt::T = T(0.1),
    save_interval::Int = 10,
    callback = nothing
) where T
    n_steps = round(Int, duration / dt)
    n_save = n_steps ÷ save_interval + 1
    
    # Storage for voltage traces
    n_layers = length(pathway.layers)
    recordings = [zeros(T, layer.n_neurons, n_save) for layer in pathway.layers]
    times = zeros(T, n_save)
    
    save_idx = 1
    for step in 1:n_steps
        step_network!(pathway, dt)
        
        if step % save_interval == 0
            save_idx += 1
            times[save_idx] = pathway.t
            for (l, layer) in enumerate(pathway.layers)
                @views recordings[l][:, save_idx] .= layer.states[:, 1]  # Voltage
            end
            
            if callback !== nothing
                callback(pathway, step)
            end
        end
    end
    
    return times[1:save_idx], recordings
end

end # module Networks
