# ============================================================
# retinal_column.jl — Retinal column model with direct coupling
# ============================================================

"""
    default_retinal_params()

Load default parameters for photoreceptor and ON bipolar cells.

# Returns
NamedTuple with:
- `PHOTORECEPTOR_PARAMS`: Rod photoreceptor parameters
- `ON_BIPOLAR_PARAMS`: ON bipolar cell parameters
- `OFF_BIPOLAR_PARAMS`: OFF bipolar cell parameters
- `A2_AMACRINE_PARAMS`: A2 amacrine cell parameters
"""
function default_retinal_params(; editable::Bool=false)
    all_params = load_all_params(; editable=false)
    params_nt = (
        PHOTORECEPTOR_PARAMS = all_params.PHOTORECEPTOR_PARAMS,
        HORIZONTAL_PARAMS = all_params.HORIZONTAL_PARAMS,
        ON_BIPOLAR_PARAMS = all_params.ON_BIPOLAR_PARAMS,
        OFF_BIPOLAR_PARAMS = all_params.OFF_BIPOLAR_PARAMS,
        A2_AMACRINE_PARAMS = all_params.A2_AMACRINE_PARAMS,
        GANGLION_PARAMS = all_params.GANGLION_PARAMS,
        MULLER_PARAMS = all_params.MULLER_PARAMS
    )
    return editable ? namedtuple_to_dict(params_nt) : params_nt
end

# ── Initial conditions ──────────────────────────────────────

"""
    retinal_column_initial_conditions(params)

Build initial conditions for photoreceptor + ON bipolar system.

# Arguments
- `params`: NamedTuple from `default_retinal_params()`

# Returns
- 25-element state vector [photoreceptor(21), on_bipolar(6), off_bipolar(7), a2(7), ganglion(6), muller(4)]
"""
function retinal_column_initial_conditions(params)
    params = params isa AbstractDict ? dict_to_namedtuple(params) : params
    # Get individual cell initial conditions
    ic_size = 0
    u0_photoreceptor = photoreceptor_state(params.PHOTORECEPTOR_PARAMS)
    println("Size of photoreceptor state vector: $(length(u0_photoreceptor))")
    ic_size += length(u0_photoreceptor)
    println("IC size after photoreceptor: $ic_size")

    u0_on_bipolar = on_bipolar_state(params.ON_BIPOLAR_PARAMS)
    println("Size of on bipolar state vector: $(length(u0_on_bipolar))")
    ic_size += length(u0_on_bipolar)
    println("IC size after on bipolar: $ic_size")

    u0_off_bipolar = off_bipolar_state(params.OFF_BIPOLAR_PARAMS)
    println("Size of off bipolar state vector: $(length(u0_off_bipolar))")
    ic_size += length(u0_off_bipolar)
    println("IC size after off bipolar: $ic_size")

    u0_a2 = a2_amacrine_state(params.A2_AMACRINE_PARAMS)
    println("Size of a2 state vector: $(length(u0_a2))")
    ic_size += length(u0_a2)
    println("IC size after a2: $ic_size")

    u0_gc = ganglion_state(params.GANGLION_PARAMS)
    println("Size of ganglion state vector: $(length(u0_gc))")
    ic_size += length(u0_gc)
    println("IC size after ganglion: $ic_size")

    u0_muller = muller_state(params.MULLER_PARAMS)
    println("Size of muller state vector: $(length(u0_muller))")
    ic_size += length(u0_muller)
    println("IC size after muller: $ic_size")

    println("Total size of initial condition vector: $ic_size")
    # Concatenate into single state vector
    return vcat(u0_photoreceptor, u0_on_bipolar, u0_off_bipolar, u0_a2, u0_gc, u0_muller)
end

# ── Auxillary Functions ─────────────────────────────────────
#The only auxillary function we need to worry about is the gap junction coupling function

function gap_junction_coupling(dV1, V1, dV2, V2, cm1, cm2, g_gap)
    I_gap1 = g_gap * (V1 - V2)
    dV1 = -I_gap1 / cm1
    dV2 = I_gap1 / cm2
    return nothing
end

function (RCM::RetinalColumnModel)(du, u, p, t)
    params, stim_func = p
    params = params isa AbstractDict ? dict_to_namedtuple(params) : params

    for cell in values(RCM.cells)
        if !(cell.cell_type in (:PC, :HC, :ONBC, :OFFBC, :A2, :GC, :MG))
            error("RetinalColumnModel callable supports only :PC, :HC, :ONBC, :OFFBC, :A2, :GC, and :MG")
        end
    end

    for cell in values(RCM.cells)
        if cell.cell_type == :PC
            uc = uview(u, cell)
            duc = duview(du, cell)
            x = cell.x
            y = cell.y
            pc_stim(tt) = stim_func(tt, x, y)
            photoreceptor_model!(duc, uc, (params.PHOTORECEPTOR_PARAMS, pc_stim), t)
        elseif cell.cell_type == :HC
            uc = uview(u, cell)
            duc = duview(du, cell)
            inputs = get(RCM.connections, cell.name, Tuple{Symbol,Symbol,Float64}[])
            glu_in = [get_out(u, RCM.cells[pre], key) for (pre, key, _) in inputs if key == :Glu]
            w_glu_in = [w for (_, key, w) in inputs if key == :Glu]
            V_hc = get_out(u, cell, :V)
            I_gap = params.HORIZONTAL_PARAMS.g_gap *
                sum(w * (get_out(u, RCM.cells[pre], key) - V_hc) for (pre, key, w) in inputs if key == :V)
            horizontal_model!(duc, uc, (params.HORIZONTAL_PARAMS, I_gap, glu_in, w_glu_in), t)
        end
        if cell.cell_type == :ONBC
            uc = uview(u, cell)
            duc = duview(du, cell)
            inputs = get(RCM.connections, cell.name, Tuple{Symbol,Symbol,Float64}[])
            glu_in = [get_out(u, RCM.cells[pre], key) for (pre, key, _) in inputs if key == :Glu]
            w_glu_in = [w for (_, key, w) in inputs if key == :Glu]
            V_onbc = get_out(u, cell, :V)
            I_gap = params.A2_AMACRINE_PARAMS.g_gap *
                sum(w * (get_out(u, RCM.cells[pre], key) - V_onbc) for (pre, key, w) in inputs if key == :V)
            on_bipolar_model!(duc, uc, (params.ON_BIPOLAR_PARAMS, glu_in, w_glu_in), t)
            duc[ONBC_IC_MAP.V] += I_gap / params.ON_BIPOLAR_PARAMS.C_m
        elseif cell.cell_type == :OFFBC
            uc = uview(u, cell)
            duc = duview(du, cell)
            inputs = get(RCM.connections, cell.name, Tuple{Symbol,Symbol,Float64}[])
            glu_in = [get_out(u, RCM.cells[pre], key) for (pre, key, _) in inputs if key == :Glu]
            w_glu_in = [w for (_, key, w) in inputs if key == :Glu]
            off_bipolar_model!(duc, uc, (params.OFF_BIPOLAR_PARAMS, glu_in, w_glu_in), t)
        elseif cell.cell_type == :A2
            uc = uview(u, cell)
            duc = duview(du, cell)
            inputs = get(RCM.connections, cell.name, Tuple{Symbol,Symbol,Float64}[])
            glu_in = [get_out(u, RCM.cells[pre], key) for (pre, key, _) in inputs if key == :Glu]
            w_glu_in = [w for (_, key, w) in inputs if key == :Glu]
            V_a2 = get_out(u, cell, :V)
            I_gap = params.A2_AMACRINE_PARAMS.g_gap *
                sum(w * (get_out(u, RCM.cells[pre], key) - V_a2) for (pre, key, w) in inputs if key == :V)
            a2_model!(duc, uc, (params.A2_AMACRINE_PARAMS, glu_in, w_glu_in), t)
            duc[A2_IC_MAP.V] += I_gap / params.A2_AMACRINE_PARAMS.C_m
        elseif cell.cell_type == :GC
            uc = uview(u, cell)
            duc = duview(du, cell)
            inputs = get(RCM.connections, cell.name, Tuple{Symbol,Symbol,Float64}[])
            glu_in = [get_out(u, RCM.cells[pre], key) for (pre, key, _) in inputs if key == :Glu]
            w_glu_in = [w for (_, key, w) in inputs if key == :Glu]
            gly_in = [get_out(u, RCM.cells[pre], key) for (pre, key, _) in inputs if key == :Y || key == :Gly]
            w_gly_in = [w for (_, key, w) in inputs if key == :Y || key == :Gly]
            ganglion_model!(duc, uc, (params.GANGLION_PARAMS, glu_in, w_glu_in, gly_in, w_gly_in), t)
        elseif cell.cell_type == :MG
            uc = uview(u, cell)
            duc = duview(du, cell)

            K_photo_efflux = zero(eltype(u))
            K_onbc_efflux = zero(eltype(u))
            K_offbc_efflux = zero(eltype(u))
            K_a2_efflux = zero(eltype(u))
            K_gc_efflux = zero(eltype(u))
            glu_gc_exc = zero(eltype(u))

            for src in values(RCM.cells)
                usc = uview(u, src)
                if src.cell_type == :PC
                    K_photo_efflux += photoreceptor_K_efflux(usc, params.PHOTORECEPTOR_PARAMS)
                elseif src.cell_type == :ONBC
                    K_onbc_efflux += on_bipolar_K_efflux(usc, params.ON_BIPOLAR_PARAMS)
                    glu_gc_exc += get_out(u, src, :Glu)
                elseif src.cell_type == :OFFBC
                    K_offbc_efflux += off_bipolar_K_efflux(usc, params.OFF_BIPOLAR_PARAMS)
                    glu_gc_exc += get_out(u, src, :Glu)
                elseif src.cell_type == :A2
                    K_a2_efflux += a2_amacrine_K_efflux(usc, params.A2_AMACRINE_PARAMS)
                elseif src.cell_type == :GC
                    K_gc_efflux += ganglion_K_efflux(usc, params.GANGLION_PARAMS)
                end
            end

            I_K_src_stalk = K_photo_efflux + K_onbc_efflux + K_offbc_efflux + K_a2_efflux + K_gc_efflux
            I_K_src_end = 0.2 * (K_a2_efflux + K_gc_efflux) + 0.1 * K_onbc_efflux
            muller_model!(duc, uc, (params.MULLER_PARAMS, I_K_src_end, I_K_src_stalk, glu_gc_exc), t)
        end
    end

    return nothing
end