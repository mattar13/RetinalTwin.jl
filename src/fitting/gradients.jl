import DifferentialEquations

const PARAM_BLOCK_TO_CELLTYPE = Dict(
    :PHOTORECEPTOR_PARAMS => :PC,
    :HORIZONTAL_PARAMS => :HC,
    :ON_BIPOLAR_PARAMS => :ONBC,
    :OFF_BIPOLAR_PARAMS => :OFFBC,
    :A2_AMACRINE_PARAMS => :A2,
    :GANGLION_PARAMS => :GC,
    :MULLER_PARAMS => :MG,
)

function _parse_param_target(target)
    target === nothing && return nothing
    if target isa Tuple{Symbol,Symbol}
        return target
    elseif target isa AbstractString
        parts = split(target, ".")
        length(parts) == 2 || error("target parameter must be \"BLOCK.PARAM\", got: $target")
        return (Symbol(parts[1]), Symbol(parts[2]))
    else
        error("Unsupported target type $(typeof(target)); use nothing, Tuple{Symbol,Symbol}, or \"BLOCK.PARAM\" string")
    end
end

function _parse_param_targets(targets)
    targets === nothing && return nothing
    if targets isa AbstractVector
        return [_parse_param_target(t) for t in targets]
    end
    return [_parse_param_target(targets)]
end

function _set_param_value(params, block::Symbol, pname::Symbol, value::Float64)
    hasproperty(params, block) || error("Unknown parameter block: $block")
    block_params = getproperty(params, block)
    hasproperty(block_params, pname) || error("Unknown parameter $pname in block $block")
    updated_block = merge(block_params, (pname => value,))
    return merge(params, (block => updated_block,))
end

function _list_scalar_params(params)
    entries = NamedTuple[]
    for block in propertynames(params)
        block_params = getproperty(params, block)
        ct = get(PARAM_BLOCK_TO_CELLTYPE, block, :UNKNOWN)
        for pname in propertynames(block_params)
            v = getproperty(block_params, pname)
            if v isa Real && isfinite(v)
                push!(entries, (celltype=ct, block=block, param=pname, value=Float64(v)))
            end
        end
    end
    return entries
end

function _resolve_output_groups(model::RetinalColumnModel, outputs)
    names = ordered_cells_by_offset(model)
    name_to_idx = Dict(nm => i for (i, nm) in enumerate(names))
    cell_types = [model.cells[nm].cell_type for nm in names]

    default_groups = let types = unique(cell_types)
        [(label=String(ct), idx=findall(==(ct), cell_types)) for ct in types]
    end
    outputs === nothing && return default_groups

    specs = outputs isa AbstractVector ? outputs : [outputs]
    groups = NamedTuple[]

    function resolve_token(token)
        s = token isa Symbol ? token : Symbol(String(token))
        if haskey(name_to_idx, s)
            return [name_to_idx[s]], String(s)
        end
        idx = findall(==(s), cell_types)
        isempty(idx) && error("Unknown output token: $token. Use a cell name (e.g. :ONBC1) or cell type (e.g. :ONBC).")
        return idx, String(s)
    end

    for spec in specs
        if spec isa Pair
            label = String(spec.first)
            rhs = spec.second isa AbstractVector ? spec.second : [spec.second]
            idx = Int[]
            for token in rhs
                token_idx, _ = resolve_token(token)
                append!(idx, token_idx)
            end
            idx = sort!(unique(idx))
            isempty(idx) && error("Output group $label resolved to no cells")
            push!(groups, (label=label, idx=idx))
        else
            idx, label = resolve_token(spec)
            push!(groups, (label=label, idx=idx))
        end
    end
    return groups
end

function _dark_adapt(model, u0, params; tspan_dark=(0.0, 2000.0), abstol=1e-6, reltol=1e-4)
    stim_dark = make_uniform_flash_stimulus(photon_flux=0.0)
    prob_dark = DifferentialEquations.ODEProblem(model, u0, tspan_dark, (params, stim_dark))
    sol_dark = DifferentialEquations.solve(
        prob_dark,
        DifferentialEquations.Rodas5();
        save_everystep=false,
        save_start=false,
        save_end=true,
        abstol=abstol,
        reltol=reltol,
    )
    return sol_dark.u[end]
end

function _fit_ir_outputs(
    model,
    u0,
    params;
    outputs=nothing,
    stim_start=80.0,
    stim_end=180.0,
    intensity_levels=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    tspan=(0.0, 1200.0),
    abstol=1e-6,
    reltol=1e-4,
)
    names = _ordered_cells(model)
    n_cells = length(names)
    nI = length(intensity_levels)
    peak_dv = fill(NaN, nI, n_cells)

    baseline_window = (stim_start - 50.0, stim_start)
    response_window = (stim_start, stim_end + 400.0)

    for (i, photon_flux) in enumerate(intensity_levels)
        selected_stimulus = make_uniform_flash_stimulus(
            stim_start=stim_start,
            stim_end=stim_end,
            photon_flux=photon_flux,
        )
        prob = DifferentialEquations.ODEProblem(model, u0, tspan, (params, selected_stimulus))
        sol = DifferentialEquations.solve(
            prob,
            DifferentialEquations.Rodas5();
            tstops=[stim_start, stim_end],
            saveat=1.0,
            abstol=abstol,
            reltol=reltol,
        )

        baseline_idx = findall(t -> baseline_window[1] <= t <= baseline_window[2], sol.t)
        response_idx = findall(t -> response_window[1] <= t <= response_window[2], sol.t)

        for (j, nm) in enumerate(names)
            v = state_trace(sol, model, nm, :V)
            if isempty(baseline_idx) || isempty(response_idx)
                peak_dv[i, j] = NaN
                continue
            end
            baseline = mean(v[baseline_idx])
            peak_dv[i, j] = maximum(abs.(v[response_idx] .- baseline))
        end
    end

    groups = _resolve_output_groups(model, outputs)
    fits = Dict{String,NamedTuple}()
    for g in groups
        idx = g.idx
        y = [finite_mean(@view peak_dv[i, idx]) for i in 1:nI]
        fits[g.label] = fit_hill_ir(intensity_levels, y)
    end
    return fits
end

"""
    calculate_ir_gradient(model, u0, param_target; alpha=1e-3, params=default_retinal_params(), outputs=nothing, fit_kwargs...)

Compute finite-difference gradients of IR Hill-fit metrics with respect to one parameter
or a list of parameters.

The returned rows include:
- `celltype`: measured output label (cell type, cell name, or custom group label)
- `param`: parameter name
- `value`: baseline parameter value
- `k_gradient`: dK/dp estimate
- `n_gradiend`: dn/dp estimate
- `rmax_gradient`: dA/dp estimate (`A` is Hill maximum response)

# Arguments
- `param_target`:
  - Single target: `"BLOCK.PARAM"` or `(Symbol(:BLOCK), Symbol(:PARAM))`
  - Multiple targets: vector of either format

# Keyword arguments
- `alpha`: relative perturbation scale, where `delta = alpha * max(abs(value), 1.0)`
- `params`: parameter NamedTuple to perturb
- `outputs`:
  - `nothing` for default groups by cell type
  - `:ONBC` or `:ONBC1`
  - vector like `[:ONBC, :A2, :GC]`
  - grouped form like `["ONBC_pool" => [:ONBC1, :ONBC2], :GC]`
- `fit_kwargs`: forwarded to simulation/fitting internals (e.g., `stim_start`, `stim_end`, `intensity_levels`, `tspan`, `abstol`, `reltol`)
"""
function calculate_ir_gradient(
    model,
    u0,
    param_target;
    alpha=1e-3,
    params=default_retinal_params(),
    outputs=nothing,
    fit_kwargs...,
)
    parsed_targets = _parse_param_targets(param_target)
    parsed_targets === nothing && error("param_target must specify one or more parameters")
    base_u0 = _dark_adapt(model, u0, params; fit_kwargs...)
    base_fits = _fit_ir_outputs(model, base_u0, params; outputs=outputs, fit_kwargs...)
    param_rows = _list_scalar_params(params)

    rows = NamedTuple[]
    for t in parsed_targets
        block, pname = t
        candidates = [r for r in param_rows if r.block == block && r.param == pname]
        isempty(candidates) && error("No parameter matched target=$(block).$(pname)")
        row = only(candidates)
        append!(
            rows,
            _gradient_for_param(
                model,
                u0,
                params,
                base_fits,
                row.block,
                row.param,
                row.value;
                alpha=alpha,
                outputs=outputs,
                fit_kwargs...,
            ),
        )
    end
    return rows
end

function _gradient_for_param(
    model,
    u0,
    base_params,
    base_fits,
    block::Symbol,
    pname::Symbol,
    value::Float64;
    alpha=1e-3,
    outputs=nothing,
    fit_kwargs...,
)
    scale = max(abs(value), 1.0)
    delta = alpha * scale
    perturbed_value = value + delta

    perturbed_params = _set_param_value(base_params, block, pname, perturbed_value)
    perturbed_u0 = _dark_adapt(model, u0, perturbed_params; fit_kwargs...)
    perturbed_fits = _fit_ir_outputs(model, perturbed_u0, perturbed_params; outputs=outputs, fit_kwargs...)
    output_labels = [g.label for g in _resolve_output_groups(model, outputs)]

    rows = NamedTuple[]
    for output_name in output_labels
        base = get(base_fits, output_name, (A=NaN, K=NaN, n=NaN, sse=NaN, ok=false))
        pert = get(perturbed_fits, output_name, (A=NaN, K=NaN, n=NaN, sse=NaN, ok=false))
        push!(
            rows,
            (
                celltype=output_name,
                param=String(pname),
                value=value,
                k_gradient=(pert.K - base.K) / delta,
                n_gradiend=(pert.n - base.n) / delta,
                rmax_gradient=(pert.A - base.A) / delta,
            ),
        )
    end
    return rows
end

"""
    run_gradient_calculation(model, u0; target=nothing, alpha=1e-3, params=default_retinal_params(), outputs=nothing, out_csv=nothing, fit_kwargs...)

Run IR-gradient calculations for selected parameters and optionally save results to CSV.

# Keyword arguments
- `target`:
  - `nothing`: all scalar parameters from `params`
  - single parameter: `"BLOCK.PARAM"` or `(Symbol(:BLOCK), Symbol(:PARAM))`
  - multiple parameters: vector of either format
- `outputs`: measured outputs to fit/compare (same formats as `calculate_ir_gradient`)
- `out_csv`: if provided, writes columns
  `celltype,param,value,k_gradient,n_gradiend,rmax_gradient`
- `fit_kwargs`: forwarded to simulation/fitting internals

# Returns
- Vector of NamedTuples, one row per `(output, parameter)` pair.
"""
function run_gradient_calculation(
    model,
    u0;
    target=nothing,
    alpha=1e-3,
    params=default_retinal_params(),
    outputs=nothing,
    out_csv=nothing,
    fit_kwargs...,
)
    selected_targets = if target === nothing
        [(row.block, row.param, row.value) for row in _list_scalar_params(params)]
    else
        parsed_targets = _parse_param_targets(target)
        param_rows = _list_scalar_params(params)
        rows = Tuple{Symbol,Symbol,Float64}[]
        for t in parsed_targets
            block, pname = t
            matches = [row for row in param_rows if row.block == block && row.param == pname]
            isempty(matches) && error("No parameter matched target=$(block).$(pname)")
            push!(rows, (block, pname, only(matches).value))
        end
        rows
    end

    base_u0 = _dark_adapt(model, u0, params; fit_kwargs...)
    base_fits = _fit_ir_outputs(model, base_u0, params; outputs=outputs, fit_kwargs...)

    rows = NamedTuple[]
    for (i, t) in enumerate(selected_targets)
        println("[$i/$(length(selected_targets))] $(t[1]).$(t[2])")
        append!(
            rows,
            _gradient_for_param(
                model,
                u0,
                params,
                base_fits,
                t[1],
                t[2],
                t[3];
                alpha=alpha,
                outputs=outputs,
                fit_kwargs...,
            ),
        )
    end

    if out_csv !== nothing
        mkpath(dirname(out_csv))
        open(out_csv, "w") do io
            println(io, "celltype,param,value,k_gradient,n_gradiend,rmax_gradient")
            for r in rows
                println(io, "$(r.celltype),$(r.param),$(r.value),$(r.k_gradient),$(r.n_gradiend),$(r.rmax_gradient)")
            end
        end
    end
    return rows
end
