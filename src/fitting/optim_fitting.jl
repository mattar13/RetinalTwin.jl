# ============================================================
# optim_fitting.jl — Optim.jl-based ERG model fitting
#
# Fits model parameters to real ERG data using staged optimization.
# Works for any ERG component (a-wave, b-wave, etc.) by selecting
# which cell types to optimize and which time window to fit in.
#
# Usage:
#   result = fit_erg(model, u0, params;
#       cell_types=[:PHOTO],
#       stimuli=[(intensity=1.0, duration_sec=0.01), ...],
#       real_t=[t1, t2, ...],           # Vector of time vectors per intensity
#       real_traces=[y1, y2, ...],       # Vector of trace vectors per intensity
#       time_window=(0.5, 1.5),
#   )
# ============================================================

using Optim
using DataFrames
using CairoMakie
import DifferentialEquations

# ── Parameter packing/unpacking ──────────────────────────────

"""
    fittable_params(cell_types::Vector{Symbol}; csv_path)

Return a vector of `(cell_type, key, spec::ParameterSpec)` tuples for all
non-fixed parameters in the given cell types where lower != upper.
"""
function fittable_params(cell_types::Vector{Symbol}; csv_path::String=default_param_csv_path())
    all_specs = load_all_param_specs(csv_path=csv_path)
    fit_list = Tuple{Symbol, Symbol, ParameterSpec}[]
    for ct in cell_types
        ct_norm = _normalize_cell_type(ct)
        hasproperty(all_specs, ct_norm) || continue
        cell_specs = getproperty(all_specs, ct_norm)
        for k in keys(cell_specs)
            spec = getproperty(cell_specs, k)
            spec.fixed && continue
            spec.lower == spec.upper && continue
            push!(fit_list, (ct_norm, k, spec))
        end
    end
    return fit_list
end

"""
    _use_log_transform(spec::ParameterSpec)

Determine if a parameter should be log-transformed for optimization.
Log-transform when the parameter and its bounds are strictly positive.
"""
_use_log_transform(spec::ParameterSpec) = spec.lower > 0 && spec.value > 0

"""
    pack_params(params::NamedTuple, fit_list)

Pack fittable parameter values from a NamedTuple into an unconstrained
optimizer vector. Applies log-transform for strictly positive parameters.
"""
function pack_params(params::NamedTuple, fit_list)
    theta = Vector{Float64}(undef, length(fit_list))
    for (i, (ct, key, spec)) in enumerate(fit_list)
        val = Float64(getproperty(getproperty(params, ct), key))
        theta[i] = _use_log_transform(spec) ? log(val) : val
    end
    return theta
end

"""
    unpack_params(theta, base_params::NamedTuple, fit_list)

Unpack optimizer vector back into a params NamedTuple. Applies inverse
transform and clamps to bounds.
"""
function unpack_params(theta::AbstractVector{<:Real}, base_params::NamedTuple, fit_list)
    p = base_params
    for (i, (ct, key, spec)) in enumerate(fit_list)
        raw = _use_log_transform(spec) ? exp(theta[i]) : theta[i]
        val = clamp(raw, spec.lower, spec.upper)
        block = getproperty(p, ct)
        updated = merge(block, NamedTuple{(key,)}((val,)))
        p = merge(p, NamedTuple{(ct,)}((updated,)))
    end
    return p
end

# ── Simulation ───────────────────────────────────────────────



"""
    simulate_erg(model, u0_dark, params, stimuli; tspan=(0.0, 6.0), dt=0.01)

Simulate ERG traces for multiple stimulus intensities.

# Arguments
- `stimuli`: Vector of NamedTuples with fields `intensity` and `duration_sec`
- Returns `(t_grid, traces)` where traces is a Vector{Vector{Float64}}

Failed ODE solves produce NaN-filled traces rather than throwing.
"""
function simulate_erg(model, u0_dark, params, stimuli; tspan=(0.0, 6.0), dt=0.01)
    t_grid = collect(range(tspan[1], tspan[2]; step=dt))
    traces = Vector{Vector{Float64}}(undef, length(stimuli))

    for (i, s) in enumerate(stimuli)
        try
            stim = make_uniform_flash_stimulus(
                stim_start=0.0,
                stim_end=s.duration_sec,
                photon_flux=s.intensity,
            )
            prob = DifferentialEquations.ODEProblem(model, u0_dark, tspan, (params, stim))
            sol = DifferentialEquations.solve(
                prob, DifferentialEquations.Rodas5();
                tstops=[0.0, s.duration_sec],
                abstol=1e-6, reltol=1e-4,
            )
            t_erg, erg = compute_field_potential(model, params, sol; dt=dt)
            traces[i] = erg
        catch
            traces[i] = fill(NaN, length(t_grid))
        end
    end

    return t_grid, traces
end

# ── Loss function ────────────────────────────────────────────

"""
    residual_traces(t_sim, sim_traces, real_t, real_traces; absolute=false)

Compute pointwise residual traces (`real - sim`) across all intensities using
the same time-alignment strategy as fitting (`searchsortedlast` on `t_sim`).

Set `absolute=true` to return `abs(real - sim)` at each sample.
Returns an empty vector when inputs are invalid.
"""
function residual_traces(
    t_sim::AbstractVector{<:Real},
    sim_traces::Vector{<:AbstractVector},
    real_t::Vector{<:AbstractVector},
    real_traces::Vector{<:AbstractVector};
    absolute::Bool=false,
)
    length(sim_traces) == length(real_t) == length(real_traces) || return Vector{Vector{Float64}}()
    isempty(t_sim) && return Vector{Vector{Float64}}()

    out = Vector{Vector{Float64}}(undef, length(sim_traces))
    for i in eachindex(sim_traces)
        st = sim_traces[i]
        rt = real_t[i]
        ry = real_traces[i]

        isempty(st) && return Vector{Vector{Float64}}()
        any(isnan, st) && return Vector{Vector{Float64}}()
        length(rt) == length(ry) || return Vector{Vector{Float64}}()

        resid = Vector{Float64}(undef, length(rt))
        for j in eachindex(rt)
            k = clamp(searchsortedlast(t_sim, rt[j]), 1, length(t_sim))
            d = ry[j] - st[k]
            resid[j] = absolute ? abs(d) : d
        end
        out[i] = resid
    end
    return out
end

"""
    mean_squared_error(t_sim, sim_traces, real_t, real_traces)

Compute full-waveform mean squared error between real and simulated ERG traces
across all intensities.

For each real sample time point, the simulated value is sampled using
`searchsortedlast` on `t_sim`, matching the existing fitting alignment logic.
Returns `Inf` when traces are invalid, simulation failed, or no samples exist.
"""
function mean_squared_error(
    t_sim::AbstractVector{<:Real},
    sim_traces::Vector{<:AbstractVector},
    real_t::Vector{<:AbstractVector},
    real_traces::Vector{<:AbstractVector},
)
    residuals = residual_traces(t_sim, sim_traces, real_t, real_traces)
    isempty(residuals) && return Inf
    n_total = sum(length, residuals)
    n_total > 0 || return Inf
    total_loss = sum(sum(abs2, r) for r in residuals)
    return total_loss / n_total
end

"""
    erg_loss(theta, model, u0, base_params, fit_list, stimuli, real_t, real_traces;
             time_window, tspan, dt)

Compute full-waveform mean squared error between simulated and real ERG traces
across all intensities.

Returns `Inf` if simulation fails for any intensity.
"""
function erg_loss(
    theta::AbstractVector{<:Real},
    model, u0, base_params, fit_list, stimuli,
    real_t::Vector{<:AbstractVector}, real_traces::Vector{<:AbstractVector};
    time_window=(0.5, 1.5),
    tspan=(0.0, 6.0),
    dt=0.01,
)
    params = unpack_params(theta, base_params, fit_list)

    # Dark adapt with current params
    u0_dark = try
        _optim_dark_adapt(model, u0, params; tspan_dark=(0.0, 2000.0))
    catch
        return Inf
    end

    t_sim, sim_traces = simulate_erg(model, u0_dark, params, stimuli; tspan=tspan, dt=dt)
    return mean_squared_error(t_sim, sim_traces, real_t, real_traces)
end

# ── Main fitting function ────────────────────────────────────

"""
    fit_erg(model, u0, base_params; cell_types, stimuli, real_t, real_traces,
            time_window, tspan, dt, nm_iterations, lbfgs_iterations, run_lbfgs, verbose)

Fit ERG model parameters to real data using staged optimization.

# Arguments
- `cell_types`: Vector of cell type symbols to optimize, e.g. `[:PHOTO]`
- `stimuli`: Vector of `(intensity=Float64, duration_sec=Float64)` per intensity
- `real_t`: Vector of time vectors (one per intensity), from ElectroPhysiology.jl
- `real_traces`: Vector of trace vectors (one per intensity)
- `time_window`: retained for backward compatibility; currently ignored
- `nm_iterations`: NelderMead iterations (default 500)
- `lbfgs_iterations`: LBFGS iterations (default 100)
- `run_lbfgs`: whether to run LBFGS refinement phase (default true)

# Returns
NamedTuple with fields:
- `params`: fitted parameter NamedTuple (can be used directly with the model)
- `loss`: final loss value
- `fit_list`: vector of (cell_type, key, spec) for fitted parameters
- `theta`: optimized parameter vector in transformed space
- `uncertainty`: DataFrame with estimates, std_error, ci95, certainty
- `t_sim`: simulation time grid
- `sim_traces`: fitted simulated traces
- `convergence`: Optim convergence info
"""
function fit_erg(
    model, u0, base_params::NamedTuple;
    cell_types::Vector{Symbol},
    stimuli,
    real_t::Vector{<:AbstractVector},
    real_traces::Vector{<:AbstractVector},
    time_window=(0.5, 1.5),
    tspan=(0.0, 6.0),
    dt=0.01,
    nm_iterations=500,
    lbfgs_iterations=100,
    run_lbfgs=true,
    verbose=true,
)
    fit_list = fittable_params(cell_types)
    n_params = length(fit_list)

    if verbose
        println("Fitting $(n_params) parameters from cell types: $(cell_types)")
        println("Loss: full-waveform MSE, $(length(stimuli)) intensities")
        for (i, (ct, key, spec)) in enumerate(fit_list)
            println("  [$i] $(ct).$(key) = $(spec.value) ∈ [$(spec.lower), $(spec.upper)]")
        end
    end

    theta0 = pack_params(base_params, fit_list)

    # Build objective closure
    objective = theta -> erg_loss(
        theta, model, u0, base_params, fit_list, stimuli,
        real_t, real_traces;
        time_window=time_window, tspan=tspan, dt=dt,
    )

    # Check initial loss
    loss0 = objective(theta0)
    verbose && println("\nInitial loss: $(round(loss0, sigdigits=6))")

    # Phase 1: NelderMead — derivative-free, robust to ODE instabilities
    verbose && println("\n── Phase 1: NelderMead ($(nm_iterations) iterations) ──")
    res_nm = Optim.optimize(
        objective, theta0, NelderMead(),
        Optim.Options(
            iterations=nm_iterations,
            show_trace=verbose,
            show_every=50,
            f_tol=1e-10,
            x_tol=1e-8,
        ),
    )
    theta_best = Optim.minimizer(res_nm)
    verbose && println("NelderMead loss: $(round(Optim.minimum(res_nm), sigdigits=6))")

    # Phase 2: LBFGS — gradient-based refinement near optimum
    res_lbfgs = nothing
    if run_lbfgs
        verbose && println("\n── Phase 2: LBFGS ($(lbfgs_iterations) iterations) ──")
        try
            res_lbfgs = Optim.optimize(
                objective, theta_best, LBFGS(),
                Optim.Options(
                    iterations=lbfgs_iterations,
                    show_trace=verbose,
                    show_every=20,
                    f_tol=1e-12,
                    x_tol=1e-10,
                ),
            )
            theta_best = Optim.minimizer(res_lbfgs)
            verbose && println("LBFGS loss: $(round(Optim.minimum(res_lbfgs), sigdigits=6))")
        catch
            verbose && println("LBFGS failed, using NelderMead result")
        end
    end

    # Reconstruct final params and simulate
    best_params = unpack_params(theta_best, base_params, fit_list)
    u0_dark = _optim_dark_adapt(model, u0, best_params)
    t_sim, sim_traces = simulate_erg(model, u0_dark, best_params, stimuli; tspan=tspan, dt=dt)

    final_loss = res_lbfgs !== nothing ? Optim.minimum(res_lbfgs) : Optim.minimum(res_nm)

    # Uncertainty estimation
    verbose && println("\n── Estimating parameter uncertainty ──")
    unc = estimate_uncertainty(theta_best, objective, fit_list)
    verbose && println(unc)

    convergence_info = (
        nm_converged = Optim.converged(res_nm),
        nm_iterations = Optim.iterations(res_nm),
        nm_loss = Optim.minimum(res_nm),
        lbfgs_converged = res_lbfgs !== nothing ? Optim.converged(res_lbfgs) : missing,
        lbfgs_loss = res_lbfgs !== nothing ? Optim.minimum(res_lbfgs) : missing,
        initial_loss = loss0,
        final_loss = final_loss,
    )

    return (
        params = best_params,
        loss = final_loss,
        fit_list = fit_list,
        theta = theta_best,
        uncertainty = unc,
        t_sim = t_sim,
        sim_traces = sim_traces,
        convergence = convergence_info,
    )
end

# ── Uncertainty estimation ───────────────────────────────────

"""
    estimate_uncertainty(theta_opt, objective, fit_list; step=1e-3)

Estimate parameter uncertainty via finite-difference Hessian diagonal.
Also detects strongly correlated parameter pairs via off-diagonal elements.

Returns a DataFrame with columns:
  cell_type, parameter, estimate, std_error, ci95_lower, ci95_upper,
  certainty, correlated_with
"""
function estimate_uncertainty(
    theta_opt::Vector{Float64},
    objective,
    fit_list;
    step=1e-3,
)
    n = length(theta_opt)
    f0 = objective(theta_opt)

    # Diagonal Hessian elements
    hess_diag = Vector{Float64}(undef, n)
    for i in 1:n
        h = max(abs(theta_opt[i]) * step, 1e-6)
        e = zeros(n); e[i] = 1.0
        fp = objective(theta_opt .+ h .* e)
        fm = objective(theta_opt .- h .* e)
        hess_diag[i] = (fp - 2f0 + fm) / h^2
    end

    # Off-diagonal elements for correlation detection (within same cell type)
    # Only check pairs within the same cell type to keep cost manageable
    corr_pairs = Dict{Int, Vector{Tuple{Int, Float64}}}()
    cell_groups = Dict{Symbol, Vector{Int}}()
    for (i, (ct, _, _)) in enumerate(fit_list)
        group = get!(cell_groups, ct, Int[])
        push!(group, i)
    end

    for (ct, indices) in cell_groups
        for a in 1:length(indices)
            for b in (a+1):length(indices)
                i, j = indices[a], indices[b]
                hi = max(abs(theta_opt[i]) * step, 1e-6)
                hj = max(abs(theta_opt[j]) * step, 1e-6)
                ei = zeros(n); ei[i] = 1.0
                ej = zeros(n); ej[j] = 1.0

                fpp = objective(theta_opt .+ hi .* ei .+ hj .* ej)
                fpm = objective(theta_opt .+ hi .* ei .- hj .* ej)
                fmp = objective(theta_opt .- hi .* ei .+ hj .* ej)
                fmm = objective(theta_opt .- hi .* ei .- hj .* ej)
                h_ij = (fpp - fpm - fmp + fmm) / (4 * hi * hj)

                # Compute correlation coefficient
                if hess_diag[i] > 0 && hess_diag[j] > 0
                    corr = h_ij / sqrt(hess_diag[i] * hess_diag[j])
                    if abs(corr) > 0.7
                        pairs_i = get!(corr_pairs, i, Tuple{Int, Float64}[])
                        pairs_j = get!(corr_pairs, j, Tuple{Int, Float64}[])
                        push!(pairs_i, (j, corr))
                        push!(pairs_j, (i, corr))
                    end
                end
            end
        end
    end

    # Build results
    rows = NamedTuple[]
    for (i, (ct, key, spec)) in enumerate(fit_list)
        raw_val = _use_log_transform(spec) ? exp(theta_opt[i]) : theta_opt[i]

        # Standard error from Hessian diagonal
        if hess_diag[i] > 0
            var_theta = 1.0 / hess_diag[i]
            se_theta = sqrt(var_theta)
            # Delta method for log-transformed params: se_original = val * se_theta
            se = _use_log_transform(spec) ? raw_val * se_theta : se_theta
        else
            se = NaN
        end

        ci_lo = isfinite(se) ? raw_val - 1.96 * se : NaN
        ci_hi = isfinite(se) ? raw_val + 1.96 * se : NaN
        certainty = isfinite(se) && raw_val != 0 ? 1.0 / (1.0 + abs(se / raw_val)) : 0.0

        # Correlated parameters
        corr_str = ""
        if haskey(corr_pairs, i)
            parts = String[]
            for (j, c) in corr_pairs[i]
                _, k, _ = fit_list[j]
                push!(parts, "$(k)($(round(c, digits=2)))")
            end
            corr_str = join(parts, "; ")
        end

        push!(rows, (
            cell_type = String(ct),
            parameter = String(key),
            estimate = raw_val,
            std_error = se,
            ci95_lower = ci_lo,
            ci95_upper = ci_hi,
            certainty = certainty,
            correlated_with = corr_str,
        ))
    end

    return DataFrame(rows)
end

# ── Diagnostics ──────────────────────────────────────────────

"""
    plot_fit_diagnostics(result, real_t, real_traces, stimuli;
                         time_window, outdir)

Generate diagnostic plots for a fitting result.

Produces:
1. `fit_traces.png` — Real vs fitted ERG traces per intensity
2. `fit_residuals.png` — Residual traces in the fit window
3. `fit_parameters.png` — Parameter estimates with 95% CI
4. `fit_correlations.png` — Correlation flags between parameters (if any)
"""
function plot_fit_diagnostics(
    result,
    real_t::Vector{<:AbstractVector},
    real_traces::Vector{<:AbstractVector},
    stimuli;
    time_window=(0.5, 1.5),
    outdir=".",
)
    mkpath(outdir)
    nI = length(stimuli)
    intensities = [s.intensity for s in stimuli]
    log_I = log10.(intensities)
    I_min, I_max = extrema(log_I)
    palette = cgrad(:viridis)
    colors = I_max > I_min ?
        get.(Ref(palette), (log_I .- I_min) ./ (I_max - I_min .+ eps())) :
        fill(palette[0.5], nI)

    # ── Fig 1: Traces overlay ──
    fig1 = Figure(size=(1200, 700))
    ax1 = Axis(fig1[1, 1],
        xlabel="Time (s)", ylabel="ERG (a.u.)",
        title="Data vs Fitted Traces")
    for i in 1:nI
        lines!(ax1, real_t[i], real_traces[i],
            color=(colors[i], 0.6), linewidth=2,
            label="data I=$(round(intensities[i], sigdigits=3))")
        lines!(ax1, result.t_sim, result.sim_traces[i],
            color=colors[i], linewidth=2, linestyle=:dash)
    end
    vlines!(ax1, [time_window[1], time_window[2]], color=:gray, linestyle=:dot, linewidth=1)
    axislegend(ax1, position=:rb)

    # ── Fig 2: Residuals in fit window ──
    fig2 = Figure(size=(1200, 700))
    ax2 = Axis(fig2[1, 1],
        xlabel="Time (s)", ylabel="Residual",
        title="Residuals in Fit Window [$(time_window[1])s, $(time_window[2])s]")
    for i in 1:nI
        # Compute residuals at real time points within window
        rt = real_t[i]
        ry = real_traces[i]
        win_mask = time_window[1] .<= rt .<= time_window[2]
        win_t = rt[win_mask]
        win_y = ry[win_mask]
        win_sim = [result.sim_traces[i][clamp(searchsortedlast(result.t_sim, t), 1, length(result.t_sim))] for t in win_t]
        residuals = win_y .- win_sim
        lines!(ax2, win_t, residuals, color=colors[i], linewidth=1.5,
            label="I=$(round(intensities[i], sigdigits=3))")
    end
    hlines!(ax2, [0.0], color=:black, linewidth=0.5)
    axislegend(ax2, position=:rb)

    # ── Fig 3: Parameter estimates with CI ──
    unc = result.uncertainty
    fig3 = Figure(size=(max(1200, nrow(unc) * 30), 700))
    ax3 = Axis(fig3[1, 1],
        xlabel="Parameter", ylabel="Estimate",
        title="Parameter Estimates ± 95% CI",
        xticklabelrotation=π/4)

    x = 1:nrow(unc)
    scatter!(ax3, x, unc.estimate, color=:black, markersize=6)
    for i in x
        lo = unc.ci95_lower[i]
        hi = unc.ci95_upper[i]
        if isfinite(lo) && isfinite(hi)
            lines!(ax3, [i, i], [lo, hi], color=:steelblue, linewidth=2)
        end
    end
    ax3.xticks = (collect(x), unc.parameter)

    # ── Fig 4: Correlated parameters ──
    corr_rows = collect(filter(r -> !isempty(r.correlated_with), eachrow(unc)))
    p4 = ""
    if !isempty(corr_rows)
        fig4 = Figure(size=(800, 400))
        ax4 = Axis(fig4[1, 1], title="Correlated Parameter Pairs (|r| > 0.7)")
        hidespines!(ax4)
        hidedecorations!(ax4)

        text_lines = ["$(r.parameter) ↔ $(r.correlated_with)" for r in corr_rows]
        text!(ax4, 0.05, 0.95, text=join(text_lines, "\n"),
            fontsize=14, align=(:left, :top))
        p4 = joinpath(outdir, "fit_correlations.png")
        save(p4, fig4)
    end

    p1 = joinpath(outdir, "fit_traces.png")
    p2 = joinpath(outdir, "fit_residuals.png")
    p3 = joinpath(outdir, "fit_parameters.png")
    save(p1, fig1)
    save(p2, fig2)
    save(p3, fig3)

    return (traces=p1, residuals=p2, params=p3, correlations=p4)
end

"""
    save_fit_datasheet(result, path)

Save the uncertainty/parameter estimates DataFrame to CSV.
"""
function save_fit_datasheet(result::NamedTuple, path::AbstractString)
    mkpath(dirname(path))
    CSV.write(path, result.uncertainty)
    return path
end
