using Random
using DataFrames
using CSV
using CairoMakie
using DSP
using FFTW
import DifferentialEquations

struct ERGDataSet
    intensities::Vector{Float64}
    t::Vector{Float64}
    traces::Matrix{Float64}
    σ::Union{Float64, Matrix{Float64}}
    species::Symbol
    adaptation::Symbol
    flash_duration::Float64
    pre_flash_time::Float64
end

struct ERGFeatures
    a_wave_amplitude::Vector{Float64}
    a_wave_implicit_time::Vector{Float64}
    b_wave_amplitude::Vector{Float64}
    b_wave_implicit_time::Vector{Float64}
    op_sum_amplitude::Vector{Float64}
    op_dominant_freq::Vector{Float64}
end

struct ParameterTarget
    block::Symbol
    name::Symbol
    lower::Float64
    upper::Float64
end

struct StageDefinition
    name::Symbol
    targets::Vector{ParameterTarget}
    time_window::Tuple{Float64, Float64}
    loss_mode::Symbol
    n_candidates::Int
    n_rounds::Int
end

struct StageResult
    name::Symbol
    loss::Float64
    best_params::NamedTuple
    history::Vector{NamedTuple}
end

struct StagedFitResult
    params::NamedTuple
    stages::Vector{StageResult}
    features_data::ERGFeatures
    features_fit::ERGFeatures
    traces_fit::Matrix{Float64}
    residuals::Matrix{Float64}
    uncertainty::DataFrame
end

function load_erg_data(filepath; flash_duration=10.0, pre_flash_time=200.0, species=:mouse, adaptation=:dark_adapted)
    tbl = CSV.File(filepath)
    names = propertynames(tbl)
    length(names) >= 2 || error("ERG CSV must have time + at least one trace column")

    tname = names[1]
    t = Float64[]
    traces_cols = [Float64[] for _ in 2:length(names)]

    for row in tbl
        push!(t, Float64(getproperty(row, tname)))
        for (j, nm) in enumerate(names[2:end])
            push!(traces_cols[j], Float64(getproperty(row, nm)))
        end
    end

    traces = hcat(traces_cols...)
    intensities = [parse(Float64, replace(String(nm), "intensity_" => "")) for nm in names[2:end]]

    return ERGDataSet(intensities, t, traces, 1.0, species, adaptation, flash_duration, pre_flash_time)
end

function preprocess_erg(data::ERGDataSet; lowpass_hz::Union{Nothing,Float64}=nothing)
    traces = copy(data.traces)
    baseline_idx = findall(<(0.0), data.t)

    for i in axes(traces, 2)
        if !isempty(baseline_idx)
            traces[:, i] .-= mean(view(traces, baseline_idx, i))
        end

        if lowpass_hz !== nothing && length(data.t) > 2
            dt = data.t[2] - data.t[1]
            fs = 1000.0 / dt
            fil = DSP.digitalfilter(DSP.Lowpass(lowpass_hz; fs=fs), DSP.Butterworth(4))
            traces[:, i] .= DSP.filtfilt(fil, traces[:, i])
        end
    end

    out = ERGDataSet(data.intensities, data.t, traces, data.σ, data.species, data.adaptation, data.flash_duration, data.pre_flash_time)
    return out, extract_erg_features(out)
end

function _find_window(t::AbstractVector, t0::Float64, t1::Float64)
    return findall(tt -> t0 <= tt <= t1, t)
end

function _dominant_op_freq(trace::AbstractVector, t::AbstractVector)
    length(trace) < 8 && return NaN
    dt = t[2] - t[1]
    fs = 1000.0 / dt
    fil = DSP.digitalfilter(DSP.Bandpass(75.0, 300.0; fs=fs), DSP.Butterworth(4))
    op = DSP.filtfilt(fil, trace)
    spec = abs.(rfft(op))
    freqs = range(0.0, fs / 2; length=length(spec))
    idx = findall(f -> 75.0 <= f <= 300.0, freqs)
    isempty(idx) && return NaN
    k = idx[argmax(spec[idx])]
    return freqs[k]
end

function extract_erg_features(data::ERGDataSet)
    nI = length(data.intensities)
    a_amp = fill(NaN, nI)
    a_time = fill(NaN, nI)
    b_amp = fill(NaN, nI)
    b_time = fill(NaN, nI)
    op_sum = fill(NaN, nI)
    op_freq = fill(NaN, nI)

    for i in 1:nI
        tr = view(data.traces, :, i)

        a_idx = _find_window(data.t, 0.0, 30.0)
        if !isempty(a_idx)
            amin, rel = findmin(tr[a_idx])
            a_amp[i] = amin
            a_time[i] = data.t[a_idx[rel]]
        end

        b_idx = _find_window(data.t, 20.0, 120.0)
        if !isempty(b_idx)
            bmax, rel = findmax(tr[b_idx])
            b_amp[i] = bmax - a_amp[i]
            b_time[i] = data.t[b_idx[rel]]
        end

        op_idx = _find_window(data.t, 15.0, 80.0)
        if !isempty(op_idx)
            op_seg = tr[op_idx]
            op_sum[i] = sum(abs, op_seg .- mean(op_seg)) / length(op_seg)
            op_freq[i] = _dominant_op_freq(collect(op_seg), data.t[op_idx])
        end
    end

    return ERGFeatures(a_amp, a_time, b_amp, b_time, op_sum, op_freq)
end

function _set_nested(params, block::Symbol, name::Symbol, value::Float64)
    block_nt = getproperty(params, block)
    updated_block = merge(block_nt, (name => value,))
    return merge(params, (block => updated_block,))
end

function _dark_adapt_state(model, u0, params; tspan_dark=(0.0, 1500.0), abstol=1e-6, reltol=1e-4)
    stim_dark = make_uniform_flash_stimulus(photon_flux=0.0)
    prob = DifferentialEquations.ODEProblem(model, u0, tspan_dark, (params, stim_dark))
    sol = DifferentialEquations.solve(
        prob,
        DifferentialEquations.Rodas5();
        save_everystep=false,
        save_end=true,
        abstol=abstol,
        reltol=reltol,
    )
    return sol.u[end]
end

function simulate_erg_dataset(model, u0, params, data::ERGDataSet; abstol=1e-6, reltol=1e-4)
    nT = length(data.t)
    nI = length(data.intensities)
    pred = zeros(nT, nI)

    u0_dark = _dark_adapt_state(model, u0, params; abstol=abstol, reltol=reltol)
    for (i, intensity) in enumerate(data.intensities)
        stim = make_uniform_flash_stimulus(
            stim_start=0.0,
            stim_end=data.flash_duration,
            photon_flux=intensity,
        )
        prob = DifferentialEquations.ODEProblem(model, u0_dark, (minimum(data.t), maximum(data.t)), (params, stim))
        sol = DifferentialEquations.solve(prob, DifferentialEquations.Rodas5(); tstops=[0.0, data.flash_duration], abstol=abstol, reltol=reltol)
        tsim, erg = compute_field_potential(model, params, sol; dt=(data.t[2] - data.t[1]))
        pred[:, i] .= [erg[clamp(searchsortedlast(tsim, tt), 1, length(tsim))] for tt in data.t]
    end

    return pred
end

function _stage_feature_loss(stage::StageDefinition, data_features::ERGFeatures, fit_features::ERGFeatures)
    if stage.loss_mode == :awave
        return mean((data_features.a_wave_amplitude .- fit_features.a_wave_amplitude) .^ 2) +
               0.2 * mean((data_features.a_wave_implicit_time .- fit_features.a_wave_implicit_time) .^ 2)
    elseif stage.loss_mode == :bwave
        return mean((data_features.b_wave_amplitude .- fit_features.b_wave_amplitude) .^ 2) +
               0.2 * mean((data_features.b_wave_implicit_time .- fit_features.b_wave_implicit_time) .^ 2)
    elseif stage.loss_mode == :ops
        return mean((data_features.op_sum_amplitude .- fit_features.op_sum_amplitude) .^ 2)
    else
        return 0.0
    end
end

function _stage_loss(stage::StageDefinition, data::ERGDataSet, traces_pred::Matrix{Float64}, data_features::ERGFeatures, fit_features::ERGFeatures)
    idx = _find_window(data.t, stage.time_window[1], stage.time_window[2])
    isempty(idx) && return Inf
    residuals = view(data.traces, idx, :) .- view(traces_pred, idx, :)
    time_sse = mean(residuals .^ 2)
    return time_sse + _stage_feature_loss(stage, data_features, fit_features)
end

function _sample_candidate(rng::AbstractRNG, params::NamedTuple, targets::Vector{ParameterTarget}, radius::Float64)
    out = params
    for t in targets
        base = getproperty(getproperty(out, t.block), t.name)
        span = (t.upper - t.lower) * radius
        lo = max(t.lower, base - span)
        hi = min(t.upper, base + span)
        trial = rand(rng) * (hi - lo) + lo
        out = _set_nested(out, t.block, t.name, trial)
    end
    return out
end

function _fit_stage(
    model,
    u0,
    params0::NamedTuple,
    data::ERGDataSet,
    data_features::ERGFeatures,
    stage::StageDefinition;
    rng=Random.default_rng(),
)
    best_params = params0
    best_traces = simulate_erg_dataset(model, u0, best_params, data)
    best_features = extract_erg_features(ERGDataSet(data.intensities, data.t, best_traces, data.σ, data.species, data.adaptation, data.flash_duration, data.pre_flash_time))
    best_loss = _stage_loss(stage, data, best_traces, data_features, best_features)

    history = NamedTuple[(round=0, candidate=0, loss=best_loss)]

    radius = 0.35
    for round in 1:stage.n_rounds
        improved = false
        for cand in 1:stage.n_candidates
            trial_params = _sample_candidate(rng, best_params, stage.targets, radius)
            trial_traces = simulate_erg_dataset(model, u0, trial_params, data)
            trial_features = extract_erg_features(ERGDataSet(data.intensities, data.t, trial_traces, data.σ, data.species, data.adaptation, data.flash_duration, data.pre_flash_time))
            trial_loss = _stage_loss(stage, data, trial_traces, data_features, trial_features)
            push!(history, (round=round, candidate=cand, loss=trial_loss))

            if trial_loss < best_loss
                best_loss = trial_loss
                best_params = trial_params
                best_traces = trial_traces
                best_features = trial_features
                improved = true
            end
        end
        radius *= improved ? 0.9 : 0.65
    end

    return StageResult(stage.name, best_loss, best_params, history), best_traces, best_features
end

function _parameter_bounds_from_csv(block::Symbol, name::Symbol, base::Float64)
    spec = get_param_spec(block, name; csv_path=default_param_csv_path())
    if spec !== nothing
        l = spec.lower
        u = spec.upper
        if l == u
            eps = max(abs(base) * 0.1, 1e-6)
            return (l - eps, u + eps)
        end
        return (Float64(l), Float64(u))
    end
    return (0.3 * base, 3.0 * max(base, 1e-4))
end

function _is_param_fixed(block::Symbol, name::Symbol)
    spec = get_param_spec(block, name; csv_path=default_param_csv_path())
    return spec === nothing ? false : spec.fixed
end

function make_target(params::NamedTuple, block::Symbol, name::Symbol)
    base = Float64(getproperty(getproperty(params, block), name))
    lo, hi = _parameter_bounds_from_csv(block, name, base)
    return ParameterTarget(block, name, lo, hi)
end

function default_stages(params::NamedTuple; mode::Symbol=:efficient)
    fine = mode == :accurate
    mult = fine ? 2 : 1

    stageA_targets = [
        make_target(params, :PHOTO, :gCa),
        make_target(params, :PHOTO, :gKV),
        make_target(params, :PHOTO, :gH),
        make_target(params, :PHOTO, :lambda),
        make_target(params, :PHOTO, :tau_Glu),
    ]
    stageB_targets = [
        make_target(params, :ONBC, :g_TRPM1),
        make_target(params, :ONBC, :g_Kv),
        make_target(params, :ONBC, :g_CaL),
        make_target(params, :ONBC, :tau_S),
        make_target(params, :ONBC, :K_Glu),
    ]
    stageC_targets = [
        make_target(params, :A2, :g_iGluR),
        make_target(params, :A2, :g_Kv),
        make_target(params, :A2, :g_CaL),
        make_target(params, :A2, :tau_A),
        make_target(params, :A2, :tau_d),
    ]
    stageD_targets = [
        make_target(params, :MULLER, :g_Kir_end),
        make_target(params, :MULLER, :g_Kir_stalk),
        make_target(params, :MULLER, :alpha_K),
        make_target(params, :MULLER, :tau_K_diffusion),
    ]
    stageA_targets = filter(t -> !_is_param_fixed(t.block, t.name), stageA_targets)
    stageB_targets = filter(t -> !_is_param_fixed(t.block, t.name), stageB_targets)
    stageC_targets = filter(t -> !_is_param_fixed(t.block, t.name), stageC_targets)
    stageD_targets = filter(t -> !_is_param_fixed(t.block, t.name), stageD_targets)

    return [
        StageDefinition(:A_awave, stageA_targets, (0.0, 40.0), :awave, 6 * mult, 4 * mult),
        StageDefinition(:B_bwave, stageB_targets, (20.0, 150.0), :bwave, 6 * mult, 4 * mult),
        StageDefinition(:C_ops, stageC_targets, (10.0, 90.0), :ops, 5 * mult, 3 * mult),
        StageDefinition(:D_slow, stageD_targets, (80.0, 350.0), :trace, 5 * mult, 3 * mult),
    ]
end

function _estimate_uncertainty(
    model,
    u0,
    params::NamedTuple,
    data::ERGDataSet,
    data_features::ERGFeatures,
    stages::Vector{StageDefinition};
    curvature_step=0.03,
)
    traces = simulate_erg_dataset(model, u0, params, data)
    features = extract_erg_features(ERGDataSet(data.intensities, data.t, traces, data.σ, data.species, data.adaptation, data.flash_duration, data.pre_flash_time))
    nobs = length(data.traces)
    residual = data.traces .- traces
    σ2 = sum(residual .^ 2) / max(nobs - 1, 1)

    rows = NamedTuple[]
    for stage in stages
        for tgt in stage.targets
            p0 = Float64(getproperty(getproperty(params, tgt.block), tgt.name))
            δ = max(abs(p0) * curvature_step, 1e-4)
            pm = clamp(p0 - δ, tgt.lower, tgt.upper)
            pp = clamp(p0 + δ, tgt.lower, tgt.upper)

            params_m = _set_nested(params, tgt.block, tgt.name, pm)
            params_p = _set_nested(params, tgt.block, tgt.name, pp)

            tr_m = simulate_erg_dataset(model, u0, params_m, data)
            tr_p = simulate_erg_dataset(model, u0, params_p, data)
            fe_m = extract_erg_features(ERGDataSet(data.intensities, data.t, tr_m, data.σ, data.species, data.adaptation, data.flash_duration, data.pre_flash_time))
            fe_p = extract_erg_features(ERGDataSet(data.intensities, data.t, tr_p, data.σ, data.species, data.adaptation, data.flash_duration, data.pre_flash_time))

            l0 = _stage_loss(stage, data, traces, data_features, features)
            lm = _stage_loss(stage, data, tr_m, data_features, fe_m)
            lp = _stage_loss(stage, data, tr_p, data_features, fe_p)

            h = (lp - 2 * l0 + lm) / ((pp - p0)^2)
            var = h > 0 ? (2 * σ2 / h) : NaN
            se = isfinite(var) && var > 0 ? sqrt(var) : NaN
            ci_lo = isfinite(se) ? p0 - 1.96 * se : NaN
            ci_hi = isfinite(se) ? p0 + 1.96 * se : NaN
            certainty = isfinite(se) && p0 != 0 ? 1 / (1 + abs(se / p0)) : 0.0

            push!(rows, (
                stage=String(stage.name),
                block=String(tgt.block),
                parameter=String(tgt.name),
                estimate=p0,
                std_error=se,
                ci95_lower=ci_lo,
                ci95_upper=ci_hi,
                certainty=certainty,
                curvature=h,
            ))
        end
    end

    return DataFrame(rows)
end

function fit_retinal_twin_staged(model, u0, data::ERGDataSet; params0=load_all_params(), mode::Symbol=:efficient, rng=Random.MersenneTwister(4))
    pdata, fdata = preprocess_erg(data)
    stages = default_stages(params0; mode=mode)

    current = params0
    stage_results = StageResult[]
    current_traces = zeros(size(pdata.traces))
    current_features = fdata

    for stage in stages
        result, traces, ffit = _fit_stage(model, u0, current, pdata, fdata, stage; rng=rng)
        push!(stage_results, result)
        current = result.best_params
        current_traces = traces
        current_features = ffit
    end

    residuals = pdata.traces .- current_traces
    unc = _estimate_uncertainty(model, u0, current, pdata, fdata, stages)

    return StagedFitResult(current, stage_results, fdata, current_features, current_traces, residuals, unc)
end

function save_fit_datasheet(result::StagedFitResult, out_csv::AbstractString)
    mkpath(dirname(out_csv))
    CSV.write(out_csv, result.uncertainty)
    return out_csv
end

function plot_fit_diagnostics(data::ERGDataSet, result::StagedFitResult; outdir::AbstractString)
    mkpath(outdir)
    nI = length(data.intensities)
    palette = cgrad(:viridis, nI, categorical=true)

    fig1 = Figure(size=(1200, 700))
    ax1 = Axis(fig1[1, 1], xlabel="Time (ms)", ylabel="ERG", title="Data vs fitted traces")
    for i in 1:nI
        lines!(ax1, data.t, data.traces[:, i], color=(palette[i], 0.6), linewidth=2, label="data I=$(round(data.intensities[i], sigdigits=3))")
        lines!(ax1, data.t, result.traces_fit[:, i], color=palette[i], linestyle=:dash, linewidth=2)
    end
    axislegend(ax1, position=:rb)

    fig2 = Figure(size=(1200, 700))
    ax2 = Axis(fig2[1, 1], xlabel="Time (ms)", ylabel="Intensity index", title="Residual heatmap")
    hm = result.residuals'
    heatmap!(ax2, data.t, 1:nI, hm, colormap=:balance)

    fig3 = Figure(size=(1200, 700))
    ax3 = Axis(fig3[1, 1], xlabel="Parameter", ylabel="Estimate", title="Parameter estimates ±95% CI")
    tbl = result.uncertainty
    x = 1:nrow(tbl)
    scatter!(ax3, x, tbl.estimate, color=:black)
    for i in x
        lines!(ax3, [i, i], [tbl.ci95_lower[i], tbl.ci95_upper[i]], color=:steelblue, linewidth=2)
    end
    ax3.xticks = (x, tbl.parameter)

    p1 = joinpath(outdir, "fit_traces.png")
    p2 = joinpath(outdir, "fit_residuals.png")
    p3 = joinpath(outdir, "fit_parameters.png")
    save(p1, fig1)
    save(p2, fig2)
    save(p3, fig3)

    return (traces=p1, residuals=p2, params=p3)
end

