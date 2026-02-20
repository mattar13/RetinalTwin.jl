hill_ir(I, A, K, n) = A * (I^n) / (K^n + I^n)

function fit_hill_ir(I::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    mask = [isfinite(I[i]) && isfinite(y[i]) && I[i] > 0 && y[i] >= 0 for i in eachindex(I)]
    x = Float64[I[i] for i in eachindex(I) if mask[i]]
    z = Float64[y[i] for i in eachindex(y) if mask[i]]
    if length(x) < 3
        return (A=NaN, K=NaN, n=NaN, sse=NaN, ok=false)
    end

    A_min = max(maximum(z), 1e-12)
    A_max = max(2.0 * A_min, A_min + 1e-6)
    A_grid = exp10.(range(log10(A_min), log10(A_max); length=40))
    K_grid = exp10.(range(log10(minimum(x)), log10(maximum(x)); length=60))
    n_grid = range(0.2, 4.0; length=60)

    best_A, best_K, best_n = NaN, NaN, NaN
    best_sse = Inf
    for A in A_grid, K in K_grid, n in n_grid
        pred = [hill_ir(xi, A, K, n) for xi in x]
        sse = sum((z .- pred) .^ 2)
        if sse < best_sse
            best_sse = sse
            best_A, best_K, best_n = A, K, n
        end
    end
    return (A=best_A, K=best_K, n=best_n, sse=best_sse, ok=isfinite(best_sse))
end
