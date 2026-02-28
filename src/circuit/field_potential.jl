using CSV

"""
    default_depth_csv_path()

Default ERG channel-depth CSV.
"""
default_depth_csv_path() = normpath(joinpath(@__DIR__, "..", "parameters", "erg_depth_map.csv"))

"""
    load_erg_depth_map(csv_path=default_depth_csv_path())

Load rows from `erg_depth_map.csv`.
Expected columns: `cell_type,current,z` and optional `weight`.
"""
function load_erg_depth_map(csv_path::AbstractString=default_depth_csv_path())
    rows = CSV.File(csv_path)
    out = NamedTuple[]

    for row in rows
        cell_type = hasproperty(row, :cell_type) ? String(getproperty(row, :cell_type)) :
                    (hasproperty(row, :CellType) ? String(getproperty(row, :CellType)) :
                     error("Missing `cell_type` column in $csv_path"))
        current = hasproperty(row, :current) ? String(getproperty(row, :current)) :
                  (hasproperty(row, :Current) ? String(getproperty(row, :Current)) :
                   error("Missing `current` column in $csv_path"))
        z = hasproperty(row, :z) ? Float64(getproperty(row, :z)) :
            (hasproperty(row, :Z) ? Float64(getproperty(row, :Z)) :
             error("Missing `z` column in $csv_path"))
        weight = hasproperty(row, :weight) ? Float64(getproperty(row, :weight)) :
                 (hasproperty(row, :Weight) ? Float64(getproperty(row, :Weight)) : 1.0)

        push!(out, (cell_type=strip(cell_type), current=Symbol(strip(current)), z=z, weight=weight))
    end
    return out
end

# Back-compat name.
load_depth_map(csv_path::AbstractString=default_depth_csv_path()) = load_erg_depth_map(csv_path)
load_depth_scales(csv_path::AbstractString=default_depth_csv_path()) = load_erg_depth_map(csv_path)

@inline function _cell_type_label(cell_type::Symbol)
    if cell_type == :PC
        return "PHOTO"
    elseif cell_type == :HC
        return "HC"
    elseif cell_type == :ONBC
        return "ONBC"
    elseif cell_type == :OFFBC
        return "OFFBC"
    elseif cell_type == :A2
        return "A2"
    elseif cell_type == :GC
        return "GC"
    elseif cell_type == :MG
        return "MULLER"
    elseif cell_type == :RPE
        return "RPE"
    elseif cell_type == :GABA
        return "GABA"
    elseif cell_type == :DA
        return "DA"
    else
        return String(cell_type)
    end
end

@inline function _depth_scale(depth_rows, cell_type::Symbol, current::Symbol)
    target_cell = _cell_type_label(cell_type)
    for r in depth_rows
        if r.cell_type == target_cell && r.current == current
            return r.weight
        end
    end
    return 1.0
end

"""
    compute_field_potential(model, sol; depth_csv=default_depth_csv_path(), params=load_all_params())

Post-simulation transretinal field potential by summing every per-channel
current contribution scaled by channel depth from `erg_depth_map.csv`.
"""
function compute_field_potential(model::RetinalColumnModel, params::NamedTuple, sol;
    depth_csv::AbstractString=default_depth_csv_path(),
    dt = 0.5
)
    depth_rows = load_erg_depth_map(depth_csv)
    t = sol.t[1]:dt:sol.t[end]
    nt = length(t)
    field_potential = zeros(nt)

    ordered = sort!(collect(values(model.cells)), by=cell -> cell.offset)

    for (i, ti) in enumerate(t)
        ui = sol(ti)
        total = 0.0

        for cell in ordered
            uc = uview(ui, cell)
            V = uc[cell.outidx.V]

            if cell.cell_type == :PC
                p = params.PHOTO
                G = uc[PC_IC_MAP.G]
                HO_sum = uc[PC_IC_MAP.HO1] + uc[PC_IC_MAP.HO2] + uc[PC_IC_MAP.HO3]
                mKv = uc[PC_IC_MAP.mKv]
                hKv = uc[PC_IC_MAP.hKv]
                mCa = uc[PC_IC_MAP.mCa]
                mKCa = uc[PC_IC_MAP.mKCa]
                Ca_s = uc[PC_IC_MAP.Ca_s]

                total += _depth_scale(depth_rows, :PC, :I_photo) * photoreceptor_I_photo(V, G, p)
                total += _depth_scale(depth_rows, :PC, :I_leak) * photoreceptor_I_leak(V, p)
                total += _depth_scale(depth_rows, :PC, :I_h) * photoreceptor_I_h(V, HO_sum, p)
                total += _depth_scale(depth_rows, :PC, :I_kv) * photoreceptor_I_kv(V, mKv, hKv, p)
                total += _depth_scale(depth_rows, :PC, :I_ca) * photoreceptor_I_ca(V, mCa, Ca_s, p)
                total += _depth_scale(depth_rows, :PC, :I_kca) * photoreceptor_I_kca(V, mKCa, Ca_s, p)
                total += _depth_scale(depth_rows, :PC, :I_cl) * photoreceptor_I_cl(V, Ca_s, p)
                total += _depth_scale(depth_rows, :PC, :I_ex) * photoreceptor_I_ex(V, Ca_s, p)
                total += _depth_scale(depth_rows, :PC, :I_ex2) * photoreceptor_I_ex2(Ca_s, p)

            elseif cell.cell_type == :HC
                p = params.HC
                c = uc[HC_IC_MAP.c]
                inputs = get(model.connections, cell.name, Tuple{Symbol,Symbol,Float64}[])
                glu_in = [get_out(ui, model.cells[pre], key) for (pre, key, _) in inputs if key == :Glu]
                w_glu_in = [w for (_, key, w) in inputs if key == :Glu]
                s_inf = spatial_synaptic(glu_in, w_glu_in, p, :hill, :K_Glu, :n_Glu)
                mCa = gate_inf(V, p.Vm_half, p.km_slope)

                total += _depth_scale(depth_rows, :HC, :I_leak) * horizontal_I_leak(V, p)
                total += _depth_scale(depth_rows, :HC, :I_exc) * horizontal_I_exc(V, s_inf, p)
                total += _depth_scale(depth_rows, :HC, :I_cal) * horizontal_I_cal(V, mCa, p)
                total += _depth_scale(depth_rows, :HC, :I_kir) * horizontal_I_kir(V, p)
                total += _depth_scale(depth_rows, :HC, :I_bk) * horizontal_I_bk(V, c, p)

            elseif cell.cell_type == :ONBC
                p = params.ONBC
                n = uc[ONBC_IC_MAP.n]
                h = uc[ONBC_IC_MAP.h]
                c = uc[ONBC_IC_MAP.c]
                S = uc[ONBC_IC_MAP.S]
                m = gate_inf(V, p.Vm_half, p.km_slope)

                total += _depth_scale(depth_rows, :ONBC, :I_leak) * on_bipolar_I_leak(V, p)
                total += _depth_scale(depth_rows, :ONBC, :I_trpm1) * on_bipolar_I_trpm1(V, S, p)
                total += _depth_scale(depth_rows, :ONBC, :I_kv) * on_bipolar_I_kv(V, n, p)
                total += _depth_scale(depth_rows, :ONBC, :I_h) * on_bipolar_I_h(V, h, p)
                total += _depth_scale(depth_rows, :ONBC, :I_cal) * on_bipolar_I_cal(V, m, p)
                total += _depth_scale(depth_rows, :ONBC, :I_kca) * on_bipolar_I_kca(V, c, p)

            elseif cell.cell_type == :OFFBC
                p = params.OFFBC
                n = uc[OFFBC_IC_MAP.n]
                h = uc[OFFBC_IC_MAP.h]
                c = uc[OFFBC_IC_MAP.c]
                A = uc[OFFBC_IC_MAP.A]
                D = uc[OFFBC_IC_MAP.D]
                m = gate_inf(V, p.Vm_half, p.km_slope)

                total += _depth_scale(depth_rows, :OFFBC, :I_leak) * off_bipolar_I_leak(V, p)
                total += _depth_scale(depth_rows, :OFFBC, :I_iglu) * off_bipolar_I_iglu(V, A, D, p)
                total += _depth_scale(depth_rows, :OFFBC, :I_kv) * off_bipolar_I_kv(V, n, p)
                total += _depth_scale(depth_rows, :OFFBC, :I_h) * off_bipolar_I_h(V, h, p)
                total += _depth_scale(depth_rows, :OFFBC, :I_cal) * off_bipolar_I_cal(V, m, p)
                total += _depth_scale(depth_rows, :OFFBC, :I_kca) * off_bipolar_I_kca(V, c, p)

            elseif cell.cell_type == :A2
                p = params.A2
                n = uc[A2_IC_MAP.n]
                h = uc[A2_IC_MAP.h]
                c = uc[A2_IC_MAP.c]
                A = uc[A2_IC_MAP.A]
                D = uc[A2_IC_MAP.D]
                m = gate_inf(V, p.Vm_half, p.km_slope)

                total += _depth_scale(depth_rows, :A2, :I_leak) * a2_amacrine_I_leak(V, p)
                total += _depth_scale(depth_rows, :A2, :I_iglu) * a2_amacrine_I_iglu(V, A, D, p)
                total += _depth_scale(depth_rows, :A2, :I_kv) * a2_amacrine_I_kv(V, n, p)
                total += _depth_scale(depth_rows, :A2, :I_h) * a2_amacrine_I_h(V, h, p)
                total += _depth_scale(depth_rows, :A2, :I_cal) * a2_amacrine_I_cal(V, m, p)
                total += _depth_scale(depth_rows, :A2, :I_kca) * a2_amacrine_I_kca(V, c, p)

            elseif cell.cell_type == :GC
                p = params.GC
                m = uc[GC_IC_MAP.m]
                h = uc[GC_IC_MAP.h]
                n = uc[GC_IC_MAP.n]
                sE = uc[GC_IC_MAP.sE]
                sI = uc[GC_IC_MAP.sI]

                total += _depth_scale(depth_rows, :GC, :I_leak) * ganglion_I_leak(V, p)
                total += _depth_scale(depth_rows, :GC, :I_na) * ganglion_I_na(V, m, h, p)
                total += _depth_scale(depth_rows, :GC, :I_k) * ganglion_I_k(V, n, p)
                total += _depth_scale(depth_rows, :GC, :I_exc) * ganglion_I_exc(V, sE, p)
                total += _depth_scale(depth_rows, :GC, :I_inh) * ganglion_I_inh(V, sI, p)

            elseif cell.cell_type == :MG
                p = params.MULLER
                K_o_end = uc[MG_IC_MAP.K_o_end]
                K_o_stalk = uc[MG_IC_MAP.K_o_stalk]
                Glu_o = uc[MG_IC_MAP.Glu_o]

                total += _depth_scale(depth_rows, :MG, :I_kir_end) * muller_I_kir_end(V, K_o_end, p)
                total += _depth_scale(depth_rows, :MG, :I_kir_stalk) * muller_I_kir_stalk(V, K_o_stalk, p)
                total += _depth_scale(depth_rows, :MG, :I_leak) * muller_I_leak(V, p)
                total += _depth_scale(depth_rows, :MG, :I_eaat) * muller_I_eaat(Glu_o, p)
            end
        end

        field_potential[i] = total
    end
    field_potential = field_potential .- field_potential[1] #Calculate DC offset
    return t, field_potential
end
