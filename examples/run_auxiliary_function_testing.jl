using Revise
using RetinalTwin
using CairoMakie

# -----------------------------------------------------------------------------
#%% Load parameters
# -----------------------------------------------------------------------------
load_param_fn = joinpath(@__DIR__, "checkpoints", "fit_checkpoint_nelder_mead_best.csv")
load_param_fn = "C:\\Users\\mtarc\\Julia\\dev\\RetinalTwin.jl\\src\\parameters\\retinal_params.csv"

params_dict = load_all_params(csv_path = load_param_fn, editable = true)
params = dict_to_namedtuple(params_dict)

# Voltage sweep range (mV)
V = -80.0:0.5:20.0

# Output directory for saved figures
savedir = joinpath(@__DIR__, "plots", "auxiliary_functions")
mkpath(savedir)

# Helper: place an axis + side legend in a grid position
function axis_with_legend(fig_pos; kwargs...)
    gl = fig_pos = GridLayout()
    ax = Axis(gl[1, 1]; kwargs...)
    return gl, ax
end

function add_side_legend!(gl, ax)
    Legend(gl[1, 2], ax, framevisible=false, labelsize=11, padding=(0,0,0,0))
end

# =============================================================================
#%% 1) Photoreceptor auxiliary functions
# =============================================================================
let p = params.PHOTO
    fig = Figure(size=(1600, 1200))
    Label(fig[0, 1:4], "Photoreceptor (PC) Auxiliary Functions", fontsize=20, font=:bold)

    # --- Row 1: Ion channel gating variables ---
    mKv_inf = [RetinalTwin.αmKv(v) / (RetinalTwin.αmKv(v) + RetinalTwin.βmKv(v)) for v in V]
    hKv_inf = [RetinalTwin.αhKv(v) / (RetinalTwin.αhKv(v) + RetinalTwin.βhKv(v)) for v in V]
    gl1 = fig[1, 1] = GridLayout()
    ax1 = Axis(gl1[1, 1], title="Kv Gating", xlabel="V (mV)", ylabel="Activation")
    lines!(ax1, collect(V), mKv_inf, label="mKv_inf", linewidth=2)
    lines!(ax1, collect(V), hKv_inf, label="hKv_inf", linewidth=2)
    Legend(gl1[1, 2], ax1, framevisible=false, labelsize=11)

    mCa_inf = [RetinalTwin.αmCa(v) / (RetinalTwin.αmCa(v) + RetinalTwin.βmCa(v)) for v in V]
    hCa_vals = [RetinalTwin.hCa(v) for v in V]
    gl2 = fig[1, 2] = GridLayout()
    ax2 = Axis(gl2[1, 1], title="Ca Gating", xlabel="V (mV)", ylabel="Activation")
    lines!(ax2, collect(V), mCa_inf, label="mCa_inf", linewidth=2)
    lines!(ax2, collect(V), hCa_vals, label="hCa", linewidth=2)
    Legend(gl2[1, 2], ax2, framevisible=false, labelsize=11)

    mKCa_inf = [RetinalTwin.αmKCa(v) / (RetinalTwin.αmKCa(v) + RetinalTwin.βmKCa(v)) for v in V]
    ax3 = Axis(fig[1, 3], title="KCa Gating", xlabel="V (mV)", ylabel="Activation")
    lines!(ax3, collect(V), mKCa_inf, color=:blue, linewidth=2)

    Ca_range = 0.0:0.01:2.0
    mKCas_vals = [RetinalTwin.mKCas(c) for c in Ca_range]
    ax3b = Axis(fig[1, 4], title="KCa Ca-Dependence", xlabel="[Ca]", ylabel="mKCas")
    lines!(ax3b, collect(Ca_range), mKCas_vals, color=:red, linewidth=2)

    # --- Row 2: h-current, Cl, phototransduction, exchanger ---
    h_open_frac = zeros(length(V))
    for (i, v) in enumerate(V)
        α = RetinalTwin.αh(v)
        β = RetinalTwin.βh(v)
        h_open_frac[i] = α / (α + β)
    end
    ax4 = Axis(fig[2, 1], title="h-Current (HCN)", xlabel="V (mV)", ylabel="h_inf (open)")
    lines!(ax4, collect(V), h_open_frac, color=:darkgreen, linewidth=2)

    mCl_vals = [RetinalTwin.mCl(c) for c in Ca_range]
    ax5 = Axis(fig[2, 2], title="Cl Channel Ca-Dependence", xlabel="[Ca]", ylabel="mCl")
    lines!(ax5, collect(Ca_range), mCl_vals, color=:teal, linewidth=2)

    G_range = 0.0:0.1:20.0
    J_vals = [RetinalTwin.J∞(g, 10.0) for g in G_range]
    ax6 = Axis(fig[2, 3], title="Phototransduction J_inf(G)", xlabel="G (cGMP)", ylabel="J_inf")
    lines!(ax6, collect(G_range), J_vals, color=:navy, linewidth=2)

    Ca_ex = 0.0:0.005:1.0
    C_vals = [RetinalTwin.C∞(c, p.Cae, p.K_ex) for c in Ca_ex]
    ax6b = Axis(fig[2, 4], title="Exchanger C_inf(Ca)", xlabel="[Ca]", ylabel="C_inf")
    lines!(ax6b, collect(Ca_ex), C_vals, color=:brown, linewidth=2)

    # --- Row 3: Glutamate release ---
    r_glu = [RetinalTwin.R_glu_inf(v, p) for v in V]
    glu_target = [p.a_Glu * RetinalTwin.R_glu_inf(v, p) for v in V]
    gl7 = fig[3, 1] = GridLayout()
    ax7 = Axis(gl7[1, 1], title="Glutamate Release Sigmoid", xlabel="V (mV)", ylabel="Value")
    lines!(ax7, collect(V), r_glu, label="R_glu_inf(V)", linewidth=2, color=:black)
    lines!(ax7, collect(V), glu_target, label="a_Glu * R_glu_inf", linewidth=2, color=:red, linestyle=:dash)
    vlines!(ax7, [p.V_Glu_half], color=:gray, linestyle=:dash)
    Legend(gl7[1, 2], ax7, framevisible=false, labelsize=11)

    V_zoom = -50.0:0.1:-20.0
    r_glu_zoom = [p.a_Glu * RetinalTwin.R_glu_inf(v, p) for v in V_zoom]
    gl8 = fig[3, 2] = GridLayout()
    ax8 = Axis(gl8[1, 1], title="Glu Target (operating range)", xlabel="V (mV)", ylabel="a_Glu * R_glu_inf")
    lines!(ax8, collect(V_zoom), r_glu_zoom, color=:red, linewidth=2)
    vlines!(ax8, [-36.186], color=:blue, linestyle=:dash, label="V_dark ~ -36.2 mV")
    vlines!(ax8, [p.V_Glu_half], color=:gray, linestyle=:dash, label="V_Glu_half=$(p.V_Glu_half)")
    Legend(gl8[1, 2], ax8, framevisible=false, labelsize=11)

    # Parameter summary text
    ax9 = Axis(fig[3, 3:4], title="Glutamate Parameters")
    hidedecorations!(ax9)
    text!(ax9, 0.05, 0.9, text="alpha_Glu = $(round(p.alpha_Glu, digits=4))", fontsize=14, space=:relative)
    text!(ax9, 0.05, 0.75, text="V_Glu_half = $(round(p.V_Glu_half, digits=2)) mV", fontsize=14, space=:relative)
    text!(ax9, 0.05, 0.6, text="V_Glu_slope = $(round(p.V_Glu_slope, digits=4)) mV", fontsize=14, space=:relative)
    text!(ax9, 0.05, 0.45, text="a_Glu = $(round(p.a_Glu, digits=4))", fontsize=14, space=:relative)
    text!(ax9, 0.05, 0.3, text="tau_Glu = $(round(p.tau_Glu, digits=2)) s", fontsize=14, space=:relative)
    text!(ax9, 0.05, 0.1, text="Glu_dark ~ $(round(p.a_Glu * RetinalTwin.R_glu_inf(-36.186, p), digits=4))", fontsize=14, space=:relative, color=:blue)

    # --- Row 4: I-V curves ---
    i_photo_vals = [p.iDARK * RetinalTwin.J∞(2.0, 10.0) * (1.0 - exp((v - 8.5) / 17.0)) for v in V]
    i_leak_vals = [p.gLEAK * (v - (-p.ELEAK)) for v in V]
    i_kv_vals = [p.gKV * (RetinalTwin.αmKv(v) / (RetinalTwin.αmKv(v) + RetinalTwin.βmKv(v)))^3 *
                 (RetinalTwin.αhKv(v) / (RetinalTwin.αhKv(v) + RetinalTwin.βhKv(v))) * (v - (-p.eK)) for v in V]
    i_h_vals = [p.gH * h_open_frac[i] * (V[i] - (-p.eH)) for i in eachindex(V)]

    gl10 = fig[4, 1:2] = GridLayout()
    ax10 = Axis(gl10[1, 1], title="Steady-State I-V Curves", xlabel="V (mV)", ylabel="Current (pA)")
    lines!(ax10, collect(V), i_photo_vals, label="I_photo", linewidth=2)
    lines!(ax10, collect(V), i_leak_vals, label="I_leak", linewidth=2)
    lines!(ax10, collect(V), i_kv_vals, label="I_Kv", linewidth=2)
    lines!(ax10, collect(V), i_h_vals, label="I_h", linewidth=2)
    hlines!(ax10, [0.0], color=:gray, linestyle=:dot)
    vlines!(ax10, [-36.186], color=:blue, linestyle=:dash, label="V_dark")
    Legend(gl10[1, 2], ax10, framevisible=false, labelsize=11)

    i_total = [RetinalTwin.I_photoreceptor(v, p) for v in V]
    gl11 = fig[4, 3:4] = GridLayout()
    ax11 = Axis(gl11[1, 1], title="Total I-V (steady-state)", xlabel="V (mV)", ylabel="Current (pA)")
    lines!(ax11, collect(V), i_total, color=:black, linewidth=2)
    hlines!(ax11, [0.0], color=:gray, linestyle=:dot)
    vlines!(ax11, [-36.186], color=:blue, linestyle=:dash, label="V_dark")
    Legend(gl11[1, 2], ax11, framevisible=false, labelsize=11)

    save(joinpath(savedir, "fig_PC_auxiliary.png"), fig)
    fig
end

# =============================================================================
#%% 2) ON Bipolar Cell auxiliary functions
# =============================================================================
let p = params.ONBC
    fig = Figure(size=(1600, 900))
    Label(fig[0, 1:3], "ON Bipolar Cell (ONBC) Auxiliary Functions", fontsize=20, font=:bold)

    # --- Row 1: Voltage-dependent gating ---
    n_inf = [RetinalTwin.gate_inf(v, p.Vn_half, p.kn_slope) for v in V]
    h_inf = [RetinalTwin.gate_inf(v, p.Vh_half, p.kh_slope) for v in V]
    m_inf = [RetinalTwin.gate_inf(v, p.Vm_half, p.km_slope) for v in V]

    gl1 = fig[1, 1] = GridLayout()
    ax1 = Axis(gl1[1, 1], title="Voltage Gating", xlabel="V (mV)", ylabel="Activation")
    lines!(ax1, collect(V), n_inf, label="n_inf (Kv)", linewidth=2)
    lines!(ax1, collect(V), h_inf, label="h_inf (Ih)", linewidth=2)
    lines!(ax1, collect(V), m_inf, label="m_inf (CaL)", linewidth=2)
    Legend(gl1[1, 2], ax1, framevisible=false, labelsize=11)

    # --- Row 1: mGluR6 / TRPM1 synapse ---
    Glu_range = 0.0:0.01:2.0
    S_inf_vals = [RetinalTwin.S_inf(g, p.K_Glu, p.n_Glu) for g in Glu_range]
    gl2 = fig[1, 2] = GridLayout()
    ax2 = Axis(gl2[1, 1], title="mGluR6: S_inf (inv_hill of Glu)", xlabel="[Glu]", ylabel="S_inf")
    lines!(ax2, collect(Glu_range), S_inf_vals, color=:purple, linewidth=2)
    vlines!(ax2, [p.K_Glu], color=:gray, linestyle=:dash, label="K_Glu=$(p.K_Glu)")
    Legend(gl2[1, 2], ax2, framevisible=false, labelsize=11)

    # S_inf scaled by a_S
    ax3 = Axis(fig[1, 3], title="TRPM1 Target: a_S * S_inf", xlabel="[Glu]", ylabel="a_S * S_inf")
    S_target = [p.a_S * RetinalTwin.S_inf(g, p.K_Glu, p.n_Glu) for g in Glu_range]
    lines!(ax3, collect(Glu_range), S_target, color=:purple, linewidth=2)
    text!(ax3, 0.5, 0.9, text="a_S=$(p.a_S), tau_S=$(p.tau_S)", fontsize=12, space=:relative)

    # --- Row 2: Ca-dependent release ---
    Ca_range = 0.0:0.01:2.0
    R_inf_vals = [RetinalTwin.R_inf(c, p.K_Release, p.n_Release) for c in Ca_range]
    gl4 = fig[2, 1] = GridLayout()
    ax4 = Axis(gl4[1, 1], title="Glu Release: R_inf(Ca)", xlabel="[Ca]", ylabel="R_inf")
    lines!(ax4, collect(Ca_range), R_inf_vals, color=:red, linewidth=2)
    vlines!(ax4, [p.K_Release], color=:gray, linestyle=:dash, label="K_Release=$(p.K_Release)")
    Legend(gl4[1, 2], ax4, framevisible=false, labelsize=11)

    # KCa hill(c)
    kca_vals = [RetinalTwin.hill(c, p.K_c, p.n_c) for c in Ca_range]
    gl5 = fig[2, 2] = GridLayout()
    ax5 = Axis(gl5[1, 1], title="KCa Activation: hill(Ca)", xlabel="[Ca]", ylabel="a_c")
    lines!(ax5, collect(Ca_range), kca_vals, color=:orange, linewidth=2)
    vlines!(ax5, [p.K_c], color=:gray, linestyle=:dash, label="K_c=$(p.K_c)")
    Legend(gl5[1, 2], ax5, framevisible=false, labelsize=11)

    # --- Row 2: I-V curves ---
    i_total = [RetinalTwin.I_on_bipolar(v, p) for v in V]
    ax6 = Axis(fig[2, 3], title="Total I-V (no synaptic drive)", xlabel="V (mV)", ylabel="Current")
    lines!(ax6, collect(V), i_total, color=:black, linewidth=2)
    hlines!(ax6, [0.0], color=:gray, linestyle=:dot)

    # --- Row 3: Parameter summary ---
    ax7 = Axis(fig[3, 1:3], title="Key ONBC Parameters")
    hidedecorations!(ax7)
    text!(ax7, 0.02, 0.8, text="Synaptic: K_Glu=$(p.K_Glu), n_Glu=$(p.n_Glu), g_TRPM1=$(p.g_TRPM1), E_TRPM1=$(p.E_TRPM1), a_S=$(p.a_S), tau_S=$(p.tau_S)", fontsize=13, space=:relative)
    text!(ax7, 0.02, 0.5, text="Release: K_Release=$(p.K_Release), n_Release=$(p.n_Release), a_Release=$(p.a_Release), tau_Release=$(p.tau_Release)", fontsize=13, space=:relative)
    text!(ax7, 0.02, 0.2, text="Channels: g_L=$(p.g_L), g_Kv=$(p.g_Kv), g_h=$(p.g_h), g_CaL=$(p.g_CaL), g_KCa=$(p.g_KCa), C_m=$(p.C_m)", fontsize=13, space=:relative)

    save(joinpath(savedir, "fig_ONBC_auxiliary.png"), fig)
    fig
end

# =============================================================================
#%% 3) OFF Bipolar Cell auxiliary functions
# =============================================================================
let p = params.OFFBC
    fig = Figure(size=(1600, 900))
    Label(fig[0, 1:3], "OFF Bipolar Cell (OFFBC) Auxiliary Functions", fontsize=20, font=:bold)

    # --- Row 1: Voltage-dependent gating ---
    n_inf = [RetinalTwin.gate_inf(v, p.Vn_half, p.kn_slope) for v in V]
    h_inf = [RetinalTwin.gate_inf(v, p.Vh_half, p.kh_slope) for v in V]
    m_inf = [RetinalTwin.gate_inf(v, p.Vm_half, p.km_slope) for v in V]

    gl1 = fig[1, 1] = GridLayout()
    ax1 = Axis(gl1[1, 1], title="Voltage Gating", xlabel="V (mV)", ylabel="Activation")
    lines!(ax1, collect(V), n_inf, label="n_inf (Kv)", linewidth=2)
    lines!(ax1, collect(V), h_inf, label="h_inf (Ih)", linewidth=2)
    lines!(ax1, collect(V), m_inf, label="m_inf (CaL)", linewidth=2)
    Legend(gl1[1, 2], ax1, framevisible=false, labelsize=11)

    # --- Row 1: iGluR synapse ---
    Glu_range = 0.0:0.01:2.0
    A_inf_vals = [RetinalTwin.A_inf(g, p.K_a, p.n_a) for g in Glu_range]
    D_inf_vals = [RetinalTwin.D_inf(g, p.K_d, p.n_d) for g in Glu_range]
    open_vals = A_inf_vals .* D_inf_vals

    gl2 = fig[1, 2] = GridLayout()
    ax2 = Axis(gl2[1, 1], title="iGluR Gating", xlabel="[Glu]", ylabel="Value")
    lines!(ax2, collect(Glu_range), A_inf_vals, label="A_inf (act)", linewidth=2, color=:red)
    lines!(ax2, collect(Glu_range), D_inf_vals, label="D_inf (desens)", linewidth=2, color=:blue)
    lines!(ax2, collect(Glu_range), open_vals, label="A*D (open)", linewidth=2, color=:black)
    vlines!(ax2, [p.K_a], color=:red, linestyle=:dash, alpha=0.5)
    vlines!(ax2, [p.K_d], color=:blue, linestyle=:dash, alpha=0.5)
    Legend(gl2[1, 2], ax2, framevisible=false, labelsize=11)

    # iGluR scaled targets
    A_target = [p.a_a * RetinalTwin.A_inf(g, p.K_a, p.n_a) for g in Glu_range]
    D_target = [p.a_d * RetinalTwin.D_inf(g, p.K_d, p.n_d) for g in Glu_range]
    gl3 = fig[1, 3] = GridLayout()
    ax3 = Axis(gl3[1, 1], title="iGluR Scaled Targets", xlabel="[Glu]", ylabel="Value")
    lines!(ax3, collect(Glu_range), A_target, label="a_a*A_inf (tau=$(p.tau_A))", linewidth=2, color=:red)
    lines!(ax3, collect(Glu_range), D_target, label="a_d*D_inf (tau=$(p.tau_d))", linewidth=2, color=:blue)
    Legend(gl3[1, 2], ax3, framevisible=false, labelsize=11)

    # --- Row 2: Ca-dependent release ---
    Ca_range = 0.0:0.01:2.0
    R_inf_vals = [RetinalTwin.R_inf(c, p.K_Release, p.n_Release) for c in Ca_range]
    gl4 = fig[2, 1] = GridLayout()
    ax4 = Axis(gl4[1, 1], title="Glu Release: R_inf(Ca)", xlabel="[Ca]", ylabel="R_inf")
    lines!(ax4, collect(Ca_range), R_inf_vals, color=:red, linewidth=2)
    vlines!(ax4, [p.K_Release], color=:gray, linestyle=:dash, label="K_Release=$(p.K_Release)")
    Legend(gl4[1, 2], ax4, framevisible=false, labelsize=11)

    # KCa
    kca_vals = [RetinalTwin.hill(c, p.K_c, p.n_c) for c in Ca_range]
    gl5 = fig[2, 2] = GridLayout()
    ax5 = Axis(gl5[1, 1], title="KCa Activation: hill(Ca)", xlabel="[Ca]", ylabel="a_c")
    lines!(ax5, collect(Ca_range), kca_vals, color=:orange, linewidth=2)
    vlines!(ax5, [p.K_c], color=:gray, linestyle=:dash, label="K_c=$(p.K_c)")
    Legend(gl5[1, 2], ax5, framevisible=false, labelsize=11)

    # I-V curves
    i_total = [RetinalTwin.I_off_bipolar(v, p) for v in V]
    ax6 = Axis(fig[2, 3], title="Total I-V (no synaptic drive)", xlabel="V (mV)", ylabel="Current")
    lines!(ax6, collect(V), i_total, color=:black, linewidth=2)
    hlines!(ax6, [0.0], color=:gray, linestyle=:dot)

    # --- Row 3: Parameters ---
    ax7 = Axis(fig[3, 1:3], title="Key OFFBC Parameters")
    hidedecorations!(ax7)
    text!(ax7, 0.02, 0.8, text="Synaptic: K_a=$(p.K_a), n_a=$(p.n_a), K_d=$(p.K_d), n_d=$(p.n_d), g_iGluR=$(p.g_iGluR), E_iGluR=$(p.E_iGluR)", fontsize=13, space=:relative)
    text!(ax7, 0.02, 0.5, text="Kinetics: a_a=$(p.a_a), tau_A=$(p.tau_A), a_d=$(p.a_d), tau_d=$(p.tau_d)", fontsize=13, space=:relative)
    text!(ax7, 0.02, 0.2, text="Channels: g_L=$(p.g_L), g_Kv=$(p.g_Kv), g_h=$(p.g_h), g_CaL=$(p.g_CaL), g_KCa=$(p.g_KCa), C_m=$(p.C_m)", fontsize=13, space=:relative)

    save(joinpath(savedir, "fig_OFFBC_auxiliary.png"), fig)
    fig
end

# =============================================================================
#%% 4) Horizontal Cell auxiliary functions
# =============================================================================
let p = params.HC
    fig = Figure(size=(1600, 900))
    Label(fig[0, 1:3], "Horizontal Cell (HC) Auxiliary Functions", fontsize=20, font=:bold)

    # --- Row 1: Voltage gating ---
    mCa_inf = [RetinalTwin.gate_inf(v, p.Vm_half, p.km_slope) for v in V]
    ax1 = Axis(fig[1, 1], title="CaL Gating", xlabel="V (mV)", ylabel="mCa_inf")
    lines!(ax1, collect(V), mCa_inf, color=:royalblue, linewidth=2)

    # Kir rectification
    kir_vals = [RetinalTwin.kir_rect(v, p.E_Kir; Vshift=p.Kir_Vshift, k=p.Kir_k) for v in V]
    gl2 = fig[1, 2] = GridLayout()
    ax2 = Axis(gl2[1, 1], title="Kir Rectification", xlabel="V (mV)", ylabel="r_kir")
    lines!(ax2, collect(V), kir_vals, color=:teal, linewidth=2)
    vlines!(ax2, [p.E_Kir], color=:gray, linestyle=:dash, label="E_Kir=$(p.E_Kir)")
    Legend(gl2[1, 2], ax2, framevisible=false, labelsize=11)

    # BK channel
    gl3 = fig[1, 3] = GridLayout()
    ax3 = Axis(gl3[1, 1], title="BK Channel: mBK_inf(V, Ca)", xlabel="V (mV)", ylabel="mBK_inf")
    for ca_val in [0.01, 0.1, 0.5, 1.0]
        mbk = [RetinalTwin.mBK_inf(v, ca_val; Vhalf0=p.Vhalf0_BK, k=p.k_BK, s=p.s_BK, Caref=p.Caref_BK) for v in V]
        lines!(ax3, collect(V), mbk, label="Ca=$ca_val", linewidth=1.5)
    end
    Legend(gl3[1, 2], ax3, framevisible=false, labelsize=11)

    # --- Row 2: Synaptic input ---
    Glu_range = 0.0:0.01:2.0
    s_inf_vals = [RetinalTwin.hill(g, p.K_Glu, p.n_Glu) for g in Glu_range]
    gl4 = fig[2, 1] = GridLayout()
    ax4 = Axis(gl4[1, 1], title="Excitatory Input: hill(Glu)", xlabel="[Glu]", ylabel="s_inf")
    lines!(ax4, collect(Glu_range), s_inf_vals, color=:red, linewidth=2)
    vlines!(ax4, [p.K_Glu], color=:gray, linestyle=:dash, label="K_Glu=$(p.K_Glu)")
    Legend(gl4[1, 2], ax4, framevisible=false, labelsize=11)

    # Release proxy
    Ca_range = 0.0:0.01:2.0
    R_inf_vals = [RetinalTwin.R_inf(c, p.K_Release, p.n_Release) for c in Ca_range]
    gl5 = fig[2, 2] = GridLayout()
    ax5 = Axis(gl5[1, 1], title="Release Proxy: R_inf(Ca)", xlabel="[Ca]", ylabel="R_inf")
    lines!(ax5, collect(Ca_range), R_inf_vals, color=:orange, linewidth=2)
    vlines!(ax5, [p.K_Release], color=:gray, linestyle=:dash, label="K_Release=$(p.K_Release)")
    Legend(gl5[1, 2], ax5, framevisible=false, labelsize=11)

    # HC feedback
    V_hc = -80.0:0.5:0.0
    fb = [RetinalTwin.hc_feedback(v; g_FB=1.0) for v in V_hc]
    ax6 = Axis(fig[2, 3], title="HC Feedback Sigmoid", xlabel="V_hc (mV)", ylabel="hc_feedback")
    lines!(ax6, collect(V_hc), fb, color=:darkgreen, linewidth=2)

    # I-V
    i_total = [RetinalTwin.I_horizontal(v, p) for v in V]
    ax7 = Axis(fig[3, 1:3], title="Total I-V (steady-state, baseline Ca)", xlabel="V (mV)", ylabel="Current")
    lines!(ax7, collect(V), i_total, color=:black, linewidth=2)
    hlines!(ax7, [0.0], color=:gray, linestyle=:dot)

    save(joinpath(savedir, "fig_HC_auxiliary.png"), fig)
    fig
end

# =============================================================================
#%% 5) A2 Amacrine Cell auxiliary functions
# =============================================================================
let p = params.A2
    fig = Figure(size=(1600, 900))
    Label(fig[0, 1:3], "A2 Amacrine Cell Auxiliary Functions", fontsize=20, font=:bold)

    # --- Row 1: Voltage gating ---
    n_inf = [RetinalTwin.gate_inf(v, p.Vn_half, p.kn_slope) for v in V]
    h_inf = [RetinalTwin.gate_inf(v, p.Vh_half, p.kh_slope) for v in V]
    m_inf = [RetinalTwin.gate_inf(v, p.Vm_half, p.km_slope) for v in V]

    gl1 = fig[1, 1] = GridLayout()
    ax1 = Axis(gl1[1, 1], title="Voltage Gating", xlabel="V (mV)", ylabel="Activation")
    lines!(ax1, collect(V), n_inf, label="n_inf (Kv)", linewidth=2)
    lines!(ax1, collect(V), h_inf, label="h_inf (Ih)", linewidth=2)
    lines!(ax1, collect(V), m_inf, label="m_inf (CaL)", linewidth=2)
    Legend(gl1[1, 2], ax1, framevisible=false, labelsize=11)

    # --- Row 1: iGluR synapse ---
    Glu_range = 0.0:0.01:2.0
    A_inf_vals = [RetinalTwin.A_inf(g, p.K_a, p.n_a) for g in Glu_range]
    D_inf_vals = [RetinalTwin.D_inf(g, p.K_d, p.n_d) for g in Glu_range]
    open_vals = A_inf_vals .* D_inf_vals

    gl2 = fig[1, 2] = GridLayout()
    ax2 = Axis(gl2[1, 1], title="iGluR Gating", xlabel="[Glu]", ylabel="Value")
    lines!(ax2, collect(Glu_range), A_inf_vals, label="A_inf", linewidth=2, color=:red)
    lines!(ax2, collect(Glu_range), D_inf_vals, label="D_inf", linewidth=2, color=:blue)
    lines!(ax2, collect(Glu_range), open_vals, label="A*D (open)", linewidth=2, color=:black)
    vlines!(ax2, [p.K_a], color=:red, linestyle=:dash, alpha=0.5)
    vlines!(ax2, [p.K_d], color=:blue, linestyle=:dash, alpha=0.5)
    Legend(gl2[1, 2], ax2, framevisible=false, labelsize=11)

    # Glycine release
    Ca_range = 0.0:0.01:2.0
    R_inf_vals = [RetinalTwin.R_inf(c, p.K_Release, p.n_Release) for c in Ca_range]
    gl3 = fig[1, 3] = GridLayout()
    ax3 = Axis(gl3[1, 1], title="Glycine Release: R_inf(Ca)", xlabel="[Ca]", ylabel="R_inf")
    lines!(ax3, collect(Ca_range), R_inf_vals, color=:red, linewidth=2)
    vlines!(ax3, [p.K_Release], color=:gray, linestyle=:dash, label="K_Release=$(p.K_Release)")
    Legend(gl3[1, 2], ax3, framevisible=false, labelsize=11)

    # --- Row 2: KCa + I-V ---
    kca_vals = [RetinalTwin.hill(c, p.K_c, p.n_c) for c in Ca_range]
    ax4 = Axis(fig[2, 1], title="KCa Activation: hill(Ca)", xlabel="[Ca]", ylabel="a_c")
    lines!(ax4, collect(Ca_range), kca_vals, color=:orange, linewidth=2)

    i_total = [RetinalTwin.I_a2_amacrine(v, p) for v in V]
    ax5 = Axis(fig[2, 2:3], title="Total I-V (no synaptic drive)", xlabel="V (mV)", ylabel="Current")
    lines!(ax5, collect(V), i_total, color=:black, linewidth=2)
    hlines!(ax5, [0.0], color=:gray, linestyle=:dot)

    # --- Row 3: Parameters ---
    ax6 = Axis(fig[3, 1:3], title="Key A2 Parameters")
    hidedecorations!(ax6)
    text!(ax6, 0.02, 0.8, text="Synaptic: K_a=$(p.K_a), n_a=$(p.n_a), K_d=$(p.K_d), n_d=$(p.n_d), g_iGluR=$(p.g_iGluR)", fontsize=13, space=:relative)
    text!(ax6, 0.02, 0.5, text="Release: K_Release=$(p.K_Release), n_Release=$(p.n_Release), a_Release=$(p.a_Release), tau_Release=$(p.tau_Release)", fontsize=13, space=:relative)
    text!(ax6, 0.02, 0.2, text="Gap junction: g_gap=$(p.g_gap)", fontsize=13, space=:relative)

    save(joinpath(savedir, "fig_A2_auxiliary.png"), fig)
    fig
end

# =============================================================================
#%% 6) Ganglion Cell auxiliary functions
# =============================================================================
let p = params.GC
    fig = Figure(size=(1600, 900))
    Label(fig[0, 1:3], "Ganglion Cell (GC) Auxiliary Functions", fontsize=20, font=:bold)

    # --- Row 1: HH gating ---
    m_inf = zeros(length(V))
    h_inf = zeros(length(V))
    n_inf = zeros(length(V))
    tau_m = zeros(length(V))
    tau_h = zeros(length(V))
    tau_n = zeros(length(V))
    for (i, v) in enumerate(V)
        am, bm = RetinalTwin.alpha_beta_m(v)
        ah, bh = RetinalTwin.alpha_beta_h(v)
        an, bn = RetinalTwin.alpha_beta_n(v)
        m_inf[i] = am / (am + bm + eps())
        h_inf[i] = ah / (ah + bh + eps())
        n_inf[i] = an / (an + bn + eps())
        tau_m[i] = 1.0 / (am + bm + eps())
        tau_h[i] = 1.0 / (ah + bh + eps())
        tau_n[i] = 1.0 / (an + bn + eps())
    end

    gl1 = fig[1, 1] = GridLayout()
    ax1 = Axis(gl1[1, 1], title="HH Gating (steady-state)", xlabel="V (mV)", ylabel="Activation")
    lines!(ax1, collect(V), m_inf, label="m_inf (Na act)", linewidth=2)
    lines!(ax1, collect(V), h_inf, label="h_inf (Na inact)", linewidth=2)
    lines!(ax1, collect(V), n_inf, label="n_inf (K act)", linewidth=2)
    Legend(gl1[1, 2], ax1, framevisible=false, labelsize=11)

    gl2 = fig[1, 2] = GridLayout()
    ax2 = Axis(gl2[1, 1], title="HH Time Constants", xlabel="V (mV)", ylabel="tau (ms)")
    lines!(ax2, collect(V), tau_m, label="tau_m", linewidth=2)
    lines!(ax2, collect(V), tau_h, label="tau_h", linewidth=2)
    lines!(ax2, collect(V), tau_n, label="tau_n", linewidth=2)
    Legend(gl2[1, 2], ax2, framevisible=false, labelsize=11)

    # --- Row 1: Synaptic gating ---
    Glu_range = 0.0:0.01:2.0
    sE_inf = [RetinalTwin.hill(g, p.K_preE, p.n_preE) for g in Glu_range]
    gl3 = fig[1, 3] = GridLayout()
    ax3 = Axis(gl3[1, 1], title="Excitatory Synapse: hill(Glu)", xlabel="[Glu]", ylabel="sE_inf")
    lines!(ax3, collect(Glu_range), sE_inf, color=:red, linewidth=2)
    vlines!(ax3, [p.K_preE], color=:gray, linestyle=:dash, label="K_preE=$(p.K_preE)")
    Legend(gl3[1, 2], ax3, framevisible=false, labelsize=11)

    # --- Row 2: Inhibitory synapse + I-V ---
    Gly_range = 0.0:0.01:2.0
    sI_inf = [RetinalTwin.hill(g, p.K_preI, p.n_preI) for g in Gly_range]
    gl4 = fig[2, 1] = GridLayout()
    ax4 = Axis(gl4[1, 1], title="Inhibitory Synapse: hill(Gly)", xlabel="[Gly]", ylabel="sI_inf")
    lines!(ax4, collect(Gly_range), sI_inf, color=:blue, linewidth=2)
    vlines!(ax4, [p.K_preI], color=:gray, linestyle=:dash, label="K_preI=$(p.K_preI)")
    Legend(gl4[1, 2], ax4, framevisible=false, labelsize=11)

    i_total = [RetinalTwin.I_ganglion(v, p) for v in V]
    ax5 = Axis(fig[2, 2:3], title="Total I-V (no synaptic drive)", xlabel="V (mV)", ylabel="Current")
    lines!(ax5, collect(V), i_total, color=:black, linewidth=2)
    hlines!(ax5, [0.0], color=:gray, linestyle=:dot)

    # --- Row 3: Parameters ---
    ax6 = Axis(fig[3, 1:3], title="Key GC Parameters")
    hidedecorations!(ax6)
    text!(ax6, 0.02, 0.8, text="HH: g_Na=$(p.g_Na), g_K=$(p.g_K), g_L=$(p.g_L), E_Na=$(p.E_Na), E_K=$(p.E_K), E_L=$(p.E_L)", fontsize=13, space=:relative)
    text!(ax6, 0.02, 0.5, text="Exc: g_E=$(p.g_E), E_E=$(p.E_E), K_preE=$(p.K_preE), n_preE=$(p.n_preE), tau_E=$(p.tau_E)", fontsize=13, space=:relative)
    text!(ax6, 0.02, 0.2, text="Inh: g_I=$(p.g_I), E_I=$(p.E_I), K_preI=$(p.K_preI), n_preI=$(p.n_preI), tau_I=$(p.tau_I)", fontsize=13, space=:relative)

    save(joinpath(savedir, "fig_GC_auxiliary.png"), fig)
    fig
end

# =============================================================================
#%% 7) Muller Glial Cell auxiliary functions
# =============================================================================
let p = params.MULLER
    fig = Figure(size=(1600, 600))
    Label(fig[0, 1:3], "Muller Glial Cell (MG) Auxiliary Functions", fontsize=20, font=:bold)

    # --- Row 1: Kir rectification + Nernst ---
    K_o_range = 1.0:0.1:15.0
    E_K_vals = [RetinalTwin.nernst_K(ko, p.K_i) for ko in K_o_range]
    gl1 = fig[1, 1] = GridLayout()
    ax1 = Axis(gl1[1, 1], title="Nernst E_K vs [K+]_o", xlabel="[K+]_o (mM)", ylabel="E_K (mV)")
    lines!(ax1, collect(K_o_range), E_K_vals, color=:navy, linewidth=2)
    vlines!(ax1, [p.K_o_rest], color=:gray, linestyle=:dash, label="K_o_rest=$(p.K_o_rest)")
    Legend(gl1[1, 2], ax1, framevisible=false, labelsize=11)

    # Kir rectification at different E_K
    gl2 = fig[1, 2] = GridLayout()
    ax2 = Axis(gl2[1, 1], title="Kir Rectification", xlabel="V (mV)", ylabel="r_kir")
    for ek in [-90.0, -80.0, -70.0, -60.0]
        kir_vals = [RetinalTwin.kir_rect(v, ek; Vshift=p.Kir_Vshift, k=p.Kir_k) for v in V]
        lines!(ax2, collect(V), kir_vals, label="E_K=$(ek)", linewidth=1.5)
    end
    Legend(gl2[1, 2], ax2, framevisible=false, labelsize=11)

    # EAAT glutamate uptake
    Glu_range = 0.0:0.01:2.0
    eaat_vals = [p.V_max_EAAT * RetinalTwin.hill(g, p.K_m_EAAT, p.n_EAAT) for g in Glu_range]
    gl3 = fig[1, 3] = GridLayout()
    ax3 = Axis(gl3[1, 1], title="EAAT Glutamate Uptake", xlabel="[Glu]_o", ylabel="J_uptake")
    lines!(ax3, collect(Glu_range), eaat_vals, color=:darkgreen, linewidth=2)
    vlines!(ax3, [p.K_m_EAAT], color=:gray, linestyle=:dash, label="K_m=$(p.K_m_EAAT)")
    Legend(gl3[1, 2], ax3, framevisible=false, labelsize=11)

    # I-V
    i_total = [RetinalTwin.I_muller(v, p) for v in V]
    ax4 = Axis(fig[2, 1:3], title="Total I-V (resting K+)", xlabel="V (mV)", ylabel="Current")
    lines!(ax4, collect(V), i_total, color=:black, linewidth=2)
    hlines!(ax4, [0.0], color=:gray, linestyle=:dot)

    save(joinpath(savedir, "fig_MG_auxiliary.png"), fig)
    fig
end
