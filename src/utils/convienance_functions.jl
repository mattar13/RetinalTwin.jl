"""
    _dark_adapt(model, u0, params; tspan_dark=(0.0, 2000.0), abstol=1e-6, reltol=1e-4)

Run the model with zero stimulus to reach dark-adapted steady state.
"""
function dark_adapt(model, u0, params; time = 2000.0, abstol=1e-6, reltol=1e-4)
    stim_dark = make_uniform_flash_stimulus(photon_flux=0.0)
    prob = DifferentialEquations.ODEProblem(model, u0, (0.0, time), (params, stim_dark))
    sol = DifferentialEquations.solve(
        prob, DifferentialEquations.Rodas5();
        save_everystep=false, save_start=false, save_end=true,
        abstol=abstol, reltol=reltol,
    )
    return sol.u[end]
end