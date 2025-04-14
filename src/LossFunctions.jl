#--- Loss Function ---
function loss_static(data, params; channel = 1, kwargs...)
    # Define simulation time points from your data
    expERG = getchannel(data, channel).data_array[1,:,1]

    sol, ERG_t = simulate_model(data, params; kwargs...)
    n_length = length(ERG_t)
    # Compute the sum-of-squared error compared to experimental data
    return sum((ERG_t .- expERG).^2)#/n_length
end

function loss_static_abm(data_a, data_ab, data_abm, full_params; channel = 3, kwargs...)
    # Define simulation time points from your data
    expERG_A = getchannel(data_a, channel).data_array[1,:,1]
    expERG_AB = getchannel(data_ab, channel).data_array[1,:,1]
    expERG_ABM = getchannel(data_abm, channel).data_array[1,:,1]

    sol, ERG_t = simulate_model(data_a, full_params; kwargs...)
    #Compute the loss for each model
    a_wave = map(t -> sol(t)[7], data_a.t)
    loss_a = sum((a_wave .- expERG_A).^2)#/length(a_wave)
    #println("Loss A: $loss_a")

    ab_wave = map(t -> sol(t)[8], data_ab.t) .+ a_wave
    loss_ab = sum((ab_wave .- expERG_AB).^2)#/length(ab_wave)
    #println("Loss AB: $loss_ab")

    abm_wave = map(t -> sol(t)[15], data_abm.t)
    loss_abm = sum((abm_wave .- expERG_ABM).^2)#/length(abm_wave)
    #println("Loss ABM: $loss_abm")
    return loss_a, loss_ab, loss_abm
end

function loss_graded(data_series, params; channel = 1, stim_start = 0.0, stim_end = 1.0)
    # Define simulation time points from your data
    loss = 0.0
    ir_loss = 0.0
    weight_loss = 0.0

    loss_vals = []
    ir_loss_vals = []
    weight_loss_vals = []
    for (i, (k, data)) in enumerate(data_series)
        expERG = getchannel(data, channel).data_array[1,:,1]
        sol, ERG_t = simulate_model(data, params; stim_start, stim_end, photon_flux = k)
        a_wave = map(t -> sol(t)[7], data.t)
        
        #Compute the IR error
        loss_ir = abs(minimum(a_wave)) .- (minimum(expERG))
        ir_loss += loss_ir
        push!(loss_vals, loss_ir)
        
        # Compute the sum-of-squared error compared to experimental data
        n_length = length(ERG_t)
        loss_val = sum((a_wave .- expERG).^2)/n_length
        loss += loss_val
        push!(loss_vals, loss_val)

        #Compute the weighted loss
        weights = map(t -> exp(-t/200), data.t)  # Exponential decay 
        weight_loss_val = sum(weights .* (a_wave .- expERG).^2)/n_length
        weight_loss += weight_loss_val
        push!(weight_loss_vals, weight_loss_val)
    end
    return ir_loss, loss, weight_loss, ir_loss_vals, loss_vals, weight_loss_vals
    #return loss_vals
end


