sigm(x, l, xh) = x > 0 ? (1/(1+exp(-l*(x-xh)))) : 0

H_inf(v, l, h) = v != 0 ? 1/(1+exp((v+l)/h)) : 0
#H_inf(v, l, h) = 1/(1+exp((v+l)/h))

B_inf(v, l, h) = v > 0 ? 1/(1+exp((v+l)/h)) : 0

Stim(t, stim_start, stim_end, photon_flux) = stim_start <= t <= stim_end ? photon_flux : 0.0