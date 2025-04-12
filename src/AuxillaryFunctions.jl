#The stimulus function determines how much light is present at a given time.
Stim(t, stim_start, stim_end, photon_flux) = stim_start <= t <= stim_end ? photon_flux : 0.0

sigm(x, l, xh) = x > 0 ? (1/(1+exp(-l*(x-xh)))) : 0

H_inf(v, l, h) = v != 0 ? 1/(1+exp((v+l)/h)) : 0
#H_inf(v, l, h) = 1/(1+exp((v+l)/h))

B_inf(v, l, h) = v > 0 ? 1/(1+exp((v+l)/h)) : 0

#Below is a work in progress for a more complicated photoreceptor model
kC = 0.1 #CNG channel opening rate
gL = 0.35 #Leakage conductance
EL =  #Leakage potential

A(Ca, A_max, kC) = A_max/(1.0 + (Ca/kC)^4)
J(cGMP, Jmax) = (Jmax*cGMP^3)/(cGMP^3 + 1000.0) #CNG currents

I_PHOTO(V, cGMP, Jmax) = -J(cGMP, Jmax)*(1.0 - exp((V-8.5)/17.0))
I_L(v, gL, EL) = gL * (v - EL)
I_KV(v, gKV, mKV, hKV, EK) = gKV * mKV^3 * hKV * (v - EK)

ECa()
I_Ca(v, gCa, mCa, ECa) = gCa * mCa^4 * hCa(v) * (v - ECa)

I_Cl(v, gCl, _Ca_s, ECl) = gCl * mCl(_Ca_s) * (v - ECl)
I_KCa(v, gKCa, mKCa, _Ca_s, EK) = gKCa * mKCa^2 * mKCas(_Ca_s) * (v - EK)
I_ex(v, J_ex, _Ca_s, Cae, K_ex) = J_ex * exp(-(v - 14) / 70) * (_Ca_s - Cae) / (_Ca_s - Cae + K_ex)
I_ex2(J_ex2, _Ca_s, Cae, K_ex2) = J_ex2 * (_Ca_s - Cae) / (_Ca_s - Cae + K_ex2)
I_h(v, gH, O1, O2, O3, EH) = gH * (O1 + O2 + O3) * (v - EH)

#Kv equations
αmKV(v) = (5 * (100 - v)) / (exp((100 - v) / 42) - 1)
βmKV(v) = 9 * exp(-(v - 20) / 40)
αhKV(v) = 0.15 * exp(-v / 22)
βhKV(v) = 0.4125 / (exp((10 - v) / 7) + 1)

#Calcium equations
αmCa(v) = (3 * (80 - v)) / (exp((80 - v) / 25) - 1)
βmCa(v) = 10 / (1 + exp((v + 38) / 7))
hCa(v) = exp((40 - v) / 18) / (1 + exp((40 - v) / 18))

αmKCa(v) = (15 * (80 - v)) / (exp((80 - v) / 40) - 1)
βmKCa(v) = 20 * exp(-v / 35)
mKCas(_Ca_s) = _Ca_s / (_Ca_s + 0.3)

#Calcium activated chloride
mCl(_Ca_s) = 1 / (1 + exp((0.37 - _Ca_s) / 0.9))

#Calcium activated potassium
αmKCa(v) = (15 * (80 - v)) / (exp((80 - v) / 40) - 1)
βmKCa(v) = 20 * exp(-v / 35)
mKCas(_Ca_s) = _Ca_s / (_Ca_s + 0.3)

#Hyperpolarising current
αh(v) = 8 / (exp((v + 78) / 14) + 1)
βh(v) = 18 / (exp(-(v + 9) / 19) + 1)

#Hyperpolarising current
αh(v) = 8 / (exp((v + 78) / 14) + 1)
βh(v) = 18 / (exp(-(v + 9) / 19) + 1)

hT(v) = [
     -4*αh(v) βh(v) 0.0 0.0 0.0
     4*αh(v) -(3*αh(v) + βh(v)) 2*βh(v) 0.0 0.0
     0.0 3*αh(v) -(2*αh(v)+2*βh(v)) 3*βh(v) 0.0
     0.0 0.0 2*αh(v) -(αh(v) + 3*βh(v)) 4*βh(v)
     0.0 0.0 0.0 αh(v) -4*βh(v)
]