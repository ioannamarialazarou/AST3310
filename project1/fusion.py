"Project 1"

# Install prerequisite libraries
from numpy import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def stellar_engines(rho, T, sanity):

  # General physics constants
    m_u   = 1.660539066e-27         # atomic mass unit [kg]
    m_p   = 1.007825032241*m_u
    m_e   = 9.1094e-31          # electron mass [kg]
    m_3He = 3.016029322650*m_u      # Helium-3 mass [kg]
    m_4He = 4.002603254130*m_u      # Helium-4 mass [kg]
    m_7Be = 7.016928720000*m_u      # Beryllium-7 mass [kg]
    m_7Li = 7.016003437000*m_u      # Lithium-7 mass [kg]
    m_14N = 14.00307400446*m_u      # Nitrogen-14 mass [kg]
    c = 2.9979e8                    # speed of light [m s^-1]
    MeV = 1.602176e-13              # Joule per MeV
    NA = 6.02214e23                 # Avogadro's number
    k_B   = 1.3806e-23              # Boltzmann constant [m^2 kg s^-2 K^-1]
    MeV   = 1.602176e-13            # Joule per MeV
    NA    = 6.02214e23              # Avogadro's number

    # Mass fraction of each atomic species, where X and Y are the fractional abundances by weight of hydrogen and helium respectively
    X = 0.7
    Y_4He = 0.29
    Y_3He = 1e-10
    Z_7Li = 1e-7
    Z_7Be = 1e-7
    Z_14N = 1e-11

    # Neutrino energies
    evpp=0.265* MeV
    ev33= 0
    ev34= 0
    eve7= 0.815* MeV
    ev17prime= 0
    ev8= 6.711* MeV
    evCNO= 1.704 * MeV

    Evs= [evpp,ev33,ev34,eve7,ev17prime,ev8,evCNO]

    T9 = 1e-9 * T    # Convert temperature in K to K^9
    epsilon = 0      # initialize total energy gain         711

    # Number Densities
    n_p = rho*X/m_u # Number of Hydrogen atoms
    n_He3 = rho*Y_3He/m_3He # Number of Helium 3 atoms, etc
    n_He4 = rho*Y_4He/m_4He
    n_7Li = rho*Z_7Li/m_7Li
    n_7Be = rho*Z_7Be/m_7Be
    n_14N = rho*Z_14N/m_14N
    n_e = rho/(2*m_u)*(1 + X) #the total number of electrons ne,tot is the sum of the electrons from hydrogen ne,H and from helium ne,He

    # Proportion functions, e.i. lambdas:
    constant = 1e-6/NA # constant to convert from cm^3 to m^3 and divide by NA (Avogadros constant)

    T9_star = T9/(1 + 4.95*1e-2*T9)
    l_pp = constant* (4.01e-15 * T9**(-2/3) * exp(-3.380*T9**(-1/3)) * (1+0.123*T9**(1/3)+1.09*T9**(2/3)+0.938*T9))
    l_33 = constant* (6.05*1e10*T9**(-2/3)*exp(-12.276*T9**(-1/3))*(1 + 0.034*T9**(1/3) - 0.522*T9**(2/3) - 0.124*T9+ 0.353*T9**(4/3) + 0.213*T9**(5/3)))
    l_34 = constant* (5.61e6*T9_star**(5/6)*T9**(-3/2)*exp(-12.826*T9_star**(-1/3)))

    if T9 < 1e-3: # include upper limit of 7Be electron capture
        l_e7 = 1.57e-7
    else:
        l_e7 = constant* (1.34e-10*T9**(-1/2)*(1 - 0.537*T9**(1/3) + 3.86*T9**(2/3) + 0.0027*T9**(-1)*exp(2.515*10**-3*T9**-1)))

    l_17_prime = constant* (1.096e9*T9**(-2/3)*exp(-8.472*T9**(-1/3)) - 4.830e8*T9_star**(5/6)*T9**(-3/2)*exp(-8.472*T9_star**(-1/3))+ 1.06e10*T9**(-3/2)*exp(-30.442*T9**(-1)))
    l_17= constant* (3.11e5*T9**(-2/3)*exp(-10.262*T9**(-1/3)) +2.53e3*T9**(-3/2)*exp(-7.306*T9**(-1)))
    l_p14 = constant* (4.90e7*T9**(-2/3)*exp(-15.228*T9**(-1/3) - 0.92*T9**2)* (1 + 0.027*T9**(1/3) - 0.778*T9**(2/3) - 0.149*T9 + 0.261*T9**(4/3) + 0.127*T9**(5/3))+ 2.37e3*T9**(-3/2)*exp(-3.011*T9**(-1)) + 2.19e4*exp(-12.53*T9**(-1)))

    lambdas= [l_pp, l_33, l_34, l_e7, l_17_prime, l_17, l_p14]
    # Base reaction
    r_common = n_p**2 * l_pp / (2*rho) #r_pp
    # PPI
    r_33 = n_He3**2 * l_33 / (2*rho)
    # PPII
    r_34 = n_He3*n_He4 * l_34 / rho
    r_e7 = n_e*n_7Be * l_e7 / rho
    r_17_prime = n_p*n_7Li * l_17_prime / rho
    # PPIII
    r_17 = n_p*n_7Be * l_17 / rho
    # CNO
    r_p14 = n_p*n_14N * l_p14/rho

    # Mass differences for each reaction
    dm_pp = (m_p + m_p + m_p) - (m_3He)
    dm_33 = ((m_3He + m_3He) - (m_4He + 2*m_p))
    dm_34 = (m_3He + m_4He) - m_7Be
    dm_e7 = m_7Be - m_7Li
    dm_17_prime = (m_7Li + m_p) - 2*m_4He
    dm_17 = (m_7Be + m_p) - 2*m_4He
    dm_p14 = (4*m_p) - m_4He
    dm= [dm_pp, dm_33, dm_34, dm_e7, dm_17_prime, dm_17, dm_p14 ]

    # Rescale reaction rates such that no reaction uses more mass than is produced by the preceding reaction
    # If the sum of the reactions rates producing He3 and He4 is larger than the
    # reaction rate that produces H, there will not be enough He4 for there to be
    # a reaction between He3 and He4 producing Be7. We therefore need to scale the
    # reaction rate of r_33, producing He4

    if r_common < (2*r_33 + r_34):
        fix_ratio = r_common / (2*r_33 + r_34)
        r_33= fix_ratio * r_33
        r_34 = fix_ratio * r_34

    # Now we consider the sum of the reaction rates of r_7e and r_71. Both of the
    # reactions require Be7 to produce respectively Li7 and B8. We therefore scale
    # the reaction rate r_34, that produces Be7.
    if r_34 < (r_e7 + r_17):
        scale = r_34/(r_e7 + r_17)
        r_e7 = scale * r_e7
        r_17 = scale * r_17

    # r_71_prime requires a Li7, which is dependant on r_7e.
    if r_e7 < r_17_prime:
        scale = r_e7/r_17_prime
        r_17_prime = scale* r_17_prime

    reactionrates = [r_common, r_33, r_34, r_e7, r_17_prime, r_17, r_p14 ]

    # Array for storing energy production from each reaction
    Qs = []
    neutrino_loss = np.zeros(4)
    chain_energies = np.zeros(4)
    c2= c*c
    Qs = np.multiply(dm,c2)
    Q = np.subtract(Qs,Evs)
    Q_common = Q[0]

    # Energy production
    E = []
    E = np.multiply(Q,reactionrates)
    energies = [(E[0]+E[1])*rho, E[1]*rho, E[2]*rho, E[3]*rho, E[4]*rho, E[5]*rho, E[6]*rho]

    # Chain energies
    E_PP1 = (2*Q_common + Q[1])*r_33
    E_PP2 = (Q_common + Q[2])*r_34 + Q[3]* r_e7 + Q[4]* r_17_prime
    E_PP3 = (Q_common + Q[2])*r_34 + Q[5]* r_17
    CNO = Q[6] * r_p14
    chain_energies = [E_PP1,E_PP2, E_PP3, CNO]

    # Neutrino losses
    neutrino_loss[0] = 2*Evs[0]/(2*Qs[0] + Qs[1])
    neutrino_loss[1] = (Evs[0] + Evs[3])/(Qs[0] + np.sum(Qs[2:5]))
    neutrino_loss[2] = (Evs[0] + Evs[5])/(Qs[0] + Qs[2] + Qs[5])
    neutrino_loss[3] = Evs[6]/Qs[6]
    neutrino_loss *= 100

    if sanity == "Chain":
        run_once = 0
        while run_once == 0:
            print("Chain energies and Neutrino losses")
            print()
            print(f" Chain energies                     Neutrino loss(%)\n\
 -----------------------------------------------------\n\
 E_PP1 {chain_energies[0]}       {neutrino_loss[0]} \n E_PP2 {chain_energies[1]}        {neutrino_loss[1]} \n E_PP3 {chain_energies[2]}       {neutrino_loss[2]} \n CNO   {chain_energies[3]}        {neutrino_loss[3]}")
            run_once = 1

    # Sanities
    sanities = np.array([4.04e2, 8.68e-9, 4.86e-5, 1.49e-6, 5.29e-4, 1.63e-6, 9.18e-8])
    if sanity == True:
        errors = abs(sanities-energies)/sanities
        header_rho = 'rho = {:.2e}'.format(rho)
        header_T = 'T = {:.2e}'.format(T)
        print('{:^35s}'.format(header_rho))
        print('{:^35s}'.format(header_T))
        print()
        print("Energy production: PP branches and CNO cycle")
        print('{:^11s}|{:^11s}|{:^11s}'.format('Energies','Sanity','Relative error'))
        print(35*'-')
        for i,j,k in zip(energies,sanities,errors):
            print('{:^11.3e}|{:^11.3e}|{:^11.6f}'.format(i,j,k))

        return errors
    else:
        return chain_energies

# Solar Core Parameters
rho = 1.62e5 # density [kg/m^3]
T = 1.57e7 # temperature [K]

results = stellar_engines(rho, T, sanity = True) #Set sanity = "Chain" to also return chain_energies

df = pd.DataFrame({'relative_energies': [], 'T': [], 'Reaction': []})
T = np.zeros(10000)
r = range(10**4, 10**9,10**5)

for i in range(10000):
    T[i] = r[i] # temperature [K]
    results = stellar_engines(rho, T[i], sanity = False)
    Total_energy = np.sum(results)
    relative_energies = np.divide(results,Total_energy)
    tmp = pd.DataFrame({'relative_energies':relative_energies, 'T':np.repeat(T[i],4), 'Reaction':['PP1','PP2','PP3','CNO']})
    df = df.append(tmp)
#Plot
sns.set(font_scale=2)
sns.set_style("white")
fig, ax = plt.subplots(figsize=(11.7, 8.27))
ax.set(xscale = "log")
ax.set(title = "Relative Energy Production as Temperature function: Reactions in Log scale")
g1 = sns.lineplot(x = "T", y = "relative_energies", hue = "Reaction", data = df)
ax.set(ylabel = 'Relative E [$J$ $kg^{-1} s^{-1}]$', xlabel = 'Temperature [$K$]')
g1.legend(loc = 'lower right', bbox_to_anchor = (1.25, 0.5), ncol=1)
# plt.savefig('energy_output.pdf', bbox_inches = 'tight')
plt.show()

"""
Energy production: PP branches and CNO cycle
 Energies  |  Sanity   |Relative error
-----------------------------------
 4.048e+02 | 4.040e+02 | 0.001971
 8.608e-09 | 8.680e-09 | 0.008270
 4.840e-05 | 4.860e-05 | 0.004074
 1.423e-06 | 1.490e-06 | 0.044863
 5.266e-04 | 5.290e-04 | 0.004618
 1.619e-06 | 1.630e-06 | 0.006671
 9.180e-08 | 9.180e-08 | 0.000042

(plot)
"""
