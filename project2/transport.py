"Project 2"

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from scipy import interpolate as inter
import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


# General physics constants
k = 1.38065e-23           # Boltzmann constant [m^2 kg s^-2 K^-1]
sigma = 5.670374419e-8    # Stefan-Boltzmann constant [W m^-2 K^-4]
c = 299792458             # speed of light [m s^-1]
m_u = 1.66053904e-27      # atomic mass unit [kg]
G = 6.67430e-11           # Gravitational constant
mu = 0.618237             # Mean molecular weight "dimensionless"
Cp = 5/2 * k/(mu*m_u)     # specific heat capacity
mum = 1.66053904e-27      # atomic mass unit [kg]
N_A = 6.0221e23           # Avogadro's number
MeV = 1.60218*10**(-13)   # Joule per MeV

# Initial stellar conditions in SI-units
L_Sun =  3.846e26          # solar luminosity [w]
R_Sun = 6.96e8             # solar radius [m]
M_Sun = 1.9891e30          # solar mass [kg]

# Initial parameters for bottom of sun's convection zone
L0 = L_Sun # fixed
M0 = M_Sun # fixed
R0 = R_Sun
rho0 = 1408*1.42e-7       # solar average density [kg m^-3]
T0 = 5770                 # solar temperature [K]

exper_P0 = 0 # used later for experimenting on pressure

"Energy"
# Number densities as a function of density
def npr(rho):
    return rho* 0.7/(mum)

def nhe4(rho):
    return rho* 0.29/(4*mum)

def nhe3(rho):
    return (rho*10**(-10))/(3*mum)

def nli(rho):
    return (rho*10**(-7))/(7*mum)

def nbe(rho):
    return (rho*10**(-7))/(7*mum)

def nni(rho):
    return (rho*10**(-11))/(14*mum)

def ne(rho):
    return rho/(2*(mum))*(1+0.7)

# Reaction rates based on number densities
def r_ik(ni,nk, rho, lam, chk):
    if chk == True:
        rate = ni * nk /(rho * (2)) * lam
    else:
        rate = ni * nk / (rho) * lam
    return rate

# Total Energy
def epsilon(T,rho):
    #fractional abundances by weight
    X = 0.7      # H
    Y_3 = 1e-10  # He_3
    Y = 0.29     # He_3 + He_4
    Z = 0.01     # metals
    Z_Li = 1e-13 # Li
    Z_Be = 1e-13 # Be
    mu = 1./(2.*X + 3.*Y/4. + Z/2.)

    Q_pp = (0.15 + 1.02)*MeV # [J]
    Q_Dp = 5.49*MeV
    Q_33 = 12.86*MeV
    Q_34 = 1.59*MeV
    Q_7e = 0.05*MeV
    Q_71_ = 17.35*MeV
    Q_71 = (0.14 + 1.02 + 6.88 + 3.0)*MeV

    T_9 = T*10**(-9.) # temperature conversion to units 10^9K
    T_9_ = T_9/(1+4.95*10**(-2.)*T_9)
    T_9__ = T_9/(1+0.759*T_9)

    N_A_Lamb_pp = 4.01*10**(-15.)*T_9**(-2./3.)*exp(-3.38*T_9**(-1./3.))*(1. + 0.123*T_9**(1./3.) + 1.09*T_9**(2./3.) + 0.938*T_9)
    N_A_Lamb_33 = 6.04e10*T_9**(-2./3.)*exp(-12.276*T_9**(-1./3.))*(1+0.034*T_9**(1./3.) - 0.522*T_9**(2./3.) - 0.124*T_9 + 0.353*T_9**(4./3.) + 0.213*T_9**(-5./3.))
    N_A_Lamb_34 = 5.61e6*T_9_**(5./6.)*T_9**(-3./2.)*exp(-12.826*T_9_**(-1./3.))
    N_A_Lamb_7e = 1.34*10**(-10.)*T_9**(-1./2.)*(1. - 0.537*T_9**(1./3.) + 3.86*T_9**(2./3.) + 0.0027*T_9**(-1.)*exp(2.515*10**(-3.)*T_9**(-1.)))
    N_A_Lamb_71_ = 1.096e9*T_9**(-2./3.)*exp(-8.472*T_9**(-1./3.)) - 4.83e8*T_9__**(5./6.)*T_9**(-3./2.)*exp(-8.472*T_9__**(-1./3.)) + 1.06e10*T_9**(-3./2.)*exp(-30.442*T_9**(-1.))
    N_A_Lamb_71 = 3.11e5*T_9**(-2./3.)*exp(-10.262*T_9**(-1./3.)) + 2.53e3*T_9**(-3./2.)*exp(-7.306*T_9**(-1.))

    n_p = rho*X/(1.*m_u)
    n_He = rho*Y/(4.*m_u) # He_4 and He_3
    n_He_3 = rho*Y_3/(3.*m_u)
    n_He_4 = n_He - n_He_3
    n_Be = rho*Z_Be/(7.*m_u)
    n_Li = rho*Z_Li/(7.*m_u)
    n_e = n_p + 2.*n_He_3 + 2.*n_He_4 + 2.*n_Be + 1.*n_Li

    Lamb_pp = 1e-6*N_A_Lamb_pp/N_A #[m^3/s]
    Lamb_33 = 1e-6*N_A_Lamb_33/N_A
    Lamb_34= 1e-6*N_A_Lamb_34/N_A
    if T < 1e6:
        if N_A_Lamb_7e > 1.57e-7/n_e:
            N_A_Lamb_7e = 1.57e-7/n_e
    Lamb_7e = 1e-6*N_A_Lamb_7e/N_A   # Be + e
    Lamb_71_ = 1e-6*N_A_Lamb_71_/N_A # Li + H
    Lamb_71 = 1e-6*N_A_Lamb_71/N_A   # Be + H

    r_pp = n_p*n_p*Lamb_pp/(rho*2.)
    r_33 = n_He_3*n_He_3*Lamb_33/(rho*2.)
    r_34 = n_He_3*n_He_4*Lamb_34/rho
    if r_pp < (r_33*2. + r_34):
        rate1 = r_pp/(2.*r_33 + r_34)
        r_33 *= rate1
        r_34 *= rate1

    r_7e = n_Be*n_e*Lamb_7e/rho
    r_71_ = n_Li*n_p*Lamb_71_/rho
    r_71 = n_Be*n_p*Lamb_71/rho
    if r_34 < (r_7e + r_71):
        rate2 = r_34/(r_7e + r_71)
        r_7e *= rate2
        r_71 *= rate2
    if r_7e < r_71_:
        rate3 = r_7e/r_71_
        r_71_ *= rate3

    # total energy production
    eps = r_pp*(Q_pp + Q_Dp) + r_33*Q_33 + r_34*Q_34 + r_7e*Q_7e + r_71_*Q_71_ + r_71*Q_71
    # PP1 branch
    eps1 = (2.*(Q_pp + Q_Dp) + Q_33)*r_33
    # PP2 branch
    eps2 = (eps1 + Q_pp + Q_Dp + Q_34)*r_34 + (eps1 + Q_pp + Q_Dp + Q_34 + Q_7e)*r_7e + (eps1 + Q_pp + Q_Dp + Q_34 + Q_7e + Q_71_)*r_71_
    # PP3 branch
    eps3 = (eps1 + Q_pp + Q_Dp + Q_34)*r_34 + (eps1 + Q_pp + Q_Dp + Q_34 + Q_71)*r_71
    return [eps, eps1/(eps1 +eps2 +eps3), eps2/(eps1 +eps2 +eps3), eps3/(eps1 +eps2 +eps3)]



"Read and process the opacity txt file"
def ReadFile(file):
    data = open(file ,'r')
    lines = data.readlines()
    n = np.int((np.shape(lines)[0]))
    tlog = np.zeros(n)
    opacity = [i for i in range(0,n)]
    errordata = np.zeros(n)

    for i, line in enumerate(lines):
        if i == 0:
            R = np.asarray(line.split()[1:])
        if i != 1 and i != 0:
            tlog[i] = line.split()[0]
            opacity[i]  = line.split()[1:]

    return R.astype(float), tlog[2:].astype(float),np.asarray(opacity[2:]).astype(float)

LogR, LogT, LogK = ReadFile("opacity.txt")

# Interpolates the values and returns a warning if out of bounds
def interpolate(R,T,k):
    R, T = np.meshgrid(R,T)
    return inter.interp2d(R,T,k, kind = "linear")

def kappaSI():
    interpolated = interpolate(LogR,LogT,LogK)
    return interpolated

def pressure(rho,T):
    Pg = rho*k*T/(mu*m_u)
    Prad = 4*sigma/(c*3) * T**4
    return Pg + Prad

# Calculates/updates g as the simulation evolves
def g(M,R):
    return 6.67430e-11 * M /(R**2)

def Hp(P,g,rho):
    return P/(g*rho)

def density(P,T):
    density = ((mu * m_u)/(k * T)) * (P -(4*sigma*T**4)/(3*c))
    return density

def u(T):
    u = 3/2 * 1/(mu*m_u)*K*T
    return u



"Differential equations"
def drdm(r,rho):
    return (1/(4*np.pi*r**2 *rho))

def dPdm(r,m):
    return -(6.67430e-11 * m)/(4*np.pi*r**4)

def dLdm(epsilon):
    return epsilon

def dTdm(kap,L,r,T):
    return - (3*kap * L)/(256*np.pi**2*sigma*r**4*T**3)

def dm_calc(step,derive):
    p = 0.01 # allowed fraction of change
    dm_ = step / derive * p
    return dm_



"Calculation of gradients and fluxes"
def grad_star(nabs,nabad,rho,T,g,M,R,hp,kappa,alpha):
    # * gradient
    lm = alpha * hp
    U = (64*sigma*T**3)/(3*kappa* rho**2 *Cp)*np.sqrt(hp/(g))
    coeffs = np.asarray((1,U/lm**2 ,4*U**2/lm**4 ,-U*(nabs-nabad)/lm**2))
    xi = np.roots(coeffs)
    # Extracting the relevant root of the polynomial
    xi = xi[np.isreal(xi)]
    xi = np.real(xi)

    return nabs - (lm**2)*(xi**3) /U

def grad_stable(L, rho,T, kappa, R,hp):
    # Stable gradient
    return 3*L*kappa*rho*hp/(64*np.pi*(R**2) * sigma * T**4)

def grad_adiab():
    # Constant adiabatic gradient
    return 2/5

def conv_flux(grad_stable,grad_star,T,kappa,rho,hp):
    # Convective flux
    return (grad_stable -grad_star)*(16*sigma*T**4)/(3*kappa*rho*hp)

def rad_flux(T,kappa,rho,hp,nabstar):
    # Radiative flux
    return (16*sigma*T**4)/(3*kappa*rho*hp)*nabstar



"Solving the differential equations by using the Euler-forward method"
def polate(T, rho, Sanity = False):
    # Interpolates or extrapolates in case of values outside boundaries
    if Sanity == False:
        x = np.log10(T)
        y = np.log10(rho/(1000*(T*10**(-6))**3))
    else:
        x = T
        y = rho

    Kappa = LogK

    b = np.zeros(2)
    c = np.zeros(2)

    if x > LogT[-1]:
        numb = -2
        b[0], b[1] = LogT[-2], LogT[-1]
    elif x < LogT[0]:
        numb = 0
        b[0], b[1] = LogT[1], LogT[0]
    else:
        a = LogT[LogT > x][0]
        numb = np.where(a == LogT)[0][0] - 1
        b[0], b[1] = a, LogT[numb]

    if y > LogR[-1]:
        numb2 = -2
        c[0], c[1] = LogR[-2], LogR[-1]
    elif y < LogR[0]:
        numb2 = 0,
        c[0], c[1] = LogR[1], LogR[0]
    else:
        a = LogR[LogR > y][0]
        numb2 = np.where(a == LogR)[0][0] - 1
        c[0], c[1] = LogR[numb2], a

    # Linear method for interpolation
    f = ( Kappa[numb + 1, numb2] * (c[1]-y) * (b[1]-x) \
        + Kappa[numb + 1, numb2 + 1] * (y-c[0]) * (b[1]-x) \
        + Kappa[numb, numb2] * (c[1]-y) * (x-b[0]) \
        + Kappa[numb, numb2 + 1] * (y-c[0]) * (x-b[0]) ) \
          / ( (c[1]-c[0]) * (b[1]-b[0]) )

    inter = 10 ** f * 0.1 if Sanity == False else f
    return inter

def solver():
    # Initialising the differential equations
    P0, M, rho, R, T, L, kappa = pressure(rho0,T0), [M0], [rho0], [R0], [T0], [L0], [polate(T0,rho0)]

    # experimenting with pressure
    if exper_P0 == 0:
        P = [P0]
    elif exper_P0 ==1:
        P = [0.8*P0]
    elif exper_P0 ==2:
        P = [4*P0]

    e = []; e1= []; e2 = []; e3= []
    e_ = epsilon(T0, rho0)
    e.append(e_[0]); e1.append(e_[1]); e2.append(e_[2]); e3.append(e_[3])
    g_0 = g(M0,R0)
    hp_0 = Hp(P0,g_0,rho0)

    gradstab =[grad_stable(L0,rho0,T0,kappa[0],R0,hp_0)]
    gradst = [grad_star(gradstab[0],grad_adiab(),rho0,T0,g_0,M0,R0,hp_0,kappa[0],1)]
    conf = [conv_flux(gradstab[0],gradst[0],T0,kappa[0],rho0,hp_0)]
    radf =[rad_flux(T0,kappa[0],rho[0],hp_0,gradst[0])]

    i = 0
    while M[i] > 0:
        #Solving the differential equations using the euler method
        #Calculating the values for the tangent for this step
        kappa.append(polate(T[i],rho[i]))
        gr  = g(M[i],R[i])
        hpi = Hp(P[i],gr,rho[i])
        R1 = drdm(R[i],rho[i])
        P1 = dPdm(R[i],M[i])
        L1 = dLdm(epsilon(T[i],rho[i])[0])
        gradstab.append(grad_stable(L[i],rho[i],float(T[i]),kappa[i],R[i],hpi))
        if grad_adiab() < gradstab[-1]:
            gradst.append(grad_star(gradstab[-1],grad_adiab(),rho[i],T[i],gr,M[i],R[i],hpi,kappa[i],1))
        else:
            gradst.append(gradstab[-1])

        coff =-3/16 * kappa[i]*rho[i]/(sigma*T[i]**3)*1/(4*np.pi*R[i]**2*rho[i])
        T1 = coff*rad_flux(T[i],kappa[i],rho[i],hpi,gradst[i+1])
        radf.append(rad_flux(T[i],kappa[i],rho[i],hpi,gradst[i]))
        conf.append(conv_flux(gradstab[-1],gradst[-1],T0,kappa[i],rho[i],hpi))

        # Calculating dm
        dm1 = abs(dm_calc(R[i-1],R1))
        dm2 = abs(dm_calc(P[i-1],P1))
        dm3 = abs(dm_calc(L[i-1],L1))
        dm4 = abs(dm_calc(T[i-1],T1))
        dm = min([dm1,dm2,dm3,dm4])

        # Adding the new step to the solution
        R.append(R[i] - R1*dm)
        P.append(P[i] - P1*dm)
        L.append(L[i] - L1*dm)
        T.append(T[i] - T1*dm)
        rho.append(density(P[i+1],T[i+1]))
        M.append(M[i] - dm)
        ee_ = epsilon(T[i],rho[i])
        e.append(ee_[0]); e1.append(ee_[1]); e2.append(ee_[2]); e3.append(ee_[3])

        i = i + 1
        # Setting a few conditions on the end values of the calculations
        if M[i]-dm < 0:
            M[i] = 0
        elif len(M) >= 30000:
            M[i] = 0

    # Solved values
    return R, P, L, T, M, rho, kappa, conf, gradst, gradstab, radf, e, e1, e2, e3



"Tests"
# Sanity check for certain given values of logT and logR
def KappaSanity():

    #extracting values from table
    logR, logT, logK = ReadFile("opacity.txt")

    logT_test = [3.750, 3.755, 3.755, 3.755, 3.755, 3.770, 3.780, 3.795, 3.770, 3.775, 3.780, 3.795, 3.800]
    logR_test = [-6.00, -5.95, -5.80, -5.70, -5.55, -5.95, -5.95, -5.95, -5.80, -5.75, -5.70, -5.55, -5.50]
    Kappa_test = np.zeros(len(logR_test))

    f = inter.interp2d(logR, logT, logK)

    #calculating the interpolated kappa values for the given test logT and logR
    for i in range(len(logR_test)):

        Kappa_test[i] = f(logR_test[i], logT_test[i])

    #printing the interpolated values which can be compared with the sanity check
    sanity_check = [2.84e-3, 3.11e-3, 2.68e-3, 2.46e-3, 2.12e-3, 4.70e-3, 6.25e-3, 9.45e-3, 4.05e-3, 4.43e-3, 4.94e-3, 6.86e-3, 7.69e-3]

    print('{:^10s} {:^10} {:^10s} {:^10s} {:^10s}'.format('log10T    ',' log10R','log10κ','   κ Sanity value ', 'Relative error'))
    print("-"*66)
    for i in range(13):
        print(f"{logT_test[i]:<10.3f} {logR_test[i]:^10.2f} {(10**Kappa_test[i])/10:^12.2e} {sanity_check[i]:^11.2e} {abs((10**Kappa_test[i])/10-sanity_check[i]):13.6f}")
    return None

# Sanity check for values and gradients
def gradtest():
    expected = np.array([3.26,32.5e6,5.94e5,1.175e-3,0.4,65.62,0.88,0.12])

    #Initialising relevant initial conditions
    L =  3.828e26
    R = 696340000
    M = 1.9891e30
    T = 0.9e6
    rho = 55.9
    R = 0.84*R
    M = 0.99*M
    gr = g(M,R)
    lum = L
    kappa = 3.98
    nabad = 2/5
    alpha = 1

    hp = Hp(pressure(rho,T),gr,rho)
    U = (64*sigma*T**3)/(3*kappa* rho**2 *Cp)*np.sqrt(hp/(gr))
    lm = alpha * hp
    Q = np.pi * lm
    S = 2*np.pi *lm**2
    d = lm
    nabs = grad_stable(lum,rho,T,kappa,R, hp)
    coeffs = np.asarray((lm**2/U , 1 ,U*S/(Q*d*lm ),-(3.26-nabad)))
    roots = np.roots(coeffs)
    xi = roots[np.isreal(roots)]
    xi = float(np.real(xi))
    v = np.sqrt((gr*lm**2)/(4*hp))*xi
    nabstar = float(grad_star(3.26, grad_adiab(),rho,T,gr,M,R,hp,kappa,1))
    con = conv_flux(nabs,nabstar,T,kappa,rho,hp)
    rad = rad_flux(T,kappa,rho,hp,nabstar)
    rcon = con/(con+rad)
    rrad = rad/(con+rad)

    calculated = np.array([nabs, hp, U, xi, nabstar, v, rcon, rrad])
    errors = np.array(abs(expected-calculated)/expected)
    names = np.array(['Nabla stable', 'Hp', 'U', 'Xi', 'Nabla star', 'v', 'Conv. flux', 'Rad. flux'])

    print('{:^5s} {:^15s} {:^11s} {:^16s}'.format('Value','        Calculated','Expected','Relative error'))
    print("----------------------------------------------------")
    for i in range(8):
        print(f"{names[i]:<12} {calculated[i]:^11.3e} {expected[i]:^11.3e} {errors[i]:^11.5f}")
    print()
    return None



"Plot cross section of star"
def cross_section(R, L, F_C, show_every=20):
    """
    R: radius, array
    L: luminosity, array
    F_C: convective flux, array
    show_every: plot every <show_every> steps
    """

    R_sun = 6.96e8      # [m]
    L_sun = 3.846e26    # [W]

    plt.figure(figsize=(800/100, 800/100))
    fig = plt.gcf()
    ax  = plt.gca()
    # experimenting with range
    r_range = 1.2*R[0]/R_sun # this!
#     r_range = 1.8*R[0]/R_sun
#     r_range = 0.6*R[0]/R_sun
    rmax = np.max(R)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    core_limit = 0.995*L_sun

    j = 0
    for k in range(0,len(R)-1):
        j += 1
        # plot every <show_every> steps
        if j%show_every == 0:
            if L[k] >= core_limit:     # outside core
                if F_C[k] > 0.0:       # plot convection outside core
                    circle_red = plt.Circle((0,0), R[k]/rmax, color = 'red', alpha=0.8, fill=False)
                    ax.add_artist(circle_red)
                else:                  # plot radiation outside core
                    circle_yellow = plt.Circle((0,0), R[k]/rmax, color = 'yellow', alpha=0.9, fill=False)
                    ax.add_artist(circle_yellow)
            else:                      # inside core
                if F_C[k] > 0.0:       # plot convection inside core
                    circle_navy = plt.Circle((0,0), R[k]/rmax, color = 'navy', alpha=0.9, fill=False)
                    ax.add_artist(circle_navy)
                else:                  # plot radiation inside core
                    circle_cyan = plt.Circle((0,0), R[k]/rmax, color = 'cyan', alpha=0.7, fill=False)
                    ax.add_artist(circle_cyan)

    # legends
    circle_red    = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color = 'red', fill = True)
    circle_yellow = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color = 'yellow', fill = True)
    circle_blue   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color = 'blue', fill = True)
    circle_cyan   = plt.Circle((2*r_range, 2*r_range), 0.1*r_range, color = 'cyan', fill = True)

    ax.legend([circle_red,circle_yellow,circle_cyan,circle_blue], \
        ['Convection outside core','Radiation outside core','Radiation inside core','Convection inside core'], \
        fontsize=13)
    plt.xlabel(r'$R/R_0$', fontsize=13)
    plt.ylabel(r'$R/R_0$', fontsize=13)

def plotting():
    plt.figure(1)
    plt.plot(R/R0, M, c = "royalblue")
    plt.xlabel("$R/R_{\odot}$")
    plt.ylabel("M [kg]")
    plt.title("Mass vs. radius", fontsize=15)
#     plt.savefig("mass_vs_radius.pdf")
#     plt.savefig("mass2.pdf")

    plt.figure(2)
    plt.axhline(0.995*L0, label = r"$0.995 \cdot L_0$",c = "r")
    plt.axhline(0.05*L0, label = r"$0.05 \cdot L_0$",c = "mediumturquoise")
    plt.plot(R/R0,L,label = "luminosity", c = "royalblue")
    plt.title("Luminosity vs. radius", fontsize=15)
    plt.xlabel("$R/R_{\odot}$")
    plt.ylabel("L [J]")
    plt.legend()
#     plt.savefig("luminicity_vs_radius.pdf")
#     plt.savefig("lum2.pdf")

    plt.figure(3)
    plt.yscale("log")
    plt.plot(R/R0, P, c = "royalblue")
    plt.title("Pressure vs. radius", fontsize=15)
    plt.xlabel("$R/R_{\odot}$")
    plt.ylabel("$P [Pa]$")
#     plt.savefig("pressure_vs_radius.pdf")
#     plt.savefig("press2.pdf")

    plt.figure(4)
    plt.yscale("log")
    plt.plot(R/R0, rho, c = "royalblue")
    plt.title("Density vs. radius", fontsize=15)
    plt.xlabel("$R/R_{\odot}$")
    plt.ylabel("$ \rho$ $[kg/m^3]$")
#     plt.savefig("density_vs_radius.pdf")
#     plt.savefig("dens2.pdf")

    plt.figure(5)
    plt.plot(R/R0,np.asarray(conf)/(np.asarray(conf)+np.asarray(radf)), c = "crimson", label = "Relative convective flux")
    plt.plot(R/R0,np.asarray(radf)/(np.asarray(conf)+np.asarray(radf)), c = "royalblue", label = "Relative radiative flux")
    plt.xlabel("$R/R_{\odot}$")
    plt.ylabel("Relative flux")
    plt.title("Relative flux vs radius", fontsize=15)
    plt.legend()
    if L0 == L_Sun and R0 == R_Sun and rho0 == 1408*1.42e-7 and T0 == 5770:
        plt.yscale("log")
#         plt.savefig("relative_flux_vs_radius.pdf")
#     plt.savefig("flux2.pdf")

    plt.figure(6)
    plt.xlabel("$R/R_{\odot}$")
    plt.ylabel("Gradients")
    plt.yscale("log")
    plt.axhline(grad_adiab(), label = r"$\nabla_{ad}$", c ="crimson")
    plt.plot(R/R0,gradstab, label = r"$\nabla_{stable}$", c="lavender")
    plt.plot(R/R0,grads, label = r"$\nabla^*$", c = "royalblue")
    plt.title("Temperature Gradients vs. radius", fontsize=15)
    plt.legend()
#     plt.savefig("temperature_gradients_vs_radius.pdf")
#     plt.savefig("tempgrad2.pdf")

    plt.figure(7)
    plt.title("Kappa vs. radius", fontsize=15)
    plt.plot(R/R0, kappa, c = "royalblue")
    plt.xlabel("$R/R_{\odot}$")
    plt.ylabel("Kappa")
#     plt.savefig("Kappa_vs_radius.pdf")
#     plt.savefig("kapp2.pdf")

    plt.figure(8)
    plt.plot(R/R0, T, c = "royalblue")
    plt.title("Temperature vs. radius", fontsize=15)
    plt.xlabel("$R/R_{\odot}$")
    plt.ylabel("T [K]")
#     plt.savefig("temperature_vs_radius.pdf")
#     plt.savefig("temprad2.pdf")

    plt.figure(9)
    plt.title('Relative energy production', fontsize=15)
    plt.xlabel('$R/R_{\odot}$', fontsize=13)
    plt.ylabel('$\\varepsilon$', fontsize=13)
    plt.plot(R/R_Sun, e1, c = "royalblue", label = '$\\varepsilon_{PPI}  / \\varepsilon_{tot}$')
    plt.plot(R/R_Sun, e2, c = "crimson", label = '$\\varepsilon_{PPII}  / \\varepsilon_{tot}$')
    plt.plot(R/R_Sun, e3, c ='lavender', label = '$\\varepsilon_{PPIII}  / \\varepsilon_{tot}$')
    plt.plot(R/R_Sun, e/max(e), c = 'mediumturquoise', label = '$\\varepsilon / \\varepsilon_{max}$')
    plt.legend()
#     plt.savefig("initial_energy_production.pdf")
#     plt.savefig("energy_production.pdf")
    plt.show()

def cross():
    # used in experimentation plotting of various cross sections
    R, P, L, T, M, rho, kappa, conf, grads, gradstab, radf, e, e1, e2, e3 = solver()
    R, P, L, T, M, rho, kappa = np.asarray(R), np.asarray(P), np.asarray(L), np.asarray(T), \
        np.asarray(M), np.asarray(rho), np.asarray(kappa)
    cross_section(R,L,conf,show_every=20)


if __name__ == "__main__":

    inp1 = input(" Enable sanity checks? [y/n]: \n ")
    while True:
        try:
            if inp1 == "y":
                print()
                print(f" Sanity Cross section of the star with initial parameters:\n \
---------------------------------------------------------\n \
Initial Density:      {rho0:6.4e} [kg/m^3]\n \
Initial Luminosity:   {L0:6.4e} [W/m^2]\n \
Initial Mass:         {M0:6.4e} [kg]\n \
Initial Temperature:  {T0:6.4e} [K]\n \
Initial Radius:       {R0:6.4e} [m]")

                # Cross_section(M, R, L, Flux_con)
                R, P, L, T, M, rho, kappa, conf, grads, gradstab, radf, e, e1, e2, e3 = solver()
                R, P, L, T, M, rho, kappa = np.asarray(R), np.asarray(P), np.asarray(L), np.asarray(T), \
                np.asarray(M), np.asarray(rho), np.asarray(kappa)
                cross_section(R,L,conf,show_every=5) # sanity cross_section
                plt.title('Sanity Cross section of star', fontsize=15)
#                 plt.savefig("Sanity_cross_section.pdf")
                plt.show()
                print()
                print("Final values")
                print("""-------------------------------
Final luminosity:  %8.6e
Final Mass:        %8.6e
Final Radius:      %8.6e
Final temperature: %8.6e
Final density:     %8.6e
"""%(100*(1-L[-1]/L0), 100*(1-M[-2]/M0), 100*(1-R[-1]/R0), T[-1], rho[-1]))

                plotting()
                inp2 = input("\n Do you wish to run the interpolation sanity check? [y/n]: \n ")
                while True:
                    try:
                        if inp2 == "y":
                            print()
                            print("Interpolation Sanity")
                            KappaSanity()
                            inp3 = input("\n Do you wish to run the gradient sanity check? [y/n]: \n ")

                            while True:
                                try:
                                    if inp3 == "y":
                                        print()
                                        print("Gradient sanity")
                                        gradtest()
                                        break
                                    elif inp3 == "n":
                                        print()
                                        print("No gradient sanity check!")
                                        False
                                        break
                                    else:
                                        assert False
                                except AssertionError:
                                    print()
                                    print("No valid gradient check input. No gradient sanity check!")
                                    break

                            break
                        elif inp2 == "n":
                            print()
                            print("No interpolation sanity check!")
                            inp3 = input("\n Do you wish to run the gradient sanity check? [y/n]: \n ")
                            while True:
                                try:
                                    if inp3 == "y":
                                        print()
                                        print("Gradient sanity")
                                        gradtest()
                                        break
                                    elif inp3 == "n":
                                        print()
                                        print("No gradient sanity check!")
                                        False
                                        break
                                    else:
                                        assert False
                                except AssertionError:
                                    print()
                                    print("No valid gradient check input. No gradient sanity check!")
                                    break
                            False
                            break
                        else:
                            assert False
                    except AssertionError:
                        print()
                        print("No valid interpolation check input. No interpolation sanity check!")
                        break
                break

            elif inp1 == "n":
                print()
                print("No sanity checks enabled.")
                break

            else:
                assert False
                break

        except AssertionError:
            print()
            print("No valid sanity check input. Please try again.")
            inp1 = input(" Enable sanity checks? [y/n]: \n ")
            continue
    print()

    # Best model parameters
    R0 = 0.85*R0
    rho0 = 240*rho0 # 3.37920e-5
    T0 = 1.1*T0

    print(f" Cross section of the star with best model parameters:\n \
-----------------------------------------------------\n \
rho: {rho0:6.4e} [kg/m^3]\n \
L:   {L0:6.4e} [W/m^2]\n \
M:   {M0:6.4e} [kg]\n \
T:   {T0:6.4e} [K]\n \
R:   {R0:6.4e} [m]")

    # Cross_section(M, R, L, Flux_con)
    R, P, L, T, M, rho, kappa, conf, grads, gradstab, radf, e, e1, e2, e3 = solver()
    R, P, L, T, M, rho, kappa, e, e1, e2, e3 = np.asarray(R), np.asarray(P), np.asarray(L), np.asarray(T), \
        np.asarray(M), np.asarray(rho), np.asarray(kappa), np.asarray(e), np.asarray(e1), np.asarray(e2), \
        np.asarray(e3)
    cross_section(R,L,conf,show_every=20)
    plt.title("Cross Section of Best Model")
#     plt.savefig("best.pdf")
    plt.show()

    print()
    print("Final values")

    print("""---------------------------------
Final Luminosity (L):  %8.6e
Final Mass (M):        %8.6e
Final Radius (R):      %8.6e
Final Temperature (T): %8.6e
Final Density (rho):   %8.6e
"""%(100*(1-L[-1]/L0), 100*(1-M[-2]/M0), 100*(1-R[-1]/R0), T[-1], rho[-1]))
    plotting()

    inp4 =input("Enable experimentation cross section plots? [y/n]: \n ")
    if inp4 == "y":
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Initial parameters for bottom of sun's convection zone
        L0 = L_Sun # fixed
        M0 = M_Sun # fixed
        R0 = R_Sun
        rho0_Sun = 1408*1.42e-7       # solar average density [kg m^-3]
        T0_Sun = 5770                 # solar temperature [K]

        # Experimenting: Tweaking Initial parameters
        # radius
        L0, M0, R0, rho0, T0 = L_Sun, M_Sun, 0.8*R_Sun, rho0_Sun, T0_Sun
        cross()
        plt.title("R = 0.8$R\odot$", fontsize=15) # R = 0.8R Sun
#         plt.savefig("0.8R.pdf")
        plt.show()

        L0, M0, R0, rho0, T0 = L_Sun, M_Sun, 4*R_Sun, rho0_Sun, T0_Sun
        cross()
        plt.title("R = 4$R\odot$", fontsize=15)   # R = 4R Sun
#         plt.savefig("4R.pdf")
        plt.show()

        # density
        L0, M0, R0, rho0, T0 = L_Sun, M_Sun, R_Sun, 5*rho0_Sun, T0_Sun
        cross()
        plt.title(r'$\rho = 5\overline{\rho}_{\odot}$', fontsize=15) # rho = 5 av. rho Sun
#         plt.savefig("5rho.pdf")
        plt.show()

        L0, M0, R0, rho0, T0 = L_Sun, M_Sun, R_Sun, 300*rho0_Sun, T0_Sun
        cross()
        plt.title(r'$\rho = 300\overline{\rho}_{\odot}$', fontsize=15) # rho = 300 av. rho Sun
#         plt.savefig("300rho.pdf")
        plt.show()

        # temperature
        L0, M0, R0, rho0, T0 = L_Sun, M_Sun, R_Sun, rho0_Sun, 0.8*T0_Sun
        cross()
        plt.title("T = 0.8$T\odot$", fontsize=15) # T = 0.8T Sun
#         plt.savefig("0.8T.pdf")
        plt.show()

        L0, M0, R0, rho0, T0 = L_Sun, M_Sun, R_Sun, rho0_Sun, 2*T0_Sun
        cross()
        plt.title("T = 2$T\odot$", fontsize=15)   # T = 2T Sun
#         plt.savefig("2T.pdf")
        plt.show()

        # pressure
        exper_P0 = 1
        L0, M0, R0, rho0, T0 = L_Sun, M_Sun, R_Sun, rho0_Sun, T0_Sun
        cross()
        plt.title("P = 0.8$P\odot$", fontsize=15)   # T = 2T Sun
#         plt.savefig("0.8P.pdf")
        plt.show()

        exper_P0 = 2
        L0, M0, R0, rho0, T0 = L_Sun, M_Sun, R_Sun, rho0_Sun, T0_Sun
        cross()
        plt.title("P = 4$P\odot$", fontsize=15)   # T = 2T Sun
#         plt.savefig("4P.pdf")
        plt.show()

    else:
        print()
        print("No experimentation cross section plotting enabled.")
