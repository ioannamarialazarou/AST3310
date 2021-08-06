"Project 3"

"""
For generating an mp4 file from an animation, FFmpeg is required.
Windows users can download it from the linked site. In order for
matplotlib to find the FFmpeg executable, you must open the source
code (FVis3.py), and at the top specify the path to the folder that
FFmpeg was downloaded to.
For the latest versions of Ubuntu, FFmpeg should already be installed,
and you can leave the path specification commented out.
"""

# visulaliser
import FVis3 as FVis
# import prerequisite libraries
from numpy import pi as pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil

import Sol_parameters as Sun    # Sun Parameters

mpl.rcdefaults()                # Resetting matplotlib settings

class convection2D:

    # initialization, physical constants
    m_u = 1.66053904e-27        # Unit atomic mass [kg]
    k_B = 1.382e-23             # Boltzmann's constant [m^2 kg s^-2 K^-1]
    G   = 6.672e-11             # Gravitational constant [N m^2 kg^-2]
    mu  = 0.61                  # Mean molecular weight
    g_y = G*Sun.M/Sun.R**2      # Gravitational acceleration

    gamma = 5/3                 # Degrees of freedom parameter
    mymu_kB = mu*m_u/k_B        # Common factor used to reduse number of FLOPS
    nabla_inc = 1e-4            # Pertubation in nabla above adiabatic value
    nabla     = 2/5+nabla_inc   # Temperature gradient for convection

    p = 1e-2                    # Variable time step parameter

    def __init__(self, xmax=12e6, nx=300, ymax=4e6, ny=100, initialise=True, perturb=False):
        # Setting up computational volume
        self.xmax = xmax    # range of x [m]
        self.ymax = ymax    # range of y [m]
        self.nx = nx        # nr cells in x
        self.ny = ny        # nr cells is y

        # 1D vertical array and size of cells, y-direction
        self.y,self.delta_y = np.linspace(0,ymax,ny,retstep=True)

        # Now in horizontal direction
        self.x,self.delta_x = np.linspace(0,xmax,nx,retstep=True)

        if initialise:
            self.initialise(perturb=perturb)
        self.forced_dt = 0  # Tracking number of forced time steps


    def initialise(self,perturb=False,nr=1):
        """
        Initialisation of parameters in 2D arrays

        T: temperature
        P: pressure
        rho: density
        e: internal energy
        u: vertical velocity
        w: horizontal velocity
        perturb: if True apply gaussian perturbation in T
        nr: number of perturbations to apply along x-axis
        """

        # Setting up arrays
        self.T   = np.zeros((self.ny,self.nx))
        self.P   = np.zeros((self.ny,self.nx))
        self.rho = np.zeros((self.ny,self.nx))
        self.e   = np.zeros((self.ny,self.nx))
        self.u   = np.zeros((self.ny,self.nx))  # initialized
        self.w   = np.zeros((self.ny,self.nx))  # initialized

        # Initial values
        beta_0 = Sun.T_photo/self.mymu_kB/self.g_y   # Substitution used in P
        for j in range(0,self.ny):     # looping vertically and filling variables
            depth_term = self.nabla*(self.y[j]-self.ymax)
            self.T[j,:] = Sun.T_photo - self.mymu_kB*self.g_y*depth_term
            self.P[j,:] = Sun.P_photo*((beta_0-depth_term)/beta_0)**(1/self.nabla)

        if perturb: # Perturb: initial temperature
            self.perturbation(nr=nr)
        self.e = self.P/(self.gamma-1)
        self.rho = self.e*(self.gamma-1)*self.mymu_kB/self.T


    def hydro_solver(self):
        "Hydrodynamic equations solver"

        rho = self.rho
        e = self.e
        w = self.w
        u = self.u
        rhow = rho*w
        rhou = rho*u
        P = self.P
        T = self.T

        # Flow directions
        flow = self.flow_directions() # [u_pos,u_neg,w_pos,w_neg]

        # Time-differentials of each primary variable
        # Density
        cent_ddx_u    = self.get_central_x(u)
        cent_ddy_w    = self.get_central_y(w)
        up_u_ddx_rho  = self.get_upwind_x(rho,flow[0],flow[1])
        up_w_ddy_rho  = self.get_upwind_y(rho,flow[2],flow[3])
        self.ddt_rho  = - rho*(cent_ddx_u+cent_ddy_w) - u*up_u_ddx_rho - w*up_w_ddy_rho

        # Horizontal momentum
        up_u_ddx_u    = self.get_upwind_x(u,flow[0],flow[1])
        up_u_ddy_w    = self.get_upwind_y(w,flow[0],flow[1])
        up_u_ddx_rhou = self.get_upwind_x(rhou,flow[0],flow[1])
        up_w_ddy_rhou = self.get_upwind_y(rhou,flow[2],flow[3])
        cent_ddx_P    = self.get_central_x(P)
        self.ddt_rhou = - rhou*(up_u_ddx_u + up_u_ddy_w) - u*up_u_ddx_rhou - w*up_w_ddy_rhou - cent_ddx_P

        # Vertical momentum
        up_w_ddy_w    = self.get_upwind_y(w,flow[2],flow[3])
        up_w_ddx_u    = self.get_upwind_x(u,flow[2],flow[3])
        up_w_ddy_rhow = self.get_upwind_y(rhow,flow[2],flow[3])
        up_u_ddx_rhow = self.get_upwind_x(rhow,flow[0],flow[1])
        cent_ddy_P    = self.get_central_y(P)
        self.ddt_rhow = - rhow*(up_w_ddy_w + up_w_ddx_u) - w*(up_w_ddy_rhow) - u*up_u_ddx_rhow - cent_ddy_P - rho*self.g_y

        # Energy
        up_u_ddx_e = self.get_upwind_x(e,flow[0],flow[1])
        up_w_ddy_e = self.get_upwind_y(e,flow[2],flow[3])
        self.ddt_e = - e*(cent_ddx_u+cent_ddy_w) - u*up_u_ddx_e - w*up_w_ddy_e - P*(cent_ddx_u + cent_ddy_w)

        # Finding optimal dt and evolving primary variables
        dt = self.timestep()
        self.rho[:,:] = rho + self.ddt_rho*dt
        self.e[:,:]   = e + self.ddt_e*dt
        self.u[:,:]   = (rhou+self.ddt_rhou*dt)/self.rho[:,:]
        self.w[:,:]   = (rhow+self.ddt_rhow*dt)/self.rho[:,:]

        # Boundary conditions (before calculation of temperature, pressure)
        self.boundary_conditions()
        self.P[:,:] = (self.gamma-1)*e
        self.T[:,:] = (self.gamma-1)*self.mymu_kB*e/self.rho[:,:]

        return dt


    def perturbation(self,nr=1):
        """
        Create gaussian perturbations in the initial temperature distribution
        Adds perturbation to the initial temperature and changes sign if more than one blob.

        nr: number of perturbation blobs (default nr = 1)
        """
        nr *= 2
        x_pos = np.arange(1,nr,2)/nr      # x position of blobs
        alt_sign = np.ones(ceil(nr/2))    # array with alternating signs to get +/- perturbations

        if nr != 2:                       # only if more than one blob
            alt_sign[0::2] = -1

        s_x = 1e6                         # equal st. deviations for circular blobs
        s_y = 1e6
        mean_y = self.ymax/2              # blobs in the midle vertically
        xx,yy = np.meshgrid(self.x,self.y)# mesh of computational volume

        for i,scale in enumerate(x_pos):  # looping over nr of blobs and perturb T
            mean_x = self.xmax*scale      # blob according to nr
            perturbation = np.exp(-0.5*((xx-mean_x)**2/s_x**2 +(yy-mean_y)**2/s_y**2))
            # the amplitude is based on the the surface T

            self.T += 0.5*Sun.T_photo*perturbation*alt_sign[i]


    def timestep(self):
        "Getting the optimal timestep based on max relative change in primary variables"

        # Max relative changes in each variable:
        max_rel_rho = np.nanmax(np.abs(self.ddt_rho/self.rho))
        max_rel_e   = np.nanmax(np.abs(self.ddt_e/self.e))
        max_rel_x   = np.nanmax(np.abs(self.u/self.delta_x))
        max_rel_y   = np.nanmax(np.abs(self.w/self.delta_y))

        # Max relative change of all variables (1e-5 is included to remove cases where delta = 0)
        delta = np.nanmax(np.array([max_rel_rho,max_rel_e,max_rel_x,max_rel_y,1e-5]))
        # Optimal dt based on max relative change
        dt = self.p/delta

        # If dt is too low or too high force it to a min/max value
        # Too low or too high dt can cause unstability, especially in equilibrium and prolong the calculations
        if dt<1e-7:
            dt = 1e-7
            self.forced_dt += 1
        elif dt>0.1:
            dt = 0.1
            self.forced_dt += 1

        return dt


    def boundary_conditions(self):
        "Setting vertical boundary conditions for Energy, Density and Velocity"

        # Energy
        self.e[0,:]  = (4*self.e[1,:]-self.e[2,:])/(3-self.g_y*2*self.delta_y*self.mymu_kB/self.T[0,:])
        self.e[-1,:] = (4*self.e[-2,:]-self.e[-3,:])/(3+self.g_y*2*self.delta_y*self.mymu_kB/self.T[-1,:])

        # Density
        self.rho[0,:]  = (self.gamma-1)*self.mymu_kB*self.e[0,:]/self.T[0,:]
        self.rho[-1,:] = (self.gamma-1)*self.mymu_kB*self.e[-1,:]/self.T[-1,:]

        # Vertical and horizontal velocity
        self.w[0,:]  = 0
        self.w[-1,:] = 0
        self.u[0,:]  = (4*self.u[1,:]-self.u[2,:])/3
        self.u[-1,:] = (4*self.u[-2,:]-self.u[-3,:])/3


    def flow_directions(self):
        """
        Calculates flow directions and returns four 2D arrays with indices for the flow the direction
        to be used in upwind differencing.
        """
        u_pos = self.u >=0    # Boolean array for + horizontal flow
        u_neg = self.u <0     # Boolean array for for - horizontal flow

        # Vertical
        w_pos = self.w >=0
        w_neg = self.w <0

        return u_pos,u_neg,w_pos,w_neg


    # Central difference schemes
    def get_central_x(self,var):
        """
        x-direction with periodic horizontal boundary
        var: variable to differentiate
        """
        return (np.roll(var,-1,axis=1)-np.roll(var,1,axis=1))/(2*self.delta_x)

    def get_central_y(self,var):
        """
        y-direction with periodic vertical boundary
        (vertical boundary is controlled in self.boundary_conditions())
        var: variable to differentiate
        """
        return (np.roll(var,-1,axis=0)-np.roll(var,1,axis=0))/(2*self.delta_y)


    # Upwind difference schemes
    def get_upwind_x(self,var,pos_id,neg_id):
        """
        x-direction with periodic horizontal boundary
        var: variable to differentiate
        pos_id: indices for positive flow upwind differencing
        neg_id: indices for negative flow upwind differencing

        Uses different expressions for the differential based on the flow sign
        Returns the resulting differential 2D array
        """

        # resulting array with differential
        res = np.zeros((self.ny,self.nx))

        diff_pos = (var-np.roll(var,1,axis=1))/self.delta_x  # if + flow
        diff_neg = (np.roll(var,-1,axis=1)-var)/self.delta_x # if - flow

        # Filling array with appropiate differentials
        res[pos_id] = diff_pos[pos_id]
        res[neg_id] = diff_neg[neg_id]

        return res

    def get_upwind_y(self,var,pos_id,neg_id):
        """
        y-direction with periodic vertical boundary
        (vertical boundary: controlled in self.boundary_conditions())
        var: variable to differentiate
        pos_id: indices for positive upwind differencing flow
        neg_id: indices for negative upwind differencing flow
        """

        res = np.zeros((self.ny,self.nx))   # array with differential
        diff_pos = (var-np.roll(var,1,axis=0))/self.delta_y  # if + flow
        diff_neg = (np.roll(var,-1,axis=0)-var)/self.delta_y # if - flow

        # Filling resulting array with appropiate differentials
        res[pos_id] = diff_pos[pos_id]
        res[neg_id] = diff_neg[neg_id]

        return res



if __name__ == '__main__':

    vis = FVis.FluidVisualiser(fontsize=17)
    solver = convection2D()
    # Setup simulation configuration: extent of axis in [Mm]
    extent = [0,solver.xmax/1e6,0,solver.ymax/1e6]
    import os

    print("Enable Sanity test? [y/n]")
    san = input()
    while True:
        try:
            if san == "y":
                sim_time = 60
                nr = 1
                perturb = False
                parameter = 'T'
                name_a = 'T_sanity_60'
                subtitle = 'Hydrostatic equilibrium over 60 seconds'
                tit ='Animated Sanity of '+parameter+',\n '+subtitle
                solver.boundary_conditions()
                fol = 'T_sanity_60'
                vis.save_data(sim_time,solver.hydro_solver,rho=solver.rho,e=solver.e,u=solver.u,w=solver.w,P=solver.P,T=solver.T,sim_fps=5,folder=fol)
                vis.animate_2D(parameter,save=True,video_name=name_a,title=tit,\
                    units={'Lx':'Mm','Lz':'Mm'},\
                    folder=fol,extent=extent,cbar_aspect=0.05)
                vis.delete_current_data()
                break
            elif san == "n":
                print()
                break
            else:
                assert False
        except AssertionError:
            print("No valid input value for sanity test! Please try again.")
            print("Enable Sanity test? [y/n]")
            san = input()
            continue

    print("Enable perturbation? [y/n]")
    pert = input()
    while True:
        try:
            if pert == "y":
                print()
                nr = 1 #change nr to 5 for 5 alternating perturbations
                perturb = True
                solver.initialise(perturb=True,nr = nr)
                if nr ==1:
                    subtitle = 'single positive perturbation '
                elif nr ==5:
                    subtitle = '5 alternating perturbations '
                break
            elif pert == "n":
                perturb = False
                print()
                solver.initialise()
                break
            else:
                assert False
        except AssertionError:
            print("No valid input value for perturbation! Fail to initialize solver. Please try again.")
            print("Enable perturbation? [y/n]")
            pert = input()
            continue

    if pert == "y" or pert == "n":
        print("Insert time duration of simulation in real time seconds.")
        sim_time = input()
        printed = 0
        while True:
            try:
                sim_time = float(sim_time)
                if isinstance(sim_time, float) ==  True:
                    while printed == 0:
                        print()
                        print("Quantities that can be visualised ")
                        print("----------------------------------")
                        print( """
'rho':  Mass density
'drho': Mass density contrast
'u':    Horizontal velocity
'w':    Vertical velocity
'e':    Internal energy density
'de':   Internal energy density contrast
'es':   Specific internal energy
'P':    Pressure
'dP':   Pressure contrast
'T':    Temperature
'dT':   Temperature contrast
'v':    Speed
'ru':   Horizontal momentum density
'rw':   Vertical momentum density
'rv':   Momentum density
'eu':   Horizontal energy flux
'ew':   Vertical energy flux
'ev':   Energy flux

----------------------------------------
'a_ev:  Horizontally averaged energy flux
'all':  Visualise all variables""")
                        printed = 1
                        break
                    printed = 1
                    print()
                    print("Insert parameter to visualise or 'q' to quit the program.")
                    print()
                    parameter = input()

                    while True:
                        try:
                            if parameter in ['rho', 'drho', 'u', 'w', 'e', 'de', 'es', 'P', 'dP', 'T', 'dT', 'v', 'ru', 'rw', 'rv', 'eu', 'ew', 'ev']:
                                    solver.boundary_conditions()
                                    name_app = parameter+'_'+str(int(sim_time))
                                    folder = parameter+'_'+str(int(sim_time))
                                    subtitle =  'over '+str(int(sim_time))+' seconds'
                                    title ='Animated '+parameter+',\n '+subtitle
                                    vis.save_data(sim_time,solver.hydro_solver,rho=solver.rho,e=solver.e,u=solver.u,w=solver.w,P=solver.P,T=solver.T,sim_fps=5,folder=folder)
                                    vis.animate_2D(parameter,save=True,video_name=name_app,title=title,\
                                        units={'Lx':'Mm','Lz':'Mm'},\
                                        folder=folder,extent=extent,cbar_aspect=0.05)
                                    vis.delete_current_data()
                                    solver.initialise(perturb=perturb,nr = nr)
                                    break
                            elif parameter == 'a_ev':
                                name_ = 'horizontal_energy_flux'
                                subtitle = 'single positive perturbation over '+str(int(sim_time))+' seconds'
                                solver.boundary_conditions()
                                fold = 'HorizontalEnergyFlux'
                                t = 'Horizontally averaged energy flux,\n'+subtitle
                                vis.save_data(sim_time,solver.hydro_solver,rho=solver.rho,e=solver.e,u=solver.u,w=solver.w,P=solver.P,T=solver.T,sim_fps=5,folder=fold)
                                vis.animate_energyflux(save=True,video_name=name_,title=t,\
                                        units={'Lx':'Mm','Lz':'Mm'},\
                                        folder=fold,extent=extent)
                                vis.delete_current_data()
                                solver.initialise(perturb=perturb,nr = nr)
                                break
                            elif parameter == 'all':
                                solver.boundary_conditions()
                                for p in ['rho', 'drho', 'u', 'w', 'e', 'de', 'es', 'P', 'dP', 'T', 'dT', 'v', 'ru', 'rw', 'rv', 'eu', 'ew', 'ev']:
                                    name_app = p+'_'+str(int(sim_time))
                                    folder = p+'_'+str(int(sim_time))
                                    subtitle =  'over '+str(int(sim_time))+' seconds'
                                    title ='Animated '+p+',\n '+subtitle
                                    vis.save_data(sim_time,solver.hydro_solver,rho=solver.rho,e=solver.e,u=solver.u,w=solver.w,P=solver.P,T=solver.T,sim_fps=5,folder=folder)
                                    vis.animate_2D(p,save=True,video_name=name_app,title=title,\
                                        units={'Lx':'Mm','Lz':'Mm'},\
                                        folder=folder,extent=extent,cbar_aspect=0.05)
                                    vis.delete_current_data()
                                    solver.initialise(perturb=perturb,nr = nr)
                                break
                            elif parameter == 'q':
                                print()
                                print('End of program.')
                                os._exit(0)
                            else:
                                assert False
                        except AssertionError:
                            print()
                            print("No valid input parameter! Please enter a valid parameter.")
                            break

                    continue
            except ValueError:
                print()
                print("Input must be a real number.")
                sim_time = input()
                continue
            break
