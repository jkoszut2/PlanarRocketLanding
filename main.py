import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ttictoc import tic,toc
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from datetime import datetime
import scipy.integrate as integrate
# from IPython import embed
import pickle


# Constants
m = 50000               # Rocket mass [kg]
m_prop = 10000          # Propellant mass [kg]
l = 70                  # Rocket length [m]
g_Mars = 3.72           # Mars gravity [m/s^2]
g_Earth = 9.81          # Earth gravity [m/s^2]
Fmax = 500e3            # Max thrust [N]
Isp = 300               # Specific impulse [N*s/kg]
dmax = 5 * np.pi / 180  # Max engine gimbal [rad]
tmax = 25 * np.pi / 180 # Max pitch [rad]
rho = 0.020             # Mars air density [kg/m^3]
Cd = 1                  # Drag coefficient
diam = 3.7              # Rocket diameter [m]

# Define states, inputs, and constraints
nx = 7 # [pitch (1), pitch rate (2), posx (3), posz (4), velx (5), velz (6), propellant mass (7)]
nu = 2 # [thrust (1), gimbalx (2)]
zMin = np.array([-tmax, -0.5, -5000, 0, -100, -100, 0])
zMax = np.array([tmax, 0.5, 5000, 3000, 100, 100, m_prop])
uMin = np.array([0.2*Fmax, -dmax])
uMax = np.array([Fmax, dmax])

# Simulation parameters
TS = 0.1 # MPC sampling period [sec]
TF = 85 # Descent time [sec]
N = round(TF/TS) # Simulation time steps

# Initial condition
zinit = np.array([-5 * np.pi / 180, 0, 1400, 1500, 70, -100, m_prop])

# Target terminal condition
zNBar = np.array([0, 0, 0, 0, 0, 0, 0]) # Target propellant mass gets ignored

create_plots = True
create_animation = True


def optSimple(zinit, zref, uref, N, TS, uprev):
    """ 
    This function is used to carry out MPC. It is similar to the
    optNonlin function with the difference being it has a simplified
    dynamics model for the purpose of decreasing computational complexity.
    :param zinit:  Initial condition vector with the following format
                   [pitch (1), pitch rate (2), posx (3), posz (4), velx (5), velz (6), propellant mass (7)]
    :param zref:  Reference trajectory computed via some trajectory optimization method
    :param uref:  Feedforward control input from trajectory optimization
    :param N:  Number of time steps
    :param TS:  Discretization period
    :param uprev:  Previous control input

    """

    # Initial condition and terminal constraints
    z0Bar = np.array(zinit)
    # z0Bar[6] -= m # subtract dry vehicle mass to get propellant mass
    zNBar = zref[:,-1]

    model = pyo.ConcreteModel()
    model.tidx = pyo.Set(initialize=range(0, N+1))
    model.zidx = pyo.Set(initialize=range(0, nx))
    model.uidx = pyo.Set(initialize=range(0, nu))

    # Create state and input variables trajectory:
    model.z = pyo.Var(model.zidx, model.tidx)
    model.u = pyo.Var(model.uidx, model.tidx)

    # Initialize guess for solution
    for i in range(uref.shape[1]):
        model.u[0, i] = uref[0, i]
        model.u[1, i] = uref[1, i]

    # Constraints
    # Initial Condition
    model.constraint1 = pyo.Constraint(model.zidx, rule=lambda model, i: model.z[i, 0] == z0Bar[i])
    # State Dynamics
    model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t+1] \
                                       == model.z[0, t] + TS*(model.z[1, t])
                                       if t < N else pyo.Constraint.Skip) # angle
    model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t+1] \
                                       == model.z[1, t] - TS*l/2/(1/12*(m+z0Bar[6])*l**2) * model.u[0, t] * model.u[1, t]
                                       if t < N else pyo.Constraint.Skip) # angular rate
    model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[2, t+1] \
                                       == model.z[2, t] + TS*(model.z[4, t])
                                       if t < N else pyo.Constraint.Skip) # horizontal position
    model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[3, t+1] \
                                       == model.z[3, t] + TS*(model.z[5, t])
                                       if t < N else pyo.Constraint.Skip) # vertical position
    model.constraint6 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[4, t+1] \
                                       == model.z[4, t] + TS*(1/(m+z0Bar[6])*model.u[0, t] * pyo.sin(model.z[0, t]+model.u[1, t]))
                                       if t < N else pyo.Constraint.Skip) # horizontal velocity
    model.constraint7 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[5, t+1] \
                                       == model.z[5, t] + TS*(-g_Mars + 1/(m+z0Bar[6])*model.u[0, t] * pyo.cos(model.z[0,t] + model.u[1, t]))
                                       if t < N else pyo.Constraint.Skip) # vertical velocity
    model.constraint8 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[6, t+1] \
                                       == model.z[6, t] - TS*(model.u[0, t]/Isp/g_Earth)
                                       if t < N else pyo.Constraint.Skip) # propellant mass
    # Input Constraints
    model.constraint9 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] <= uMax[0]
                                       if t < N else pyo.Constraint.Skip) # max thrust
    model.constraint10 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] >= uMin[0]
                                       if t < N else pyo.Constraint.Skip) # min thrust
    model.constraint11 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] <= uMax[1]
                                       if t < N else pyo.Constraint.Skip) # max gimbal angle
    model.constraint12 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] >= uMin[1]
                                       if t < N else pyo.Constraint.Skip) # min gimbal angle
    model.constraint1slew = pyo.Constraint(model.tidx, rule=lambda model, t: (model.u[1, t+1]-model.u[1, t])/TS <= 0.1
                                       if t < N else pyo.Constraint.Skip) # max slew rate
    model.constraint2slew = pyo.Constraint(model.tidx, rule=lambda model, t: (model.u[1, t+1]-model.u[1, t])/TS >= -0.1
                                       if t < N else pyo.Constraint.Skip) # min slew rate
    model.constraintu1prev1 = pyo.Constraint(rule=lambda model : (model.u[1, 0] - uprev[1])/TS <= 0.1)
    model.constraintu1prev2 = pyo.Constraint(rule=lambda model : -(model.u[1, 0] - uprev[1])/TS <= 0.1)

    # State Constraints
    # No state constraints used in this formulation. It is assumed that the states will follow the reference closely.
    # Objective:
    model.cost = pyo.Objective(expr = sum( 5000*(model.z[0, t] - zref[0, t])**2 + 1000*(model.z[1, t] - zref[1, t])**2 \
                                           + 5*(model.z[2, t] - zref[2, t])**2 + 25*(model.z[3, t] - zref[3, t])**2 \
                                           + 5*(model.z[4, t] - zref[4, t])**2 + 5*(model.z[5, t] - zref[5, t])**2 \
                                           + 1000*(model.u[1, t] - uref[1, t])**2 for t in model.tidx if t < N), \
                               sense=pyo.minimize)

    # Solve optimization problem
    tic()
    results = pyo.SolverFactory('ipopt').solve(model) # Adjust tol using options={'tol': 1e-5}
    # log_infeasible_constraints(model, log_expression=True, log_variables=True)
    # logging.basicConfig(filename='example.log', level=logging.INFO)
    runtime = toc()
    print(f"Solver runtime: {runtime} sec")
    time = []
    z1 = [pyo.value(model.z[0,0])]
    z2 = [pyo.value(model.z[1,0])]
    z3 = [pyo.value(model.z[2,0])]
    z4 = [pyo.value(model.z[3,0])]
    z5 = [pyo.value(model.z[4,0])]
    z6 = [pyo.value(model.z[5,0])]
    z7 = [pyo.value(model.z[6,0])]
    u1 = [pyo.value(model.u[0,0])]
    u2 = [pyo.value(model.u[1,0])]

    for t in model.tidx:
        if t < N+1:
            time.append(t*TS)
        if t < N:
            z1.append(pyo.value(model.z[0,t+1]))
            z2.append(pyo.value(model.z[1,t+1]))
            z3.append(pyo.value(model.z[2,t+1]))
            z4.append(pyo.value(model.z[3,t+1]))
            z5.append(pyo.value(model.z[4,t+1]))
            z6.append(pyo.value(model.z[5,t+1]))
            z7.append(pyo.value(model.z[6,t+1]))
        if t < N-1:
            u1.append(pyo.value(model.u[0,t+1]))
            u2.append(pyo.value(model.u[1,t+1]))
    twtol = 1e-5 # Constraint printing tolerance
    printflag = 0
    if (np.min(z1)-twtol) < zMin[0] or (np.max(z1)+twtol) > zMax[0]:
        printflag = 1
        if (np.min(z1)-twtol) < zMin[0]:
            print(f"Pitch exceeds soft constraint by {np.rad2deg(zMin[0]-np.min(z1))} deg")
        elif (np.max(z1)+twtol) > zMax[0]:
            print(f"Pitch exceeds soft constraint by {np.rad2deg(np.max(z1)-zMax[0])} deg")
    if (np.min(z4)-twtol) < zMin[3]:
        printflag = 1
        print(f"Altitude AGL by {pyo.value(np.min(z4)-twtol)} m")
    if printflag == 0:
        for k in range(len(z1)):
            z_curr = np.array([z1[k], z2[k], z3[k], z4[k], z5[k], z6[k], z7[k]])
            if not np.all(z_curr<=zMax+twtol) or not np.all(z_curr>=zMin-twtol):
                print(f"WARNING: CONSTRAINT(S) EXCEEDED (k={k})")
                print(f"Max constraints pass status: {z_curr<=zMax}")
                print(f"Min constraints pass status: {z_curr>=zMin}")
    return [time, np.array([z1, z2, z3, z4, z5, z6, z7]),
            np.array([u1, u2]), runtime]


def optNonlin(zinit, zNBar, N, TS):
    """ 
    This function is used to generate the reference trajectory.
    :param zinit:  Initial condition vector with the following format
                   [pitch (1), pitch rate (2), posx (3), posz (4), velx (5), velz (6), propellant mass (7)]
    :param zNBar:  Terminal condition vector using the same format as zinit
    :param N:  Number of time steps
    :param TS:  Discretization period
    """

    # Initial condition and terminal constraints
    z0Bar = np.array(zinit)

    model = pyo.ConcreteModel()
    model.tidx = pyo.Set(initialize=range(0, N+1))
    model.zidx = pyo.Set(initialize=range(0, nx))
    model.uidx = pyo.Set(initialize=range(0, nu))

    # Create state and input variables trajectory:
    model.z = pyo.Var(model.zidx, model.tidx)
    model.u = pyo.Var(model.uidx, model.tidx)

    # Constraints
    # Initial Condition
    model.constraint1 = pyo.Constraint(model.zidx, rule=lambda model, i: model.z[i, 0] == z0Bar[i])
    # State Dynamics
    model.constraint2 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t+1] \
                                       == model.z[0, t] + TS*(model.z[1, t])
                                       if t < N else pyo.Constraint.Skip) # angle
    model.constraint3 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t+1] \
                                       == model.z[1, t] - TS*l/2/(1/12*(m+model.z[6, t])*l**2) * model.u[0, t] * pyo.sin(model.u[1, t])
                                       if t < N else pyo.Constraint.Skip) # angular rate
    model.constraint4 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[2, t+1] \
                                       == model.z[2, t] + TS*(model.z[4, t]) # horizontal position
                                       if t < N else pyo.Constraint.Skip)
    model.constraint5 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[3, t+1] \
                                       == model.z[3, t] + TS*(model.z[5, t]) # vertical position
                                       if t < N else pyo.Constraint.Skip)
    model.constraint6 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[4, t+1] \
                                       == model.z[4, t] + TS*(1/(m+model.z[6,t]) * model.u[0, t] * pyo.sin(model.z[0, t]+model.u[1, t]))
                                       if t < N else pyo.Constraint.Skip) # horizontal velocity
    model.constraint7 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[5, t+1] \
                                       == model.z[5, t] + TS*(-g_Mars + 1/(m+model.z[6,t]) * model.u[0, t] * pyo.cos(model.z[0,t]+model.u[1, t]))
                                       if t < N else pyo.Constraint.Skip) # vertical velocity
    model.constraint8 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[6, t+1] \
                                       == model.z[6, t] - TS*(model.u[0, t]/Isp/g_Earth)
                                       if t < N else pyo.Constraint.Skip) # propellant mass
    # Input Constraints
    model.constraint9 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] <= uMax[0]
                                       if t < N else pyo.Constraint.Skip) # max thrust
    model.constraint10 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[0, t] >= uMin[0]
                                       if t < N else pyo.Constraint.Skip) # minn thrust
    model.constraint11 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] <= uMax[1]
                                       if t < N else pyo.Constraint.Skip) # max gimbal angle
    model.constraint12 = pyo.Constraint(model.tidx, rule=lambda model, t: model.u[1, t] >= uMin[1]
                                       if t < N else pyo.Constraint.Skip) # min gimbal angle
    model.constraint1slewp = pyo.Constraint(model.tidx, rule=lambda model, t: (model.u[1, t+1]-model.u[1, t])/TS <= 0.1
                                       if t < N else pyo.Constraint.Skip) # max slew rate
    model.constraint2slewp = pyo.Constraint(model.tidx, rule=lambda model, t: -(model.u[1, t+1]-model.u[1, t])/TS <= 0.1
                                       if t < N else pyo.Constraint.Skip) # min slew rate
    # Final Condition
    model.constraint16 = pyo.Constraint(rule=lambda model : model.z[0, N] == zNBar[0])
    model.constraint17 = pyo.Constraint(rule=lambda model : model.z[1, N] == zNBar[1])
    model.constraint18 = pyo.Constraint(rule=lambda model : model.z[2, N] == zNBar[2])
    model.constraint19 = pyo.Constraint(rule=lambda model : model.z[3, N] == zNBar[3])
    model.constraint20 = pyo.Constraint(rule=lambda model : model.z[4, N] == zNBar[4])
    model.constraint21 = pyo.Constraint(rule=lambda model : model.z[5, N] == zNBar[5])
    # State Constraints
    model.constraint22 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t] <= zMax[0] if t <= N else pyo.Constraint.Skip)
    model.constraint23 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[0, t] >= zMin[0] if t <= N else pyo.Constraint.Skip)
    model.constraint24 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t] <= zMax[1] if t <= N else pyo.Constraint.Skip)
    model.constraint25 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[1, t] >= zMin[1] if t <= N else pyo.Constraint.Skip)
    model.constraint26 = pyo.Constraint(model.tidx, rule=lambda model, t: model.z[3, t] >= zMin[3] if t <= N else pyo.Constraint.Skip)

    # Objective:
    model.cost = pyo.Objective(expr = (z0Bar[6] - model.z[6, N])**2)

    # Now we can solve:
    tic()
    results = pyo.SolverFactory('ipopt').solve(model)
    runtime = toc()
    print(f"Solver runtime: {runtime} sec")
    time = []
    z1 = [pyo.value(model.z[0,0])]
    z2 = [pyo.value(model.z[1,0])]
    z3 = [pyo.value(model.z[2,0])]
    z4 = [pyo.value(model.z[3,0])]
    z5 = [pyo.value(model.z[4,0])]
    z6 = [pyo.value(model.z[5,0])]
    z7 = [pyo.value(model.z[6,0])]
    u1 = [pyo.value(model.u[0,0])]
    u2 = [pyo.value(model.u[1,0])]

    for t in model.tidx:
        if t < N+1:
            time.append(t*TS)
        if t < N:
            z1.append(pyo.value(model.z[0,t+1]))
            z2.append(pyo.value(model.z[1,t+1]))
            z3.append(pyo.value(model.z[2,t+1]))
            z4.append(pyo.value(model.z[3,t+1]))
            z5.append(pyo.value(model.z[4,t+1]))
            z6.append(pyo.value(model.z[5,t+1]))
            z7.append(pyo.value(model.z[6,t+1]))
        if t < N-1:
            u1.append(pyo.value(model.u[0,t+1]))
            u2.append(pyo.value(model.u[1,t+1]))
    return [time, np.array([z1, z2, z3, z4, z5, z6, z7]),
            np.array([u1, u2]), runtime]

def PlantSim(zinit, u, wx, N, TS):
    """ 
    This function is used to simulate the fully nonlinear system
    Aerodynamic drag forces are currently neglected but wind disturbances can be applied.
    
    """
    u = u.reshape((nu,))
    def PlantODE(t, x):
        x = x.reshape(nx,1)
        xdot = np.zeros((nx,))
        xdot[0] = x[1]
        xdot[1] = -l/2/(1 / 12 * (m+x[6]) * l**2) * (u[0] * np.sin(u[1]))
        xdot[2] = x[4]
        xdot[3] = x[5]
        xdot[4] = 1/(m+x[6]) * (u[0] * np.sin(x[0] + u[1]) + 0.5*rho*diam*l*Cd*wx**2)
        xdot[5] = -g_Mars + 1/(m+x[6]) * u[0] * np.cos(x[0] + u[1])
        xdot[6] = - u[0]/Isp/g_Earth
        return xdot.flatten()
    out = integrate.solve_ivp(PlantODE, (0, TS), zinit, method='RK45', rtol=1e-8)
    return np.hstack((zinit.reshape(nx,1), out.y[:,-1].reshape(nx,1)))


def plotResults(time, z, u, time_mpc, u_mpc, every=1, ani=False):
    z1 = z[0,:]
    z2 = z[1,:]
    z3 = z[2,:]
    z4 = z[3,:]
    z5 = z[4,:]
    z6 = z[5,:]
    z7 = z[6,:]
    u1 = u[0,:]
    u2 = u[1,:]

    fig = plt.figure()
    ax_static = plt.axes()
    ax_static.set_title('Thrust Profile')
    ax_static.plot(z3, z4)
    pts = np.arange(0,time[-1],TS*every)
    quiv_fac = 0.002
    ax_static.plot(zNBar[2], zNBar[3], zNBar[3]*0, '*')
    ptp_x = np.ptp([z3[round(j/TS)] for j in pts])
    ptp_y = np.ptp([z4[round(j/TS)] for j in pts])
    ptp_z = np.ptp([z5[round(j/TS)] for j in pts])
    for i in pts:
        ind = round(i/TS)
        ax_static.scatter(z3[ind], z4[ind], s=4, marker='o', color='r')
        ax_static.quiver(z3[ind], z4[ind],
                  np.sin(z1[ind]+u2[ind]), np.cos(z1[ind]+u2[ind]),
                  scale = 1/(l*u1[ind]/Fmax*quiv_fac), color='g') # Thrust vector

    ax_static.set_box_aspect(aspect = ptp_y/ptp_x)
    ax_static.set_xlabel('Position x [m]')
    ax_static.set_ylabel('Position y [m]')

    if ani:
        # Animation
        fig, axs = plt.subplots(3, 1, figsize=(6, 6),
                                gridspec_kw={'height_ratios': [1, 1, 3]})
        plt.subplots_adjust(left = 0.15, top = 0.96, right = 0.95, bottom = 0.1,
                            hspace = 0.5, wspace = 0.5)

        def get_arrow(ind):
            # Spacecraft
            length = l
            x = z3[ind]
            z = z4[ind]
            u = np.sin(z1[ind])*length
            w = np.cos(z1[ind])*length
            return x,z,u,w
        def get_arrow2(ind):
            # Spacecraft
            length = l
            x = z3[ind]
            z = z4[ind]
            u = -0.75*l*np.sin(z1[ind]+u2[ind])*max(0.4, u1[ind]/Fmax)
            w = -0.75*l*np.cos(z1[ind]+u2[ind])*max(0.4, u1[ind]/Fmax)
            return x,z,u,w
        global quiver
        global quiver2
        quiver = axs[1].quiver(*get_arrow(0))
        quiver2 = axs[1].quiver(*get_arrow2(0))
        axs[0].set_xlabel('Time [sec]')
        axs[0].set_ylabel(r'$F$ [kN]')
        axs[0].set_xlim(0, time[-1]+1)
        axs[0].set_ylim(0, 1.1*Fmax/1000)
        axs[1].set_xlabel('Time [sec]')
        axs[1].set_ylabel(r'$\delta$ [deg]')
        axs[1].set_xlim(0, time[-1]+1)
        axs[1].set_ylim(-1.1*dmax*180/np.pi, 1.1*dmax*180/np.pi)
        axs[2].set_xlim(ax_static.get_xlim())
        axs[2].set_ylim(zNBar[3]-100 ,zinit[3]+100)
        axs[2].set_xlabel('Position x [m]')
        axs[2].set_ylabel('Position z [m]')

        prop_cycle = plt.rcParams['axes.prop_cycle']
        mpc_colors = prop_cycle.by_key()['color']
        u_line, = axs[1].plot(time[:ind+1], u2[:ind+1]*180/np.pi, color='C0',
                              label=r'$\delta(t)$')
        u_mpc_line, = axs[1].plot(time_mpc[:len(u_mpc[0,0,:])],
                                  u_mpc[0,1,:]*180/np.pi, color=mpc_colors[0],
                                  label=r'$\delta[k|t], k=0,1,\cdots,N_{MPC}$')

        head_scaling = 0.5
        def update(ind):
            global quiver
            global quiver2
            quiver.remove()
            quiver2.remove()
            axs[0].plot(time[:ind+1], u1[:ind+1]/1000, color='C0')
            u_line.set_data(time[:ind+1], u2[:ind+1]*180/np.pi)
            u_line.set_color('C0')
            if ind%1 == 0: # Adjust update period to change animation size
                u_mpc_line.set_data(time_mpc[ind:ind+len(u_mpc[0,0,:])], u_mpc[ind,1,:]*180/np.pi)
                u_mpc_line.set_color(mpc_colors[(ind+1)%len(mpc_colors)])
                axs[1].legend(loc="best", prop={'size': 7})
            axs[2].plot(z3[:ind+1], z4[:ind+1], color='C0', zorder=0)
            quiver = axs[2].quiver(*get_arrow(ind), color='g',
                               headwidth=3*head_scaling,
                               headlength=5*head_scaling,
                               headaxislength=4.5*head_scaling, zorder=5)
            quiver2 = axs[2].quiver(*get_arrow2(ind), color='r',
                               headwidth=0*head_scaling,
                               headlength=0*head_scaling,
                               headaxislength=0*head_scaling,
                               scale=1/(0.25*quiv_fac),
                               zorder=10)

        ani = FuncAnimation(fig, update, frames=np.arange(0,round((TF+TS)/TS),1), interval=200)

        # save animation
        aniname = 'Landing_' + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + '.gif'
        ani.save(aniname, writer='ffmpeg', fps=50, dpi=300)


def plotResultsSimple(time, z, u, axs):
    z1 = z[0,:]
    z2 = z[1,:]
    z3 = z[2,:]
    z4 = z[3,:]
    z5 = z[4,:]
    z6 = z[5,:]
    z7 = z[6,:]
    u1 = u[0,:]
    u2 = u[1,:]
    axs[0].plot(time, z1)
    axs[0].set_ylabel(r'$\theta$ [rad]')
    axs[1].plot(time, z2)
    axs[1].set_ylabel(r'$\omega_x$ [rad/s]')
    axs[2].plot(time, z3)
    axs[2].set_ylabel(r'$P_x$ [m]')
    axs[3].plot(time, z4)
    axs[3].set_ylabel(r'$P_z$ [m]')
    axs[3].set_xlabel('Time [sec]')
    axs[4].plot(time, np.linalg.norm(np.vstack((zNBar[2]-z3, zNBar[3]-z4)), axis=0))
    axs[4].set_ylabel(r'$\| P_{error} \|$ [m]')
    axs[5].plot(time, z5)
    axs[5].set_ylabel(r'$V_x$ [m/s]')
    axs[6].plot(time, z6)
    axs[6].set_ylabel(r'$V_z$ [m/s]')
    axs[7].plot(time[:-1], u1)
    axs[7].set_ylabel(r'$F$ [N]')
    axs[8].plot(time[:-1], np.array(u2)*180/np.pi)
    axs[8].set_ylabel(r'$\delta_x$ [deg]')
    axs[9].plot(time, np.array(z7))
    axs[9].set_ylabel(r'$m_{prop}$ [kg]')
    axs[9].set_xlabel('Time [sec]')
    plt.tight_layout()


def createPlots(runtimes, runtime_MPC, zref, uref, zopt, uopt, u_mpc_history):
    """ Function to create plots of results """

    fig, ax1 = plt.subplots()
    ax1.set_title('IPOPT Solver Runtime')
    # Use linspace instead of arange because the former gives a predictable length
    # whereas the latter does not due to floating point error. See
    # np.arange(0, 0.1*488, 0.1) and np.arange(0, 0.1*489, 0.1).
    # The difference in lengths is two as opposed to the expected value of one.
    # np.arange(0,TS*(len(runtimes)-0),TS) gets replaced with linspace call below
    ax1.plot(np.linspace(0, TS*(len(runtimes)-1), len(runtimes)),
             runtimes, label='Instanteous')
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(0, TS*(len(runtimes)-1), len(runtimes)),
             np.cumsum(runtimes), 'r', label='Cumulative')
    ax1.set_xlabel('Time Step [sec]')
    ax1.set_ylabel('Instanteous Solver Runtime [sec]')
    ax2.set_ylabel('Cumulative Solver Runtime [sec]', rotation=270, labelpad=15)
    handles,labels = [],[]
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    ax1.legend(handles, labels, loc='best')
    print(f"NLP Fuel Consumed: {m_prop-zref[6,-1]} kg")
    print(f"NLP X-position: {zref[2,-1]} m")
    print(f"NLP Z-position: {zref[3,-1]} m")
    print(f"MPC Runtime: {runtime_MPC} sec")
    print(f"MPC Fuel Consumed: {m_prop-zopt[6,-1]} kg")
    print(f"MPC Fuel Suboptimality: {zref[6,-1]-zopt[6,-1]} kg")
    print(f"MPC X-position Error: {zopt[2,-1]} m")
    print(f"MPC Z-position Error: {zopt[3,-1]} m")

    plotResults(np.linspace(0, TS*len(runtimes), len(runtimes)+1), zopt, uopt,
                np.linspace(0, TS*(len(runtimes)+len(u_mpc_history[0,0,:])),
                            len(runtimes)+1+len(u_mpc_history[0,0,:])),
                u_mpc_history, every=10, ani=create_animation)

    fig, axs = plt.subplots(5, 2, figsize=(8,6), sharex=True)
    fig.suptitle('Reference Trajectory (Blue) vs MPC (Orange)')
    plt.subplots_adjust(top=0.925)
    axs = axs.T.flatten()
    plotResultsSimple(np.linspace(0, TS*(zref.shape[1]-1), zref.shape[1]), zref, uref, axs)
    plotResultsSimple(np.linspace(0, TS*len(runtimes), len(runtimes)+1), zopt, uopt, axs)

    plt.show()


def main():

    # Reference Trajectory Generation
    # Uncomment below to generate a new trajectory
    # [timeref, zref, uref, _] = optNonlin(zinit, zNBar, N, TS)
    # embed()
    # Save Trajectory
    # with open('reference_trajectory_tmp.pkl', 'wb') as f:
    #     pickle.dump(timeref, f)
    #     pickle.dump(zref, f)
    #     pickle.dump(uref, f)

    # Load Trajectory
    with open('reference_trajectory.pkl', 'rb') as f:
        timeref = pickle.load(f)
        zref = pickle.load(f)
        uref = pickle.load(f)

    # Simulate Landing using MPC
    N_MPC = 50 # MPC Horizon length
    N_extra = 10 # Maximum steps for final approach error correction
    zinitk = zinit # Initial condition at given time t, i.e., z[k|t] for k=0
    zopt = np.zeros((nx,N+1))
    zopt[:,0] = zinitk
    uopt = np.zeros((nu,N))
    u_mpc_history = np.zeros((N,nu,N_MPC))
    runtimes = []
    extra_runs = 0
    tic()
    for i in range(round(TF/TS)+1+N_extra):
        print(f"Loop {i}")
        print(f"Altitude: {zinitk[3]} m")
        # Lookup reference trajectory
        if i+N_MPC <= N-1:
            zrefk = zref[:,i:i+N_MPC+1]
            urefk = uref[:,i:i+N_MPC+1]
        else:
            N_tmp_extra = np.min([N_MPC+1, i+N_MPC-(N-1)])
            if N - i > 0:
                zrefk = zref[:,i:]
                urefk = uref[:,i:]
                zrefk = np.hstack((zrefk, np.tile(zNBar.reshape(nx,1), N_tmp_extra)))
                urefk = np.hstack((urefk, np.tile(np.array([[zref[6,-1]*g_Mars, 0]]).T, N_tmp_extra)))
            else: # NOTE: N_extra must be > 0 to get in here
                zrefk = np.tile(zNBar.reshape(nx,1), N_tmp_extra)
                urefk = np.tile(np.array([[zref[6,-1]*g_Mars, 0, 0]]).T, N_tmp_extra)
        # Find optimal input sequence
        if i == 0:
            uprev = np.array([uref[0, 0], uref[1, 0]])
        else:
            uprev = uopt[:,i-1]
        [time, z, u, runtime] = optSimple(zinitk, zrefk, urefk, N_MPC, TS, uprev)
        
        # Wind disturbance
        if 0 <= i <= N+N_extra:
            wx = np.random.normal(10,5) # mean, std
            # print(f"Wind speed: wx m/s")
        else:
            w = 0
        # Apply first entry of optimal input sequence to real system
        z = PlantSim(zinitk, u[:,0], wx, 1, TS)
        # Update stored states
        if i < N:
            zopt[:,i+1] = z[:,1]
            uopt[:,i] = u[:,0]
            u_mpc_history[i, :, :] = u
        else:
            zopt = np.hstack((zopt,z[:,1].reshape(nx,1)))
            uopt = np.hstack((uopt,u[:,0].reshape(nu,1)))
            u_mpc_history = np.concatenate((u_mpc_history, u.reshape(1,nu,N_MPC)))
            extra_runs += 1
        zinitk = z[:,1]
        runtimes += [runtime]
        # Check if landing criteria are met
        if ((zinitk[3]-zNBar[3] <= 0.1) and (np.max(np.abs(zinitk[0:2]-zNBar[0:2])) <= 0.05) \
             and (np.max(np.abs(zinitk[4:6]-zNBar[4:6])) <= 1)) or zinitk[3] <= 0:
                break
    runtime_MPC = toc()
    final_num_steps = i
    # Truncate state and input vectors if terminated early
    if final_num_steps < N+1:
        zopt = zopt[:, :final_num_steps+2]
        uopt = uopt[:, :final_num_steps+1]

    # Create Plots
    if create_plots:
        createPlots(runtimes, runtime_MPC, zref, uref, zopt, uopt, u_mpc_history)


main()
