from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params
from darts.input.input_data import InputData
from darts.physics.geothermal.geothermal import Geothermal, GeothermalPH, GeothermalIAPWSFluidProps, GeothermalPHFluidProps
from darts.engines import well_control_iface

class Model(CICDModel):
    def __init__(self, run_params, n_points=128, iapws_physics: bool = True):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.run_params = run_params    # Save run parameters

        self.set_reservoir()

        self.iapws_physics = iapws_physics
        self.set_input_data(n_points)
        self.set_physics()

        self.set_sim_params(first_ts=1e-4, mult_ts=4, max_ts=365, runtime=3650, tol_newton=1e-5, tol_linear=1e-6,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()


    def set_reservoir(self):
        (nx, ny, nz) = (62, 62, 18 + 25 + 18)
        nb = nx * ny * nz

        if self.run_params['model_name'] == 'homogeneous':
            perm = np.hstack([[0.001] * 18, 
                              [200] * 25,                         
                              [0.001] * 18]).astype(float)
            perm = np.broadcast_to(perm[None, None, :], (nx, ny, nz))
            poro = np.hstack([[0.001] * 18, 
                              [0.10] * 25, 
                              [0.001] * 18]).astype(float)
            poro = np.broadcast_to(poro[None, None, :], (nx, ny, nz))
            rcond = np.hstack([[216] * 18, 
                               [350] * 25, 
                               [216] * 18]).astype(float)
            rcond = np.broadcast_to(rcond[None, None, :], (nx, ny, nz))
        elif self.run_params['model_name'] == 'model 0':
            perm = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_0_perm.txt', 'PERM')
            perm = perm[:nb]
            poro = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_0_poro.txt', 'PORO')
            poro = poro[:nb]
            rcond = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_0_rcond.txt', 'RCOND')
            rcond = rcond[:nb]
        elif self.run_params['model_name'] == 'model 1':
            perm = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_1_perm.txt', 'PERM')
            perm = perm[:nb]
            poro = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_1_poro.txt', 'PORO')
            poro = poro[:nb]
            rcond = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_1_rcond.txt', 'RCOND')
            rcond = rcond[:nb]
        elif self.run_params['model_name'] == 'model 2':
            perm = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_2_perm.txt', 'PERM')
            perm = perm[:nb]
            poro = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_2_poro.txt', 'PORO')
            poro = poro[:nb]
            rcond = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_2_rcond.txt', 'RCOND')
            rcond = rcond[:nb]
        elif self.run_params['model_name'] == 'model 3':
            perm = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_3_perm.txt', 'PERM')
            perm = perm[:nb]
            poro = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_3_poro.txt', 'PORO')
            poro = poro[:nb]
            rcond = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_3_rcond.txt', 'RCOND')
            rcond = rcond[:nb]
        elif self.run_params['model_name'] == 'model 4':
            perm = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_4_perm.txt', 'PERM')
            perm = perm[:nb]
            poro = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_4_poro.txt', 'PORO')
            poro = poro[:nb]
            rcond = load_single_keyword('HeterogeneousModel/largeModel/area1800_varRcond_4_rcond.txt', 'RCOND')
            rcond = rcond[:nb]

        dx = np.hstack([[200] * 5, [100] * 5, [50] * 3,
                        [50] * 36,
                        [50] * 3, [100] * 5, [200] * 5]).astype(float)
        dy = dx
        dz = np.hstack([[30] * 9, [20] * 2, [10] * 3, [6] * 2, [4] * 2,
                        [4] * 25, 
                        [4] * 2, [6] * 2, [10] * 3, [20] * 2, [30] * 9]).astype(float)
        
        dx = np.broadcast_to(dx[:, None, None], (nx, ny, nz))
        dy = np.broadcast_to(dy[None, :, None], (nx, ny, nz))
        dz = np.broadcast_to(dz[None, None, :], (nx, ny, nz))

        # discretize structured reservoir   
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000,
                                         hcap=2200, rcond=rcond)
        self.reservoir.boundary_volumes['yz_minus'] = 1e20
        self.reservoir.boundary_volumes['yz_plus'] = 1e20
        self.reservoir.boundary_volumes['xz_minus'] = 1e20
        self.reservoir.boundary_volumes['xz_plus'] = 1e20

        return

    def set_wells(self):
        # add well's locations
        iw = [31, 31]  
        jw = [21, 41] 

        # add well
        self.reservoir.add_well("INJ")
        for k in range(18, 18+25):  #range(1, self.reservoir.nz)
            self.reservoir.add_perforation("INJ", cell_index=(iw[0], jw[0], k + 1),
                                           well_radius=0.16, multi_segment=False)

        # add well
        self.reservoir.add_well("PRD")
        for k in range(18, 18+25): 
            self.reservoir.add_perforation("PRD", cell_index=(iw[1], jw[1], k + 1),
                                           well_radius=0.16, multi_segment=False)

    def set_physics(self):
        self.idata.obl.min_p = 1
        self.idata.obl.max_p = 500
        self.idata.obl.min_e = 1000
        self.idata.obl.max_e = 15000
        if self.iapws_physics:
            self.physics = Geothermal(self.idata, self.timer)
        else:
            self.physics = GeothermalPH(self.idata, self.timer)
            self.physics.determine_obl_bounds(state_min=[self.idata.obl.min_p, 273.15],
                                              state_max=[self.idata.obl.max_p, 373.15])

    def set_well_controls(self, rate=None):
        if rate is None:
            rate = self.run_params['WR']
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=True,
                                               target=rate,
                                               phase_name='water',
                                               inj_composition=[],
                                               inj_temp=300)
            else:
                self.physics.set_well_controls(wctrl=w.control,
                                               control_type=well_control_iface.VOLUMETRIC_RATE,
                                               is_inj=False,
                                               target=rate
                                               )

    def compute_temperature(self, X):
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_input_data(self, n_points):
        init_type = 'gradient'
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type=init_type)

        self.idata.rock.compressibility = 0.  # [1/bars]
        self.idata.rock.compressibility_ref_p = 1.  # [bars]
        self.idata.rock.compressibility_ref_T = 273.15  # [K]

        if self.iapws_physics:
            self.idata.fluid = GeothermalIAPWSFluidProps()
        else:
            self.idata.fluid = GeothermalPHFluidProps()

        self.idata.obl.n_points = n_points
        self.idata.obl.min_p = 1.
        self.idata.obl.max_p = 351.
        self.idata.obl.min_e = 1000.  # kJ/kmol, will be overwritten in PHFlash physics
        self.idata.obl.max_e = 10000.  # kJ/kmol, will be overwritten in PHFlash physics

    def set_initial_conditions(self, mesh=None, pressure_grad=100, temperature_grad=30,
                                        ref_depth_p=0, p_at_ref_depth=1,
                                        ref_depth_T=0, T_at_ref_depth=293.15,
                                        add_press_grad=None):  #set_nonuniform_initial_conditions
        """
        Function to set nonuniform initial reservoir condition

        :param mesh: :class:`Mesh` object
        :param pressure_grad: Pressure gradient, calculates pressure based on depth [1/km]
        :param temperature_grad: Temperature gradient, calculates temperature based on depth [1/km]
        :param ref_depth_p: the reference depth for the pressure, km
        :param p_at_ref_depth: the value of the pressure at the reference depth, bars
        :param ref_depth_T: the reference depth for the temperature, km
        :param T_at_ref_depth: the value of the temperature at the reference depth, K
        """
        #INPUT: q and direction for GRADIENT
        q = self.run_params['q']            #m/s
        direction = self.run_params['dir']  #0, 45, 90, 135, 180, 225, 270, 315
        
        if mesh is None:
            mesh = self.reservoir.mesh

        depth = np.array(mesh.depth, dtype=float)
        # 1) Uniform or array -> specify constant or array of values for each variable to self.physics.set_initial_conditions_by_array()
        # Set the initial pressure and temperature       
        # pressure = np.array(mesh.pressure, copy=False)
        pressure = np.zeros(mesh.n_res_blocks, dtype=float)  
        pressure[:] = (depth[:pressure.size] / 1000 - ref_depth_p) * pressure_grad + p_at_ref_depth
        temperature = (depth[:pressure.size] / 1000 - ref_depth_T) * temperature_grad + T_at_ref_depth

        # Set the initial enthalpy for each block.            
        # enthalpy = np.array(mesh.enthalpy, copy=False)
        enthalpy = np.zeros(mesh.n_res_blocks, dtype=float)
        for j in range(mesh.n_res_blocks):                         
            # Create a state vector with the current pressure and a placeholder for enthalpy.
            state = value_vector([pressure[j], temperature[j]])
            # Compute total enthalpy using the physics property container.
            enthalpy[j] = self.physics.property_containers[0].compute_total_enthalpy(state)
       
        #Additional pressure field
        nx = self.reservoir.nx  
        ny = self.reservoir.ny  
        nz = self.reservoir.nz   
        n_res = mesh.n_res_blocks  # number of reservoir blocks 
        mz = 25                    # middle layers of the reservoir

        dx = self.reservoir.global_data['dx']
        dy = self.reservoir.global_data['dy']
        dx_vec = dx[:,0,0]          # 62 cell widths in x
        dy_vec = dy[0,:,0]          # 62 cell widths in y

        # true distances from the origin along each axis
        x_dist = np.concatenate(([0], np.cumsum(dx_vec)[:-1]))
        y_dist = np.concatenate(([0], np.cumsum(dy_vec)[:-1]))
        
        state_new = value_vector([np.mean(pressure), np.mean(enthalpy)])
        mu = self.physics.property_containers[0].viscosity_ev['water'].evaluate(state_new) #cP?
        print('PRESSURE is', np.mean(pressure))
        print('TEMPERATURE is', np.mean(temperature))
        print('ENTHALPY is', np.mean(enthalpy))
        print('VISCOSITY is', mu*0.001)

        mean_perm = []
        for i in range((nz-mz)//2, (nz-mz)//2 + mz): #range(18, 43)
            A = dx[:, :, i] * dy[:, :, i]
            perm = self.reservoir.global_data['permx'][:, :, i]
            mean_perm.append(np.sum(perm * A) / np.sum(A)) 
        mean_perm = np.array(mean_perm)
        k_eff = np.mean(mean_perm)         #mD
        print('PERMEABILITY is', k_eff)
        
        gradient =( 1.0e-5 * q * (mu*0.001) / (k_eff*9.869233e-16) )   #bar/m
        print('GRADIENT is', gradient)           
            
        if add_press_grad is None:
            add_press_grad = gradient #gradient
        
        if direction == 0:
            extra_p_along_x = np.zeros(nx)
            extra_p_along_y = y_dist[::-1] * add_press_grad   
        elif direction == 45:
            extra_p_along_x = 0.5 * np.sqrt(2) * x_dist[::-1] * add_press_grad
            extra_p_along_y = 0.5 * np.sqrt(2) * y_dist[::-1] * add_press_grad
        elif direction == 90:
            extra_p_along_x = x_dist[::-1] * add_press_grad
            extra_p_along_y = np.zeros(ny)
        elif direction == 135:
            extra_p_along_x = 0.5 * np.sqrt(2) * x_dist[::-1] * add_press_grad
            extra_p_along_y = 0.5 * np.sqrt(2) * y_dist * add_press_grad
        elif direction == 180:
            extra_p_along_x = np.zeros(nx)
            extra_p_along_y = y_dist * add_press_grad
        elif direction == 225:
            extra_p_along_x = 0.5 * np.sqrt(2) * x_dist * add_press_grad
            extra_p_along_y = 0.5 * np.sqrt(2) * y_dist * add_press_grad
        elif direction == 270:
            extra_p_along_x = x_dist * add_press_grad
            extra_p_along_y = np.zeros(ny)
        elif direction == 315:
            extra_p_along_x = 0.5 * np.sqrt(2) * x_dist * add_press_grad
            extra_p_along_y = 0.5 * np.sqrt(2) * y_dist[::-1] * add_press_grad
        
        extra_p_along_x = extra_p_along_x[:, np.newaxis, np.newaxis]
        extra_p_along_y = extra_p_along_y[np.newaxis, :, np.newaxis]
        p3d = pressure[:n_res].reshape(nx, ny, nz, order='F')
        p3d += extra_p_along_x + extra_p_along_y
        pressure[:n_res] = p3d.flatten(order='A')
        print('Pressure at the first block is', pressure[0])
        print('Pressure at the last block is', pressure[-1])
        print('All pressure values', pressure)

        # Set the initial enthalpy for reservoir blocks
        for j in range(n_res):                          
            # Create a state vector with the current pressure and a placeholder for enthalpy.
            state = value_vector([pressure[j], temperature[j]])
            # Compute total enthalpy using the physics property container.
            enthalpy[j] = self.physics.property_containers[0].compute_total_enthalpy(state)

        initial_values = {'pressure': pressure,
                          'enthalpy': enthalpy,}
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=initial_values)
