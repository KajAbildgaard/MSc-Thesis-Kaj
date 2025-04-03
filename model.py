from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params
from darts.input.input_data import InputData
from darts.physics.geothermal.geothermal import Geothermal, GeothermalPH, GeothermalIAPWSFluidProps, GeothermalPHFluidProps


class Model(CICDModel):
    def __init__(self, n_points=128, iapws_physics: bool = True):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir()

        self.iapws_physics = iapws_physics
        self.set_input_data(n_points)
        self.set_physics()

        self.set_sim_params(first_ts=1e-3, mult_ts=8, max_ts=365, runtime=3650, tol_newton=1e-2, tol_linear=1e-6,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()

        T_init = 350.
        state_init = value_vector([200., 0.])
        enth_init = self.physics.property_containers[0].compute_total_enthalpy(state_init, T_init)
        self.initial_values = {self.physics.vars[0]: state_init[0],
                               self.physics.vars[1]: enth_init
                               }

    def set_reservoir(self):
        (nx, ny, nz) = (60, 60, 3)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000
        perm = load_single_keyword('permXVanEssen.in', 'PERMX')
        perm = perm[:nb]

        poro = np.ones(nb) * 0.2
        dx = 30
        dy = 30
        dz = np.ones(nb) * 30

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000,
                                         hcap=2200, rcond=500)
        self.reservoir.boundary_volumes['yz_minus'] = 1e8
        self.reservoir.boundary_volumes['yz_plus'] = 1e8
        self.reservoir.boundary_volumes['xz_minus'] = 1e8
        self.reservoir.boundary_volumes['xz_plus'] = 1e8

        return

    def set_wells(self):
        # add well's locations
        iw = [30, 30]
        jw = [14, 46]

        # add well
        self.reservoir.add_well("INJ")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("INJ", cell_index=(iw[0], jw[0], k + 1),
                                           well_radius=0.16, multi_segment=True)

        # add well
        self.reservoir.add_well("PRD")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("PRD", cell_index=(iw[1], jw[1], k + 1),
                                           well_radius=0.16, multi_segment=True)

    def set_physics(self):
        if self.iapws_physics:
            self.physics = Geothermal(self.idata, self.timer)
        else:
            self.physics = GeothermalPH(self.idata, self.timer)
            self.physics.determine_obl_bounds(state_min=[self.idata.obl.min_p, 273.15],
                                              state_max=[self.idata.obl.max_p, 373.15])

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_water_inj(8000, 300)
                # w.control = self.physics.new_bhp_water_inj(230, 308.15)
            else:
                w.control = self.physics.new_rate_water_prod(8000)
                # w.control = self.physics.new_bhp_prod(180)

    def compute_temperature(self, X):
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_input_data(self, n_points):
        #init_type = 'uniform'
        init_type = 'gradient'
        self.idata = InputData(type_hydr='thermal', type_mech='none', init_type=init_type)

        self.idata.rock.compressibility = 0.  # [1/bars]
        self.idata.rock.compressibility_ref_p = 1.  # [bars]
        self.idata.rock.compressibility_ref_T = 273.15  # [K]

        if self.iapws_physics:
            self.idata.fluid = GeothermalIAPWSFluidProps()
        else:
            self.idata.fluid = GeothermalPHFluidProps()

        # example - how to change the properties
        # self.idata.fluid.density['water'] = DensityBasic(compr=1e-5, dens0=1014)

        #from darts.physics.properties.basic import ConstFunc
        #self.idata.fluid.conduction_ev['water'] = ConstFunc(172.8)

        if init_type== 'uniform': # uniform initial conditions
            self.idata.initial.initial_pressure = 200.  # bars
            self.idata.initial.initial_temperature = 350.  # K
        elif init_type == 'gradient':         # gradient by depth
            self.idata.initial.reference_depth_for_pressure = 0  # [m]
            self.idata.initial.pressure_gradient = 100  # [bar/km]
            self.idata.initial.pressure_at_ref_depth = 1 # [bars]
        
            self.idata.initial.reference_depth_for_temperature = 0  # [m]
            self.idata.initial.temperature_gradient = 30  # [K/km]
            self.idata.initial.temperature_at_ref_depth = 273.15 + 20 # [K]

        # p3d = self.reservoir.mesh.pressure[:-(2*nz)].reshape(self.idata['nx'], self.idata['ny'], self.idata['nz'], order='F')
        # extra_p_along_y = np.linspace(20, 0, self.idata['ny'])
        # extra_p_along_y = np.expand_dims(extra_p_along_y, [0, 2])
        # p3d += extra_p_along_y
        # self.reservoir.mesh.pressure = p3d.flatten(order='A')  

        # # well controls
        # wctrl = self.idata.wells.controls  # short name
        # wctrl.type = 'rate'
        # #wctrl.type = 'bhp'
        # if wctrl.type == 'bhp':
        #     self.idata.wells.controls.inj_bhp = 250 # bars
        #     self.idata.wells.controls.prod_bhp = 100 # bars
        # elif wctrl.type == 'rate':
        #     self.idata.wells.controls.inj_rate = 5500 # m3/day
        #     self.idata.wells.controls.inj_bhp_constraint = 300 # upper limit for bhp, bars
        #     self.idata.wells.controls.prod_rate = 5500 # m3/day
        #     self.idata.wells.controls.prod_bhp_constraint = 70 # lower limit for bhp, bars
        # self.idata.wells.controls.inj_bht = 300  # K

        self.idata.obl.n_points = n_points
        self.idata.obl.min_p = 1.
        self.idata.obl.max_p = 351.
        self.idata.obl.min_e = 1000.  # kJ/kmol, will be overwritten in PHFlash physics
        self.idata.obl.max_e = 10000.  # kJ/kmol, will be overwritten in PHFlash physics
