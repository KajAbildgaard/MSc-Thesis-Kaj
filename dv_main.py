from darts.engines import value_vector

from dv_model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

Runs = [#[Prod/Recharge,   model,        q (m/s), WR (m3/day), TEST_yrs_prd, TEST_yrs_recharge]
        ['Darcy Velocity test 1D',     'homogeneous', 2.3e-07, 0,          40,         0],] 

def main(input, output_directory, dir):
    rp = {'model_name': input[1][0],
          'q':          input[2][0],
          'dir':        dir,
          'WR':         input[3][0]}
    
    m = Model(run_params=rp, iapws_physics=True)
    m.init(discr_type='mpfa')
    m.set_output(output_folder=output_directory)
    m.platform = 'cpu'
    m.reconstruct_velocities()

    if input[0][0] == 'Production':
        m.run(days=100*365, verbose=False)
        m.output_to_vtk(ith_step=0, output_directory=output_directory)
        m.output_to_vtk(ith_step=1, output_directory=output_directory)

    elif input[0][0] == 'Recharge':
        m.run(days=365, verbose=False)
        td_dict = m.physics.engine.time_data

        prd_col = next(k for k in td_dict.keys() if "PRD : temperature (K)" in k)
        inj_col = next(k for k in td_dict.keys() if "INJ : temperature (K)" in k)
        T0   = td_dict[prd_col][0]     
        Tinj = td_dict[inj_col][0]
        threshold = T0 - 0.15 * (T0 - Tinj)

        max_search_years = 1000
        years = 1
        while years < max_search_years:
            last_T = td_dict[prd_col][-1]  
            if last_T <= threshold:
                break
            m.run(days=365, verbose=False)
            years += 1
        print('years === %d' % years)
        m.output_to_vtk(ith_step=0, output_directory=output_directory)
        m.output_to_vtk(ith_step=years, output_directory=output_directory)

        days_recharge = 2500*365   
        m.set_well_controls(rate=0)
        m.run(days=40*365, restart_dt=m.params.first_ts, verbose=False) # 40 years
        m.set_sim_params(max_ts=3650)                                   # max_ts = 10 years
        m.run(days=days_recharge - 40*365, verbose=False)
        m.output_to_vtk(ith_step=years+2, output_directory=output_directory)

    else:     # TEST runs
        days_prod     = input[4][0]*365
        days_recharge = input[5][0]*365

        m.run(days=days_prod, verbose=False)
        darcy_velocity = m.physics.engine.darcy_velocities
        reshaped_velocities = np.array(darcy_velocity).reshape(m.reservoir.n, m.physics.nph, 3)
        vel_path = os.path.join(output_directory, 'darcy_velocities.npy')
        np.save(vel_path, reshaped_velocities)
        # m.output_to_vtk(ith_step=0, output_directory=output_directory)
        # m.output_to_vtk(ith_step=1, output_directory=output_directory)

        if days_recharge != 0:
            m.set_well_controls(rate=0)
            m.run(days=40*365, restart_dt=m.params.first_ts, verbose=False) #ith_step=2
            m.set_sim_params(max_ts=3650)
            m.run(days=days_recharge - 40*365, verbose=False) #ith_step=3
            m.output_to_vtk(ith_step=3, output_directory=output_directory)

    m.print_timers()

    td = pd.DataFrame.from_dict(m.physics.engine.time_data)
    td_path = os.path.join(output_directory, 'darts_time_data.pkl')
    td.to_pickle(td_path)

    excel_path = os.path.join(output_directory, 'time_data.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        td.to_excel(writer, sheet_name='Sheet1')

    string_prd = 'PRD : temperature (K)'
    string_inj = 'INJ : temperature (K)'
    col_prd = [col for col in td.columns if string_prd in col][0]
    col_inj = [col for col in td.columns if string_inj in col][0]
    T0_prd = td[col_prd].iloc[0]
    T0_inj = td[col_inj].iloc[0]
    threshold = T0_prd - 0.15 * (T0_prd - T0_inj)
    try:
        print('lifetime = %d years' % (td['time'][td[col_prd] <= threshold].iloc[0] / 365))
    except IndexError:
        print('LIFETIME NOT REACHED')

def run_main(input):
    if input[2][0] == 0:
        dir = 0
        output_directory = f'output/{input[0][0]}/q={input[2][0]}, WR={input[3][0]}/{input[1][0]}'
        os.makedirs(output_directory, exist_ok=True)
        main(input, output_directory, dir)
    else:
        dir = [0, 45, 90, 135, 180, 225, 270, 315]  
        for i in range(len(dir)):
            output_directory = f'output/{input[0][0]}/q={input[2][0]}, WR={input[3][0]}/{input[1][0]}, dir={dir[i]}'
            os.makedirs(output_directory, exist_ok=True)
            main(input, output_directory, dir[i])   

# RUN MAIN WITH ALL INPUTS
for i, run in enumerate(Runs):
    input = [[val] for val in run]

    run_main(input=input)
