from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

Runs = [#[Prod/Recharge,   model,        q (m/s), WR (m3/day), TEST_yrs_prd, TEST_yrs_recharge]
        ['Recharge',     'model 1', 0, 8000,         40,         0],
        ['Recharge',     'model 0',     2.3e-07, 8000,         40,         0],
        ['Recharge',     'model 1',     2.3e-07, 8000,         40,         0],
        ['Recharge',     'model 2',     2.3e-07, 8000,         40,         0],
        ['Recharge',     'model 3',     2.3e-07, 8000,         40,         0],
        ['Recharge',     'model 4',     2.3e-07, 8000,         40,         0],] 

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

def main(input, output_directory, dir):
    model_lifetimes = {'homogeneous':  44,   # DO IF STATEMENT FOR T_PRD<345.2
                       'model 0':      24,
                       'model 1':      11,
                       'model 2':      15,
                       'model 3':      25,
                       'model 4':      59}
    rp = {'model_name': input[1][0],
          'q':          input[2][0],
          'dir':        dir,
          'WR':         input[3][0]}
    
    m = Model(run_params=rp, iapws_physics=True)
    m.init(discr_type='mpfa', output_folder=output_directory)

    if input[0][0] == 'Production':
        m.run(days=100*365, verbose=False)
        m.output_to_vtk(ith_step=0, output_directory=output_directory)
        m.output_to_vtk(ith_step=1, output_directory=output_directory)

    elif input[0][0] == 'Recharge':
        lifetime_years = model_lifetimes[rp['model_name']]
        days_recharge = 2500*365
        m.run(days=lifetime_years*365, verbose=False)
        m.output_to_vtk(ith_step=0, output_directory=output_directory)
        m.output_to_vtk(ith_step=1, output_directory=output_directory)

        m.set_well_controls(rate=0)
        m.run(days=40*365, restart_dt=m.params.first_ts, verbose=False)  # 40 years
        m.set_sim_params(max_ts=3650)      # max_ts = 10 years
        m.run(days=days_recharge - 40*365, verbose=False)
        m.output_to_vtk(ith_step=2, output_directory=output_directory)

    else:     # TEST runs
        days_prod     = input[4][0]*365
        days_recharge = input[5][0]*365

        m.run(days=days_prod, verbose=False)
        m.output_to_vtk(ith_step=0, output_directory=output_directory)
        m.output_to_vtk(ith_step=1, output_directory=output_directory)

        if days_recharge != 0:
            m.set_well_controls(rate=0)
            m.run(days=40*365, restart_dt=m.params.first_ts, verbose=False)
            m.set_sim_params(max_ts=3650)
            m.run(days=days_recharge - 40*365, verbose=False)
            m.output_to_vtk(ith_step=2, output_directory=output_directory)

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

# RUN MAIN WITH ALL INPUTS
for i, run in enumerate(Runs):
    input = [[val] for val in run]

    run_main(input=input)
