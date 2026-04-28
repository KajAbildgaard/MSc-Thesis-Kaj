# open_darts-1.3.1 USED
from darts.engines import value_vector

from DarcyVelocity_model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

Runs = [#[Prod/Recharge,   model,        q (m/s), WR (m3/day), TEST_yrs_prd, TEST_yrs_recharge]
        ['NewDarcyinj353',     'homogeneous', 0,         8000,          50,         0],
        ['NewDarcyinj353',     'homogeneous', 0,         4000,          50,         0],] 


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

    days_prod     = input[4][0]*365
    days_recharge = input[5][0]*365

    m.run(days=days_prod, verbose=False)
    darcy_velocity = m.physics.engine.darcy_velocities
    reshaped_velocities = np.array(darcy_velocity).reshape(m.reservoir.n, m.physics.nph, 3)
    vel_path = os.path.join(output_directory, 'darcy_velocities.npy')
    np.save(vel_path, reshaped_velocities)
    m.output.output_to_vtk()

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
    if input[2][0] == 0:  # If q=0, only run for dir=0, otherwise run for all directions
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
