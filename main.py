from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

Inputs = [['Production'],     # Production or Recharge
          ['model 4'],        # Model name
          [3.1e-07],          # Background flow rate (m/s) 3.1e-07
          [8000],             # Well rate (m3/day)

          [40],               # Number of production years
          [2500]]             # Number of recharge years     

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
    days_prod = input[4][0]*365
    days_recharge = input[5][0]*365

    rp = {
    'model_name': input[1][0],
    'q':           input[2][0],
    'dir':         dir,
    'WR':          input[3][0]}

    m = Model(run_params=rp, iapws_physics=True)
    m.init(discr_type='mpfa', output_folder=output_directory) 

    m.run(days=days_prod, verbose=False)                                
    m.output_to_vtk(ith_step=0, output_directory=output_directory)      
    m.output_to_vtk(ith_step=1, output_directory=output_directory)      

    if input[0][0] == 'Recharge':
        m.set_well_controls(rate=0)                                          
        m.run(days=40*365, verbose=False) # 40 years, max_ts=365
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
input = {}
for i in range(len(Inputs[0])):  
    input[i] = [[row[i]] for row in Inputs]

    run_main(input=input[i])
