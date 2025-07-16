from darts.engines import value_vector, redirect_darts_output

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

Runs = [#[Prod/Recharge,   model,        q (m/s), WR (m3/day), TEST_yrs_prd, TEST_yrs_recharge]
        ['temptest',     'model 4',1.15e-07,        4000,          50,         0],]
   

def main(input, output_directory, dir):
    rp = {'model_name': input[1][0],
          'q':          input[2][0],
          'dir':        dir,
          'WR':         input[3][0]}
    
    m = Model(run_params=rp, iapws_physics=True)
    m.init(discr_type='mpfa', output_folder=output_directory)
    redirect_darts_output(os.path.join(output_directory, 'run.log'))

    if input[0][0] == 'Production':
        if input[2][0] == 0:  
            for t in range(100):
                m.run(days=365, verbose=False)
            td = m.physics.engine.time_data
            threshold = m.compute_threshold(td)
            years = 100
            max_years = 1000
            while years < max_years:
                last_T = td[next(k for k in td.keys() if "PRD : temperature (K)" in k)][-1]
                if last_T <= threshold:
                    break
                m.run(days=365, verbose=False)
                years += 1
            m.output_to_vtk(ith_step=0, output_directory=output_directory)
            m.output_to_vtk(ith_step=years, output_directory=output_directory)

        elif input[2][0] != 0:  
            years_base = m.load_or_error(input[3][0], input[1][0])
            for t in range(100):
                m.run(days=365, verbose=False)
            years = 100
            if years_base > 100:
                while years < years_base:
                    m.run(days=365, verbose=False)
                    years += 1
            m.output_to_vtk(ith_step=0, output_directory=output_directory)
            m.output_to_vtk(ith_step=years, output_directory=output_directory)

    elif input[0][0] == 'Recharge':
        m.run(days=365, verbose=False)
        td = m.physics.engine.time_data
        threshold = m.compute_threshold(td)
        years = 1
        max_years = 1000
        while years < max_years:
            last_T = td[next(k for k in td.keys() if "PRD : temperature (K)" in k)][-1]  
            if last_T <= threshold:
                break
            m.run(days=365, verbose=False)
            years += 1
        m.output_to_vtk(ith_step=0, output_directory=output_directory)
        m.output_to_vtk(ith_step=years, output_directory=output_directory)

        years_recharge = 2500  
        m.set_well_controls(rate=0)
        for i in range(40):
            if i == 0:
                m.run(days=365, restart_dt=m.params.first_ts, verbose=False)
                years += 1
            else:
                m.run(days=365, verbose=False)
                years += 1
        m.set_sim_params(max_ts=3650)                                 
        for i in range((years_recharge-40)//10):
            m.run(days=3650, verbose=False)
            years += 1                          #years = nr. of m.run
        m.output_to_vtk(ith_step=years, output_directory=output_directory)

    else:     # TEST runs
        years_prod     = input[4][0]
        years_recharge = input[5][0]
        years = 0

        for i in range(years_prod):
            m.run(days=365, verbose=False)
            years += 1
        m.output_to_vtk(ith_step=0, output_directory=output_directory)
        m.output_to_vtk(ith_step=years, output_directory=output_directory)

        if years_recharge != 0:
            m.set_well_controls(rate=0)
            for i in range(40):
                if i == 0:
                    m.run(days=365, restart_dt=m.params.first_ts, verbose=False)
                    years += 1
                else:
                    m.run(days=365, verbose=False)
                    years += 1
            m.set_sim_params(max_ts=3650)                                 
            for i in range((years_recharge-40)//10):
                m.run(days=3650, verbose=False)
                years += 1                         
            m.output_to_vtk(ith_step=years, output_directory=output_directory)

    m.print_timers()

    td = pd.DataFrame.from_dict(m.physics.engine.time_data)
    cols = ["time",
            "INJ : temperature (K)",
            "INJ : water rate (m3/day)",
            "INJ : steam rate (m3/day)",
            "INJ : energy (kJ/day)",
            "INJ : BHP (bar)",
            "PRD : water rate (m3/day)",
            "PRD : steam rate (m3/day)",
            "PRD : temperature (K)",
            "PRD : energy (kJ/day)",
            "PRD : BHP (bar)"]
    td = td[cols]
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
