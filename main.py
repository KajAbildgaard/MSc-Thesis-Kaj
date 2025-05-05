from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

output_directory = r'output/Production/q=3.1e-07/homogeneous, dir=90'
os.makedirs(output_directory, exist_ok=True)

days_prod = 40*365
days_recharge = 0*365

m = Model(iapws_physics=True)
m.init(discr_type='mpfa', output_folder=output_directory) 

m.run(days=days_prod, verbose=False)                                # for t in range(years_prod):
m.output_to_vtk(ith_step=0, output_directory=output_directory)      #     m.run(365)
m.output_to_vtk(ith_step=1, output_directory=output_directory)      # m.set_well_controls(rate=0)

# m.set_well_controls(rate=0)                                         # for t in range(years_recharge//50): #output each 50 years  
# m.run(days=days_recharge, verbose=False)                            #     m.run(50*365)
# m.output_to_vtk(ith_step=2, output_directory=output_directory)      # m.output_to_vtk(output_directory=output_directory)

m.print_timers()

td = pd.DataFrame.from_dict(m.physics.engine.time_data)
td_path = os.path.join(output_directory, 'darts_time_data.pkl')
td.to_pickle(td_path)

excel_path = os.path.join(output_directory, 'time_data.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    td.to_excel(writer, sheet_name='Sheet1')

string = 'PRD : temperature'
ax1 = td.plot(x='time', y=[col for col in td.columns if string in col])
col_name = [col for col in td.columns if string in col]

array = td[[col for col in td.columns if string in col]].to_numpy()
print('lifetime = %d years' % (td['time'][td[col_name[0]] < 348].iloc[0] / 365))
