from darts.engines import value_vector

from model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


years = 20
m = Model(iapws_physics=True)
m.init()

for t in range(years):
    m.run(365)
m.output_to_vtk()
m.print_timers()


td = pd.DataFrame.from_dict(m.physics.engine.time_data)
td.to_pickle("darts_time_data.pkl")
writer = pd.ExcelWriter('time_data.xlsx')
td.to_excel(writer, sheet_name='Sheet1')
writer.close()

string = 'PRD : temperature'
ax1 = td.plot(x='time', y=[col for col in td.columns if string in col])
col_name = [col for col in td.columns if string in col]

array = td[[col for col in td.columns if string in col]].to_numpy()
print('lifetime = %d years' % (td['time'][td[col_name[0]] < 348].iloc[0] / 365))

ax1.plot([0, years*365], [348, 348])
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Days', fontsize=14)
ax1.legend(['temp', 'limit'], fontsize=14)
ax1.set(xlim=(0, years*365), ylim=(346, 351))
plt.grid()
plt.savefig('out.png')