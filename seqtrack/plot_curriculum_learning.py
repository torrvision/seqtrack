import pdb
import numpy as np
import os

import matplotlib as mpl
mpl.use('Agg') # generate images without having a window appear
import matplotlib.pyplot as plt


def getColormap(N,cmapname):
    '''Returns a function that maps each index in 0, 1, ..., N-1 to a distinct RGB color.'''
    colornorm = mpl.colors.Normalize(vmin=0, vmax=N-1)
    scalarmap = mpl.cm.ScalarMappable(norm=colornorm, cmap=cmapname)
    def mapIndexToRgbColor(index):
        return scalarmap.to_rgba(index)
    return mapIndexToRgbColor

def plot_curriculum_learning():
    # Data
    t2_40k = 208 * 72 / 3600.
    t4_40k = 394 * 72 / 3600.
    t8_40k = 764 * 72 / 3600.
    t16_40k = 1530 * 72 / 3600.
    t32_40k = 2250 * 72 / 3600.
    m0 = {
          'TRE': [0.555, 0.562, 0.569, 0.564, 0.569, 0.568],
          'OPE': [0.508, 0.507, 0.509, 0.512, 0.518, 0.516],
          'time': np.cumsum([t8_40k]*6),
          'name': 'No curriculum learning',
          'T': 8,
          }
    m1 = {
          'TRE': [0.493, 0.531, 0.562, 0.568, 0.560, 0.558],
          'OPE': [0.420, 0.464, 0.515, 0.536, 0.506, 0.510],
          'time': np.cumsum([t2_40k]*2 + [t8_40k]*4),
          'name': 'CL-Single[T2]',
          'T': 8,
          }
    m2 = {
          'TRE': [0.516, 0.535, 0.565, 0.569, 0.562, 0.560],
          'OPE': [0.450, 0.468, 0.514, 0.518, 0.515, 0.511],
          'time': np.cumsum([t4_40k]*2 + [t8_40k]*4),
          'name': 'CL-Single[T4]',
          'T': 8,
          }
    m3 = {
          'TRE': [0.579, 0.568, 0.576, 0.598, 0.588, 0.578],
          'OPE': [0.531, 0.517, 0.543, 0.554, 0.552, 0.533],
          'time': np.cumsum([t32_40k]*6),
          'name': 'No curriculum learning',
          'T': 32,
          }
    m4 = {
          'TRE': [0.507, 0.527, 0.557, 0.577, 0.579], # TODO: need to update
          'OPE': [0.449, 0.453, 0.501, 0.523, 0.524], # TODO: need to update
          'time': np.cumsum([t2_40k, t4_40k, t8_40k, t16_40k, t32_40k]), # TODO: need to update
          'name': 'CL-Progressive',
          'T': 32,
          }

    # Plot all in one.
    mpl.rcParams.update({'font.size': 20})
    mpl.rcParams['axes.linewidth'] = 3
    mpl.rcParams['xtick.major.pad']='8'
    mpl.rcParams['ytick.major.pad']='8'
    mpl.rcParams.update({'figure.autolayout': True})
    col = getColormap(12, 'Set3')

    fig = plt.figure(figsize=(18,7))
    for i, metric in enumerate(['OPE', 'TRE']):
        plt.subplot(1,2,i+1)
        plt.plot(m0['time'], m0[metric], '--^', c=col(0), linewidth=4.0, label='(T={}) '.format(m0['T']) + m0['name'])
        plt.plot(m1['time'], m1[metric], '-^',  c=col(9), linewidth=4.0, label='(T={}) '.format(m1['T']) + m1['name'])
        plt.plot(m2['time'], m2[metric], '-^',  c=col(4), linewidth=4.0, label='(T={}) '.format(m2['T']) + m2['name'])
        plt.plot(m3['time'], m3[metric], '--^', c=col(5), linewidth=4.0, label='(T={}) '.format(m3['T']) + m3['name'])
        plt.plot(m4['time'], m4[metric], '-^',  c=col(3), linewidth=4.0, label='(T={}) '.format(m4['T']) + m4['name'])
        plt.axis([0, 300, 0.40, 0.65])
        plt.grid()
        plt.ylabel('Performance (IoU)', fontsize=20, labelpad=8)
        plt.xlabel('Time (hours)', fontsize=20, labelpad=8)
        plt.title('{}'.format(metric), y=1.04)
        plt.legend(frameon=False, loc=4, fontsize=16, handlelength=5)
    plt.savefig('results_CL.pdf')
    plt.close()

if __name__ == '__main__':
    plot_curriculum_learning()
