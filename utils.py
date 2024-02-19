import os
import wfdb
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import neurokit2 as nk
from scipy import signal
from tqdm import tqdm


def compute_distances(d1, d2):
    distances = {'MAE': 0, 'MSE': 0, 'RMSE': 0, 'Correlation': [0]*d1[0].shape[1], 'Relative Deviation': []}
    total = min(len(d1), len(d2))
    for i in range(total):
        distances['MAE'] += np.mean(np.abs(d1[i] - d2[i])) / total
        distances['MSE'] += np.mean((d1[i] - d2[i])**2) / total
        distances['RMSE'] += np.sqrt(np.mean((d1[i] - d2[i])**2)) / total
        distances['Relative Deviation'] += np.mean(np.abs((d1[i] - d2[i]) / d1[i])) * 100 / total
        for j in range(d1[0].shape[1]):
            distances['Correlation'][j] += np.abs(np.mean(np.multiply(d1[i][j], d2[i][0])) / np.sqrt(np.mean(d1[i][j]**2) * np.mean(d2[i][0]**2))) / total

    return distances

def vcgplot(vcg, N, label = 'VCG'):

    x, y, z = vcg[:N,0], vcg[:N,1], vcg[:N,2]

    # Create a set of line segments
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the 3D-line collection object
    t = np.linspace(1,N,N)
    lc = Line3DCollection(segments, color=plt.cm.viridis(t / np.max(t)), linewidth=2)

    # Create a 3D figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=120, elev=20)
    ax.set_xlabel('X',labelpad=15, fontsize = 15)
    ax.set_ylabel('Y',labelpad=15, fontsize = 15)
    ax.set_zlabel('Z',labelpad=15, fontsize = 15)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.yaxis._axinfo["grid"].update({"linestyle": 'dashed'})
    ax.xaxis._axinfo["grid"].update({"linestyle": 'dashed'})
    ax.zaxis._axinfo["grid"].update({"linestyle": 'dashed'})
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))
    ax.add_collection3d(lc)

    # Add colorbar
    #m = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(np.min(z), np.max(z)))
    #m.set_array(z)
    #cbar = fig.colorbar(m, shrink=0.5)
    #cbar.ax.set_ylabel('Time', rotation=0)
    #cbar.set_ticks([])
    #cbar.set_label(r'Time', labelpad=27, y=0.45, fontsize = 14)

    plt.title(label)
    plt.grid(True, alpha = 0.1)
    plt.show()

def read_record(filepath):
    record = wfdb.rdrecord(filepath)
    return record.__dict__['p_signal'], record.__dict__['comments']

def read_PTB(clean = False):
    data_paths = []
    PTB_PATH = 'data/ptb-diagnostic-ecg-database-1.0.0'
    INFO_file = f'{PTB_PATH}/RECORDS'
    with open(INFO_file) as f:
        data_paths = f.readlines()
        data_paths = [x.strip() for x in data_paths]

    dataset = []
    for record_dir in tqdm(data_paths):
        if os.path.exists(f'{PTB_PATH}/{record_dir}.hea'):
            record, comment = read_record(f'{PTB_PATH}/{record_dir}')
            if clean:
                ecg = np.zeros((15, 1000))
                for j in range(15):
                    clean_ecg = np.array(nk.ecg_clean(record[:, j][:4000], sampling_rate=1000))
                    ecg[j] = signal.resample(clean_ecg, 1000)
                dataset.append((ecg.T,comment))
            else:
                dataset.append((record, comment))      
            #dataset.append((record, comment[4].split(':')[1].strip()))

    return dataset