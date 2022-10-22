# %%
from gettext import npgettext
import pyvista as pv
import numpy as np
from pathlib import Path
import random
from scipy.spatial import cKDTree as KDTree
# %% load scan
data_scan = Path(
    "/Users/hassanmallahzadeh/UBCLife/Fall-2022/CPSC535P/assignment1/data/scan_noisy.obj")
data_template = Path(
    "/Users/hassanmallahzadeh/UBCLife/Fall-2022/CPSC535P/assignment1/data/template.obj")
scan = pv.read(data_scan)
templ = pv.read(data_template)
templ_orig = templ.copy()
n = 8000#number of points to choose
y0 = np.sum(scan.points,axis=0)/scan.points.shape[0]
x0 = np.sum(templ.points,axis=0)/templ.points.shape[0]
templ.points = templ.points - x0 + y0 
#scan_short = scan.extract_feature_edges(boundary_edges=True).points
#scan_short = random.choices(scan_short.points, k=n)
niter = 40
for i in range(niter):
 
    scan_short = scan.points
    tree = KDTree(templ.points.astype(np.double))
    dist, idx = tree.query(scan_short)
    templ_correspond = templ.points[idx]
    y0 = np.sum(scan_short,axis=0)/scan_short.shape[0]
    x0 = np.sum(templ_correspond,axis=0)/templ_correspond.shape[0]
    x_x0 = templ_correspond - x0
    y_y0 = scan_short - y0
    H = np.zeros((3,3))
    for idx, x in enumerate(x_x0):
        H = np.add(H,np.outer(y_y0[idx] , x_x0[idx]))
    [U, D, Vt] = np.linalg.svd(H)
    R = np.dot(np.transpose(Vt), np.transpose(U))
    R = R.T
    y0 = np.array([y0])
    x_new = np.transpose(np.matmul(R,np.transpose(templ.points - x0)))+y0
    templ.points = x_new
# %% visualize data
pl = pv.Plotter()
mesht = pl.add_mesh(templ_orig, show_edges=True, color='gray', opacity=0.5)
mesht = pl.add_mesh(templ, show_edges=True, color='blue', opacity=0.5)
mesht = pl.add_mesh(scan, show_edges=True, color='red', opacity=0.5)
pl.camera_position = 'xy'
# # TODO for some reason the interactive widget doesn't work in my VSCode. Ideas?
pl.show(title='template vs scan', jupyter_backend='static')