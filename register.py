# %%
from gettext import npgettext
import pyvista as pv
import numpy as np
from pathlib import Path
import random
from scipy.spatial import cKDTree as KDTree
data_scan = Path(
    "/Users/hassanmallahzadeh/UBCLife/Fall-2022/CPSC535P/assignment1/data/scan_posed.obj")
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
niter = 5
scan = scan.compute_normals(cell_normals=False)
for i in range(niter):
    templ = templ.compute_normals(cell_normals=False)
    tree = KDTree(templ.points.astype(np.double))
    dist, ids_templcorrespond = tree.query(scan.points)#index: scan id, value:template id
    print(len(ids_templcorrespond))
    print(len(dist))
  #  todelete = np.empty((0,0), int)
    #same index in scan_chosen & temp_chosen correspond to eachother.
    scan_chosen = np.empty((0,0), int)
    templ_chosen = np.empty((0,0), int)
    for id_scan,id_templ in enumerate(ids_templcorrespond):
        n1 = scan.active_normals[id_scan,:]
        n2 = templ.active_normals[id_templ,:]
       # print(np.dot(n1,n2))
        if np.dot(n1,n2) > 0:
            scan_chosen = np.append(scan_chosen,id_scan)
            templ_chosen = np.append(templ_chosen,id_templ)
    # ids_templcorrespond = np.delete(ids_templcorrespond,todelete)
    # scan_chosen = np.delete(scan_chosen,todelete)
    y0 = np.sum(scan.points[scan_chosen],axis=0)/scan_chosen.shape[0]
    x0 = np.sum(templ.points[templ_chosen],axis=0)/templ_chosen.shape[0]
    x_x0 = templ.points[templ_chosen] - x0
    y_y0 = scan.points[scan_chosen] - y0
    # %%
    H = np.zeros((3,3))
    for id, val in enumerate(x_x0):
        H = np.add(H,np.outer(y_y0[id] , x_x0[id]))
    [U, D, Vt] = np.linalg.svd(H)
    R = np.dot(np.transpose(Vt), np.transpose(U))
    R = R.T
    y0 = np.array([y0])
    x_new = np.transpose(np.matmul(R,np.transpose(templ.points - x0)))+y0
    templ.points = x_new
# %% visualize data
pl = pv.Plotter()
mesht = pl.add_mesh(templ, show_edges=True, color='gray', opacity=0.5)
#mesht = pl.add_mesh(templ, show_edges=True, color='blue', opacity=0.5)
mesht = pl.add_mesh(scan, show_edges=True, color='red', opacity=0.5)
pl.camera_position = 'xy'
# # TODO for some reason the interactive widget doesn't work in my VSCode. Ideas?
pl.show(title='template vs scan', jupyter_backend='static')