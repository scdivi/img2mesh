#!/usr/bin/env python3
from nutils import *
from matplotlib import collections, colors
import os, pathlib
from scan import get_scandata, TopoMap, nutils_topo
from immersed_meshes import get_background_mesh, get_skeleton_mesh, \
     get_ghost_mesh, get_fragment_topo

# main
def main(fname         = "walle_70",
         nelems        = 18        ,
         maxrefine     = 3         ,
         topopreserve  = False
         ):

    fdir = f'data/{fname}'
    topo, ttopo, geom = nutils_topo(fdir, nelems, maxrefine, topopreserve)

    bezier = ttopo.sample('bezier', 2**maxrefine+1)
    points, vals = bezier.eval([geom, 0])
    with export.mplfigure('trimmed.png') as fig:
      ax = fig.add_subplot(111, aspect=1)
      ax.autoscale(enable=True, axis='both', tight=True)
      cmap = colors.ListedColormap("darkgray")
      im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, vals, shading='gouraud', cmap=cmap)
      ax.axis('off')

    # Write tikz data files
    tikzdir = 'tikz/'
    if not os.path.exists(f'{tikzdir}'):
        os.mkdir(f'{tikzdir}')

    #####################
    # write domain tikz # 
    #####################
    # sample points
    bezier       = ttopo.sample('bezier', 4) #2**maxrefine+1)
    points, vals = bezier.eval([geom, 0.])
    with open(f'{tikzdir}{fname}_domain.dat','w') as fout:
      for tri in bezier.tri:
        verts = points[tri]
        fout.write('\\filldraw {} -- cycle;\n'.format(' -- '.join('({:7.5f},{:7.5f})'.format(*v) for v in verts)))
  
    #####################
    # write octree tikz # 
    #####################
    # fgragment topo TODO
    # ftopo = get_fragment_topo(ttopo)

    # # sample fragmented topo
    # fbezier = ftopo.sample('bezier', 2)
    # fpoints = fbezier.eval(geom)
    # with open(f'{tikzdir}{fname}_octree.dat','w') as fout:
    #   for seg in fbezier.hull:
    #     sverts = fpoints[seg,:]
    #     fout.write('\draw {} -- {};\n'.format(*tuple('({:7.5f},{:7.5f})'.format(*v) for v in sverts)))

    #######################
    # write boundary tikz # 
    #######################
    bnd_bezier = ttopo.boundary['trimmed'].sample('bezier', 2)
    bnd_points = bnd_bezier.eval(geom)
    bx = bnd_points[:,0]
    by = bnd_points[:,1]
    bsegs = []
    for i in range(int(len(bnd_points)/2)):
      bsegs.append([[bx[2*i], by[2*i]], [bx[2*i+1], by[2*i+1]]])
    with open(f'{tikzdir}{fname}_boundary.dat','w') as fout:
      for sverts in bsegs:
        fout.write('\draw {} -- {};\n'.format(*tuple('({:7.5f},{:7.5f})'.format(*v) for v in sverts)))

    ###########################
    # write ambient mesh tikz # 
    ###########################
    abezier = topo.sample('bezier', 2)
    apoints = abezier.eval(geom)
  
    with open(f'{tikzdir}{fname}_ambient.dat','w') as fout:
      for seg in abezier.hull:
        sverts = apoints[seg,:]
        fout.write('\draw {} -- {};\n'.format(*tuple('({:7.5f},{:7.5f})'.format(*v) for v in sverts)))
  
    ##############################
    # write background mesh tikz # 
    ##############################
    # get background mesh
    background = get_background_mesh(ttopo, topo)

    # sample background mesh
    bbezier = background.sample('bezier', 2)
    bpoints = bbezier.eval(geom)

    with open(f'{tikzdir}{fname}_background.dat','w') as fout:
      for seg in bbezier.hull:
        sverts = bpoints[seg,:]
        fout.write('\draw {} -- {};\n'.format(*tuple('({:7.5f},{:7.5f})'.format(*v) for v in sverts)))

    # plot background mesh
    with export.mplfigure('background_mesh.png') as fig:
      ax = fig.add_subplot(111, aspect = 'equal')
      ax.autoscale(enable=True, axis='both', tight=True)
      cmap = colors.ListedColormap("darkgray")
      im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, vals, shading='gouraud', cmap=cmap)
      ax.add_collection( collections.LineCollection( bpoints[bbezier.hull], colors='k', linewidth=1) )
      ax.axis('off')

    ############################
    # write skeleton mesh tikz # 
    ############################
    # skeleton mesh
    skeleton = get_skeleton_mesh(ttopo, topo)

    # sample skeleton mesh
    sbezier = skeleton.sample('bezier', 2)
    spoints = sbezier.eval(geom)
    # segments of skeleton mesh
    sx = spoints[:,0]
    sy = spoints[:,1]
    ssegs = []
    for i in range(int(len(spoints)/2)):
      ssegs.append([[sx[2*i], sy[2*i]], [sx[2*i+1], sy[2*i+1]]])

    with open(f'{tikzdir}{fname}_skeleton.dat','w') as fout:
      for sverts in ssegs:
        fout.write('\draw {} -- {};\n'.format(*tuple('({:7.5f},{:7.5f})'.format(*v) for v in sverts)))

    # plot skeleton mesh
    with export.mplfigure('skeleton_mesh.png') as fig:
      ax = fig.add_subplot(111, aspect = 'equal')
      ax.autoscale(enable=True, axis='both', tight=True)
      cmap = colors.ListedColormap("darkgray")
      im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, vals, shading='gouraud', cmap=cmap)
      ax.add_collection( collections.LineCollection( ssegs, colors='k', linewidth=2) )
      ax.axis('off')

    #########################
    # write ghost mesh tikz # 
    #########################
    # ghost mesh
    ghost = get_ghost_mesh(ttopo, topo, skeleton)

    # sample ghost mesh
    gbezier = ghost.sample('bezier', 2)
    gpoints = gbezier.eval(geom)
    # segments of ghost mesh
    gx = gpoints[:,0]
    gy = gpoints[:,1]
    gsegs = []
    for i in range(int(len(gpoints)/2)):
      gsegs.append([[gx[2*i], gy[2*i]], [gx[2*i+1], gy[2*i+1]]])

    with open(f'{tikzdir}{fname}_ghost.dat','w') as fout:
      for sverts in gsegs:
        fout.write('\draw {} -- {};\n'.format(*tuple('({:7.5f},{:7.5f})'.format(*v) for v in sverts)))

    # plot ghost mesh
    with export.mplfigure('ghost_mesh.png') as fig:
      ax = fig.add_subplot(111, aspect = 'equal')
      ax.autoscale(enable=True, axis='both', tight=True)
      cmap = colors.ListedColormap("darkgray")
      im = ax.tripcolor(points[:,0], points[:,1], bezier.tri, vals, shading='gouraud', cmap=cmap)
      ax.add_collection( collections.LineCollection( gsegs, colors='k', linewidth=2) )
      ax.axis('off')


if __name__ == '__main__':
  cli.run(main)
