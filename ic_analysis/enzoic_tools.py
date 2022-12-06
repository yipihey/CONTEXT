# Collect useful routines to read and operate on enzo initial conditions as written by MUSIC
# Author: Tom Abel 10/2022
#
import h5py
import numpy as np
import configparser

CONFIG_FILE = 'parameter_file.txt'

PFLOAT = np.float128
FLOAT = np.float64

SINT  = np.uint16
UNDEFINED = -1

# Functions and classes

def read_sset_hdf5(filename):
    """ read single set from hfd5 file"""
    with h5py.File(filename, "r") as f:
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])
        return data

def read_xyz(fdir,fnums,fstub="ParticleDisplacements_"):
    pt = []
    for fnum in fnums:
        pi = []
        for idim, dim in enumerate(["x.","y.","z."]):
            fname = fdir + fstub + dim + str(fnum)
            data = read_sset_hdf5(fname)
            pi.append( data[0] )
        pt.append(np.array(pi))
    return pt

def read(fdir,fnums,fstub="GridDensity."):
    pt = []
    for fnum in fnums:
        pi = []
        fname = fdir + fstub + str(fnum)
        data = read_sset_hdf5(fname)
        pt.append(np.array(data[0]))
    return pt

from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pylab as plt

def plot_grids(dims, gd, vmin=-1.2, vmax=1.2,length_top=65536.,levels=-1):
    top_mean = (gd[0]).mean()
    for pdim in dims:
        fig = plt.figure(figsize=(40,20))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(4, 3),  # creates 4x4 grid of axes
                        axes_pad=0.,  # pad between axes in inch.
                        cbar_mode="single",
                        cbar_pad="1%",
                        cbar_size=0.5
                        )

        c = 0
        for ax, im, cb in zip(grid, gd[:], grid.cbar_axes):
            # Iterating over the grid returns the Axes.
            this_mean = im.mean()
            std = (im/this_mean-1).std()
            mim = ax.imshow((np.mean(im,axis=pdim )/this_mean - 1)/std,vmin=vmin, vmax=vmax,
                        cmap="bwr")
            if levels[0] == -1:
                level = c # if levels not defined assumed it goes from 0 onwards
            else: 
                level = levels[c]
            ax.annotate(str("{width:.2g}").format(width=length_top/2**level * im.shape[0]/128)+"Mpc/h",xy=(10,10),color="black",size=15)
            ax.annotate(str("mean={this_mean:.2e}").format(this_mean=this_mean),xy=(10,18),color="black",size=12)
            ax.annotate(str("std={std:.2e}").format(std=std),xy=(10,26),color="black",size=12)
            ax.annotate(str("min={min:.2e}").format(min=im.min()),xy=(10,34),color="black",size=12)
            c += 1
        grid.cbar_axes[0].colorbar(mim,label="std")
        plt.show()

class parameters():
    def __init__(self, config_path="./", file_name="parameter_file.txt"):
        with open(config_path+file_name, 'r') as f:  # https://stackoverflow.com/questions/2819696/parsing-properties-file-in-python/25493615#25493615
            config_string = '[enzo]\n' + f.read()
        config = configparser.ConfigParser()
        config.read_string(config_string)
        self.params = config["enzo"]
        self.path = config_path
        self.file_name = file_name
        
# Could write some helper functions but should probably use what configParser provides?
#        def par(self, string):
#            ret = -1 
#            FLOAT_array_list =[""]
#            isFLOATa = string in FLOAT_array_list
#            if isFLOATa:
#                return np.array(string.split(),dtype=FLOAT)
#            elif isPFLOATa: 
#                return np.array(string.split(),dtype=PFLOAT)
#            elif isSINTa:
#                return np.array(string.split(),dtype=SINT)


class patch:
    def __init__(self,  GridRank=3, 
                        LeftEdge=UNDEFINED, 
                        RightEdge=UNDEFINED, 
                        Dimensions=UNDEFINED
            ):
        self.GridRank = GridRank
        if LeftEdge is UNDEFINED:
            self.LeftEdge = np.zeros(self.GridRank, dtype=PFLOAT)    
        else:
            self.LeftEdge = LeftEdge
        if RightEdge is UNDEFINED:
            self.RightEdge = np.ones(self.GridRank, dtype=PFLOAT)
        else:
            self.RightEdge = RightEdge
        if Dimensions is UNDEFINED:
            self.Dimensions = np.zeros(self.GridRank, dtype=SINT)
        else:
            self.Dimensions = Dimensions        

        self.BinaryLevel = np.int64(-np.log2(((self.RightEdge)[0]-self.LeftEdge[0])/self.Dimensions[0]))
        self.dx = (self.RightEdge-self.LeftEdge)/self.Dimensions

class hierarchy:
    verbose = False
# keep global counter of how many we have
    def __init__(self, path="./", parameter_file="./parameter_file.txt"):

        self.path = path
        self.parameter_file = parameter_file
        self.params = parameters(config_path=path, file_name=parameter_file).params

        Ngrids = SINT(self.params["CosmologySimulationNumberOfInitialGrids"])
        TopGridRank = SINT(self.params["TopGridRank"])
        topgrid = patch(GridRank=3, Dimensions=np.array(self.params["TopGridDimensions"].split(),dtype=SINT))
        topgrid.GridNum = 0
        self.grids = [topgrid]

        for i in range(1,Ngrids):
            stb = "[" + str(i) + "]"
            cg = patch(GridRank=TopGridRank, 
                Dimensions=np.array(self.params["CosmologySimulationGridDimension"+stb].split(),dtype=SINT), 
                LeftEdge=np.array(self.params["CosmologySimulationGridLeftEdge"+stb].split(),dtype=PFLOAT), 
                RightEdge=np.array(self.params["CosmologySimulationGridRightEdge"+stb].split(),dtype=PFLOAT)
                )
            cg.GridNum = i
            self.myprint("GridNum: "+str(i))
            self.grids.append(cg)
        self.NumberOfGrids = Ngrids

    def clear(self):
        for cg in self.grids:
            del cg
        self.grids =[]
        self.NumberOfGrids = 0

    def myprint(self, string):
        if hierarchy.verbose:
            print(string)
        else:
            pass

    def read_density(self):
        for cg in self.grids:
            self.myprint("read: "+self.path+self.params["CosmologySimulationDensityName"]+'.'+str(cg.GridNum))
            cg.density = read(self.path,[cg.GridNum],fstub=self.params["CosmologySimulationDensityName"]+'.')[0]
            self.myprint(cg.density.shape)

    def read_grid_velocities(self):
        for cg in self.grids:
            fname = self.path+self.params["CosmologySimulationVelocity1Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            x = read_sset_hdf5(fname)[0]
            fname = self.path+self.params["CosmologySimulationVelocity2Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            y = read_sset_hdf5(fname)[0]
            fname = self.path+self.params["CosmologySimulationVelocity3Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            z = read_sset_hdf5(fname)[0]
            cg.grid_velocities =  np.array([x,y,z])    

    def set_xyz(self):
        for cg in self.grids:
            x = cg.dx[0]*(np.arange(cg.Dimensions[0])+0.5) + cg.LeftEdge[0]
            y = cg.dx[1]*(np.arange(cg.Dimensions[1])+0.5) + cg.LeftEdge[1]
            z = cg.dx[2]*(np.arange(cg.Dimensions[2])+0.5) + cg.LeftEdge[2]
            self.myprint("set_xyz")
            cg.xyz = np.array(np.meshgrid(x, y, z,indexing="ij"))

    def read_displacements(self):
        for cg in self.grids:
            fname = self.path+self.params["CosmologySimulationParticleDisplacement1Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            x = read_sset_hdf5(fname)[0]
            fname = self.path+self.params["CosmologySimulationParticleDisplacement2Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            y = read_sset_hdf5(fname)[0]
            fname = self.path+self.params["CosmologySimulationParticleDisplacement3Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            z = read_sset_hdf5(fname)[0]
            cg.displacements = np.array([x,y,z])

    def read_particle_velocities(self):
        for cg in self.grids:
            fname = self.path+self.params["CosmologySimulationParticleVelocity1Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            x = read_sset_hdf5(fname)[0]
            fname = self.path+self.params["CosmologySimulationParticleVelocity2Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            y = read_sset_hdf5(fname)[0]
            fname = self.path+self.params["CosmologySimulationParticleVelocity3Name"]+'.'+str(cg.GridNum)
            self.myprint("read: "+fname)
            z = read_sset_hdf5(fname)[0]
            cg.particle_velocities =  np.array([x,y,z])    

        