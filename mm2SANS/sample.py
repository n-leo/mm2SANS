import numpy as np
import scipy.constants

import matplotlib.pyplot as plt # data plotting
from matplotlib import gridspec # use this instead of indices!
from mpl_toolkits.axes_grid1 import AxesGrid

import seaborn as sns
sns.set_style('whitegrid'); sns.set_context('talk')

# colourmaps
try:
    import cmocean
    cmap_int = cmocean.cm.ice_r
    colorbar_crop_percent = 7.5
    #cmap_pm  = cmocean.tools.crop_by_percent(cmocean.cm.balance, colorbar_crop_percent, which='both', N=None)
except ImportError as e:
    print('cmocean is not installed - but it makes nice colourmaps :-)')
    cmap_int = plt.cm.plasma
    #cmap_pm  = plt.cm.bwr
    pass


class Sample:
    """ Sample class """

    def __init__(self,
                   sample_name = ''
                 # structural parameters
                 , positions=None
                 , periodicity=None
                 , number_of_unit_cells = 1
                 , voxel_volumes=None
                 , volume_correction= None
                 # magnetic parameters
                 , moments=None, saturation_magnetisation=1.
                 # neutron parameters
                 , scattering_length_density=3.
                 , print_diagnostics = True
                ):
        """
        Initialise a sample object. 
        
        :Parameters:

        *sample_name*: string
            Optional, default empty. Description of the sample.
        *print_diagnostics* Boolean
            Default True. Output remarks upon creation of the Sample object.

        ::Structural parameters::
        *R_veclist*: array (n_R, 3) | [m]
            List of position vectors. 
        *periodicity*: None, array (3) | [m, m, m]
            Default None.
            If None, the input structure has no periodic repetition, and only the structure factor will be calculated.
            periodicity = (L_x, L_y, L_z), with L_i from 0 (==0 means without repetition, e.g. for thin films).
        *R_volumes*: None, float or array (n_R, 1) | [m^3]
            Default None. Voxel volumes. (DON'T CALL IT "VOXELS"!)
            CAREFUL: If n_R < 3, voxel volumes must be explicitly set, as automatic volume calculation does not work!
            If volumes==None, the voxel volumes (assumed to be spheres) are automatically calculated. 
            Automatically values are too big, and thus are corrected by either setting one of these parameter:
                - net_volume: net volume of the entire structure
                - filling_factor: net volume is filling_factor * (volume of bounding box)
        *number_of_unit_cells* integer 
            Default 1. Number of unit cells within the input data.
            Is required to normalise the calcualted scattering signal to the formula unit.
        *volume_correction* None or (string, float)
            Default None. Correction of the voxel volumes. Of None, no correction will be applied.
            - ('filling fraction', float f 0-1) e.g. for porous materials. The net volume of all voxels corresponds to f * Volume(unit cell from periodicity)
            - ('net volume', float V_net [m^3]) e.g. for individual particles. The net volume of all voxels corresponds to V_net

        ::Magnetic parameters::
        *moments*: None, array (3), or array (n_R, 3)
            Default None. Sample is then assumed to be non-magnetic.
            Magnetic moment for each position, (usually) normalised to 1.
            If a 3-vector is given only, a uniform magnetisation is assumed.
        *saturation_magnetisation* float or array (n_R) | A/m
            Default 1.
            Saturation magnetisation to calculate moment of each data point and its associated volume.
            If set to 1, it is assumed that input already correspond to the moments in multiples of mu_Bohr

        ::Neutron parameters::
        *scattering_length_density*: float or array with shape (n_R) | [1e-6/Angstrom^2]
            Default 3.
            Material-dependent neutron scattering length density. Can be uniform, or position dependent.
            Use e.g. periodictable.nsf.neutron_sld(*args, **kw) to get values (https://periodictable.readthedocs.io).
            Will be transformed into 1/m^2, for calculation purposes.

        :Returns:
        Sample object.
        """

        self.print_diagnostics = print_diagnostics
        self.sample_name = sample_name

        # structure: position coordinates, shift coordinates to centre of mass
        self.R_veclist = self._check_array_dimension(positions)
        self.R_veclist = self.R_veclist #- np.mean( self.R_veclist, axis=0)

        # number of mesh points, and mesh type
        self.number_of_points = len(self.R_veclist) # right order?
        self._regular_grid = self._get_mesh_type()

        # number of unit cells in the structure
        self.number_of_unit_cells = number_of_unit_cells

        # periodic repetition
        self.periodicity = np.array( periodicity )
        if periodicity is None:
            self.periodicity = np.array( [0., 0., 0.] )

        # bounding box of structure, for control
        self.bounding_box = self.periodicity
        # if len( np.shape(self.R_veclist) ) == 1:
        #     self.bounding_box = np.array([0, 0, 0])
        # else:
        #     self.bounding_box = np.array( [
        #         np.max( self.R_veclist[:, i] ) - np.min( self.R_veclist[:, i] ) for i in range( 3 )
        #         ] )
        # if self.print_diagnostics is True:
        #     print(f'Data bounding box size: ({self.bounding_box[0]*1e9:.1f}, {self.bounding_box[1]*1e9:.1f}, {self.bounding_box[2]*1e9:.1f}) nm.')

        # set or calculate volume of each data point
        # regular grid: calculate scalar cuboid volume
        # irregular grid: list of volumes for each position
        if voxel_volumes is None: # no voxel volumes given
            if self.number_of_points < 3:
                print( 'ERROR: Not enough positions to calculate point volumes.' )
                print( '\tUsed volume of sphere with radius of 2 nm instead.' )
                self.R_volumes = 4. / 3. * np.pi * 2e-9 ** 3
            else:
                self.R_volumes = self._calculate_point_volumes()
            # 2021-05-29: why differnt length of magnetisation plots? differnt volume?
            # volumes should not vary that much...
            #self.R_volumes = np.mean(self.R_volumes)
        else: # voxel volumes are provided
            if np.prod(np.shape(voxel_volumes)) != self.number_of_points:
                if self.print_diagnostics is True:
                    print('REMARK to input data: The length of point volumes does not correspond to number of positions.')
                self.R_volumes = np.mean( voxel_volumes )
            else:
                self.R_volumes = voxel_volumes

        # get correction factor for voxel volumes
        if volume_correction is None:
            volume_correction_factor = 1.
        else:
            if volume_correction[0] == 'filling factor':
                if np.prod(self.periodicity) == 0:
                    volume_correction_factor = 1.
                    print('WARNING: Volume correction with filling factor requires that the system has a well-defined periodicity.')
                else:
                    volume_correction_factor = (volume_correction[1] * np.prod(self.periodicity)) / self._calc_net_volume()
            elif volume_correction[0] == 'net volume':
                volume_correction_factor = volume_correction[1] / self._calc_net_volume()
            else:
                print('WARNING: Volume correction must be given in the form ("filling factor"|"net volume", float).')
                volume_correction_factor = 1.

        # correct voxel volumes
        if volume_correction_factor == 1.:
            if self.print_diagnostics is True:
                print( 'REMARK: Voxel volumes were not corrected.' )
        else:
            self.R_volumes = volume_correction_factor * self.R_volumes

        # scattering lengths for each position
        if isinstance(scattering_length_density, float) or isinstance(scattering_length_density, complex) or isinstance(scattering_length_density, int):
            self.scattering_length_density = scattering_length_density
        else:
            if np.prod(np.shape(scattering_length_density)) == self.number_of_points:
                self.scattering_length_density = np.array( scattering_length_density )
            else:
                print('REMARK to input data: Number of scattering length density values does not correspond to the number of positions')
                print('\tInstead use mean scattering length density for all points!')
                self.scattering_length_density = np.mean( scattering_length_density )

        # position-dependent magnetic moments
        # 2021-05-28: update definition of M_veclist
        self.M_veclist = np.zeros_like(self.R_veclist)
        if (moments is None) or (np.mean( np.linalg.norm( moments, axis=0 ) ) == 0):
            self.is_magnetic = False,
            if self.print_diagnostics is True:
                print('REMARK: Sample is non-magnetic.')
        else:
            self.is_magnetic = True
            self.M_veclist = self._check_array_dimension(moments)
            if (np.shape(self.M_veclist)[0] != self.number_of_points) and (self.number_of_points > 1):
                print('REMARK to input data: The length of moment values does not correspond to the number of positions.')
                print('\tInstead use (uniform) mean magnetisation!')
                self.M_veclist = np.transpose(
                    np.tile( np.mean( self.M_veclist, axis=0 ), self.number_of_points).reshape((self.number_of_points,3))
                    )
                #self.M_veclist = np.array([ np.mean([self.M_veclist], axis=0) for _ in range(self.number_of_points) ])
        # mean magnetisation vector (e.g. for subtraction of a reference scattering pattern)
        self.mean_magnetisation = np.mean( self.M_veclist, axis=0 )

        # saturation magnetisation - can be position dependent
        self.saturation_magnetisation = saturation_magnetisation
        if not isinstance(self.saturation_magnetisation, float) or isinstance(self.saturation_magnetisation, int):
            if np.prod(np.shape(self.saturation_magnetisation)) == self.number_of_points:
                self.saturation_magnetisation = np.array(self.saturation_magnetisation)
            else:
                print('ERROR: Number of values for M_sat do not correspond to the number of positions.')
                print('\tUse mean value instead')
                self.saturation_magnetisation = np.mean(self.saturation_magnetisation)

        # calculate position-dependent magnetisation, first in Am^2, and then transform to muliples of mu_B
        # multiplication with b_H gives the magnetic scattering length
        # (moments from micromagnetic simulations usually are normalised to length 1)
        self.M_veclist = np.transpose( np.multiply(
                np.transpose( self.M_veclist )
                , np.multiply(
                    np.divide( self.saturation_magnetisation, scipy.constants.value( 'Bohr magneton' )), self.R_volumes
                    )
                ) )

        # print some sample statistics
        statistics  = f'{self.number_of_points} positions '
        statistics += f'with an average sphere diameter of {2. * np.power(3./4./np.pi * self._calc_net_volume() / self.number_of_points, 1./3.)*1e9:.2f} nm'
        if self.is_magnetic is True:
            statistics += f', and an average moment of {np.mean(np.linalg.norm(self.M_veclist, axis=0)):.1e} mu_Bohr.'
        else:
            statistics += '.'
        if self.print_diagnostics is True:
            print(statistics)

        return


    def _get_mesh_type(self):
        """ 
        Determine type of mesh from the list of coordinates.
        
        :Returns:
        *regular_grid* [Bool] Indicates if grid is regular.
            True: Grid is regular, e.g. representing input from finite-differences simulations.
            False: Mesh is isregular, e.g. representing input from finite-element simulations.
        """

        regular_grid = True

        if self.number_of_points > 1:
            unique_diff, unique_counts = np.unique( np.abs( np.diff( self.R_veclist, axis=0 ) ).ravel(), return_counts=True )
            # assuming we have a regular mesh with positions in a regular (grid-like) order
            # then the count frequency of the most common distance should be more than a third of the net number of entries
            regular_grid = np.sort( unique_counts )[::-1][0] > np.prod( 3 * self.number_of_points ) / 4

        return regular_grid


    @staticmethod
    def _check_array_dimension(input):
        """ Checks that dimension of vector arrays is (n, 3), and transposes input array, if neccessary """

        input = np.array(input)
        output = input

        if np.shape(input)[-1] != 3:
            # transpose, if first dimension is correct
            if np.shape(input)[0] == 3:
                output = np.transpose(input)
            else:
                print('ERROR: Array does not have shape (n, 3) or (3, n) and thus is not a list of vectors!')

        # make sure that all entries are numpy arrays
        output = np.array([np.array(out) for out in output])

        return output


    def _calculate_point_volumes(self):
        """
        Calculate the volume of each data point and the net volume of the structure.
        
        :Returns:
        Scalar volume for regular grid, and a list of point volumes for irregular list.
        """

        if self._regular_grid == True:
            # scalar volume of cuboid cell: product of smallest non-zero distances in each direction
            R_volume = np.prod([np.unique(np.abs(np.diff(self.R_veclist[:,i])))[1] for i in range(3)])
            if self.print_diagnostics is True:
                print(f"Structural data is on regular grid, with a cell volume of {R_volume:.2e} m^3.")
        else:
            # get pairwise distances between each data point
            R_distmatrix = np.linalg.norm(np.repeat(self.R_veclist[:, np.newaxis], np.shape(self.R_veclist)[0], 1) - self.R_veclist, axis = -1)
            # find non-zero nearest-neighbour distance for each point
            R_distmatrix = np.reshape( R_distmatrix[R_distmatrix > 0], (np.shape(R_distmatrix)[0], np.shape(R_distmatrix)[1] - 1) )
            # sphere volume will (probably?) underestimate total volume - check and correct output!
            R_volume = 4. / 3. * np.pi * np.power( np.min(R_distmatrix, axis = -1), 3. )
            print(f"Structural data is an irregular-spaced mesh, with position-dependent point volumes (on average {np.mean(R_volume):.2e} m^3).")

        # 2021-05-28: in 3d plots magnetisation vectors have differnt length => volumes are not equal...
        return np.mean(R_volume)


    def _calc_net_volume(self):
        """ Calculates net volume of the structure. """

        if self._regular_grid == True:
            net_volume = self.R_volumes * len( self.R_veclist )
        else:
            net_volume = np.sum( self.R_volumes )

        return net_volume


    def plot_scattering_length(self, ax=None, plane='xz', step_size=2e-9, r_unit='nm', title='', show_magnetic=False):
        """
        Plot sample density projected onto one of the principal planes.

        *Parameters*
        *ax* matplotlib axis to plot upon.
            Default None. If None, a new figure will be created.
        *plane* string with two letters.
            Default 'xz' (parallel to detector plane, if no rotations are applied).
            Plane on which sample density is projected to.
            When used in the Experiment class, the plane 'VW' is given with respect to the beamline coordinate system.
        *step_size* float, m
            Default 2 nm. Spatial resolution of the mesh.
        *length_scale* Reference length scale for axis scaling.
            Default 'nm'. Other options are 'm', 'AA'/'Angstrom', 'nanometres' etc (see q_unit...)
        *title* string
            Optional axis title (default is empty).
        *show_magnetic* Boolean
            Default False. Whether magnetisation data also should be shown.

        *Returns*
        matplotlib figure or axis
        """

        # check if matplotlib axis is given, if not create a new figure
        sns.set_style( 'white' )
        return_value = ax
        plot_cbar = False
        if ax is None:
            plot_cbar = True
            fig = plt.figure(figsize = (5,5))
            grid = AxesGrid( fig, 111,
                             nrows_ncols=(1,1),
                             axes_pad=0.45,
                             cbar_mode='single',
                             cbar_location='right',
                             cbar_pad=0.25
                             )
            ax = grid.axes_row[0][0]
            ax.set_aspect('equal')
            plot_cbar = True
            return_value = fig
        ax.set_title(title)

        # get columns and set axis labels
        col_index_x, col_index_y = self._get_coordinate_index(plane[0]), self._get_coordinate_index(plane[1])
        r_unit_label, r_unit_scaling_factor = self._get_r_axislabel_settings(r_unit)
        ax.set_xlabel(f'$r_{plane[0]}$ ({r_unit_label})')
        ax.set_ylabel(f'$r_{plane[1]}$ ({r_unit_label})')

        # axis and image limits -- if periodicity is zero set to the respective data range
        # TODO: choose better data ranges?
        plot_min, plot_max = -0.5 * self.periodicity, +0.5 * self.periodicity
        if np.prod(self.periodicity) == 0:
            plot_min, plot_max = -0.5 * self.bounding_box, +0.5 * self.bounding_box

        # coordinate mesh for density image
        x_steps = np.arange( plot_min[col_index_x], plot_max[col_index_x], step_size )
        y_steps = np.arange( plot_min[col_index_y], plot_max[col_index_y], step_size )
        mesh_xy = np.transpose( np.meshgrid( x_steps, y_steps, indexing='ij' ) )

        # calculate projected density as an image
        # TODO: vectorise how the images are added?
        projected_density = []
        for i, curr_pos in enumerate( self.R_veclist ):
            # distance to current position
            mesh_dist = np.linalg.norm( mesh_xy - np.array( [curr_pos[col_index_x], curr_pos[col_index_y]] ), axis=-1 )
            # radius
            if isinstance( self.R_volumes, float):
                radius = np.power( 3. / 4. / np.pi * self.R_volumes, 1./3. )
            else:
                radius = np.power( 3. / 4. / np.pi * self.R_volumes[i], 1./3.)
            # scattering length
            if isinstance(self.scattering_length_density, float) or isinstance(self.scattering_length_density, complex):
                sld = self.scattering_length_density
            else:
                sld = self.scattering_length_density[i]
            # density image for current position
            projected_density.append(
                    2. * sld
                    * np.sqrt(
                        (radius ** 2 - np.power( mesh_dist, 2. ))
                        * np.heaviside( radius - mesh_dist, 0. )
                    )
                )
        projected_density_sum = np.sum( projected_density, axis=0 )

        # set axes limits
        color_lim = np.max( np.abs( projected_density_sum ) )
        plot_min, plot_max = r_unit_scaling_factor * plot_min, r_unit_scaling_factor * plot_max
        # set axes limits
        ax.set_xlim(plot_min[col_index_x], plot_max[col_index_x])
        ax.set_ylim(plot_min[col_index_y], plot_max[col_index_y])

        # plot figure
        extent = np.array([plot_max[col_index_x], plot_min[col_index_x], plot_min[col_index_y], plot_max[col_index_y]])#+step_size/2.
        #extent = [mesh_xy[col_index_x][0], mesh_xy[col_index_x][-1], mesh_xy[col_index_y][0], mesh_xy[col_index_y][-1]]
        cbar_ref = ax.imshow(
              np.real( projected_density_sum )  # scattering length might be complex!
            , cmap=cmocean.cm.balance
            , vmin = -color_lim, vmax = +color_lim
            , extent= extent
            , interpolation='bilinear'
            , origin='lower'
            , zorder = 1
            )
        if (self.is_magnetic is True) and (show_magnetic is True):
            ax.quiver(
                  r_unit_scaling_factor * self.R_veclist[:, col_index_x]
                , r_unit_scaling_factor * self.R_veclist[:, col_index_y]
                , np.transpose(self.M_veclist)[col_index_x] / self.saturation_magnetisation / np.mean(self.R_volumes)
                , np.transpose(self.M_veclist)[col_index_y] / self.saturation_magnetisation / np.mean(self.R_volumes)
                , color='white'
                , pivot = 'middle'
                , linewidth=1.5
                #, scale = 10
                , zorder = 5
                )

        # add colorbar
        if plot_cbar is True:
            cax = grid.cbar_axes[0]
            plt.colorbar( cbar_ref, cax=cax )
            cax.set_ylabel( '$b_N$ m/f.u.)', rotation=90 )

        return


    @staticmethod
    def _get_r_axislabel_settings(r_unit):
        """
        Return r axis label and conversion factor.
        Copy-paste from function Probe._get_q_axislabel_settings()

        :Parameters:
        *r_unit*: string
            Specifies unit of q values (unit of q is 1/length)
            - 'nm', 'nanometre', 'nanometer'
            - 'AA', 'angstrom', 'Angstrom'
            - 'm' or None

        :Returns:
        *r_unit_label*: string
            Label for plot q axis.
        *r_unit_scaling_factor*: float
            Conversion factor, q values will be divided to get value according to q unit.
        """

        r_unit_label = 'm'
        r_unit_scaling_factor = 1.

        if (r_unit == 'nm') or (r_unit == 'nanometre') or (r_unit == 'nanometer'):
            r_unit_label = 'nm'
            r_unit_scaling_factor = 1e9
        elif (r_unit == 'AA') or (r_unit == 'angstrom') or (r_unit == 'Angstrom'):
            r_unit_label = 'Å'
            r_unit_scaling_factor = 1e10
        elif (r_unit is None) or (r_unit == 'm'):
            pass

        return r_unit_label, r_unit_scaling_factor

    @staticmethod
    def _get_coordinate_index(index_string):
        """ Returns coordinate index for plotting, depending on string value. """

        if (index_string == 'x') or (index_string == 'U'):
            index = 0
        elif (index_string == 'y') or (index_string == 'V'):
            index = 1
        else: # z, W
            index = 2

        return index



