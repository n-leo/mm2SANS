import numpy as np

# to get projections for neutron polarisation, >scipy 1.2 required
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt # data plotting
from matplotlib import gridspec # use this instead of indices!
import seaborn as sns
sns.set_style('whitegrid'); sns.set_context('talk')

from mm2SANS.detector import Detector
from mm2SANS.beamline import Beamline

class Probe:
    """ Setup neutron probe. """

    def __init__(
        self
        # detector properties
        , sans_instrument='PSI_SANS1'
        # beamline properties
        , neutron_wavelength=6e-10, detector_distance=15.
        , neutron_polarisation=None, magnetic_field=0., flipping_ratio=1.
        , sample_rotations=[], sample_environment_rotations=[]
        , detector_offset_V=0., detector_offset_W=0, angle_unit='deg'
        # q-map properties
        , qmap_disorder=0.35, q_unit='nm'
        ):
        """
        Initialise (polarised) neutron Probe object from Detector and Beamline class.

        :Parameters:
        
        ::for Detector::
        *sans_instrument* | string or None
            SANS instrument, to initilise Detector object
        
        :for Beamline::
        *neutron_wavelength*
            neutron wavbelength
        *detector_distance*
            detector_distance
        *neutron_polarisation*
            neutron polarisation
        *optional parameters* described in Beamline class
            detector_offset_V, detector_offset_W, angle_unit
        TODO: add parameters from beamline... => see beamline code...
        # TODO: add sample and sample environment rotations!
            
        ::additional parameters for Probe::
        *qmap_disorder* | float between 0 and 0.5
            default value = 0.35
        *q_unit* | string
            Specifies unit of q values (unit of q is 1/length)
            - 'nm', 'nanometre', 'nanometer'
            - 'AA', 'angstrom', 'Angstrom'
            - 'm' or None
        
        :Returns:
        Obejct with Detector, Beamline sub-objects, and q-map used for simulation of SANS patterns.
        """
        
        # detector settings (usually not changed for the beamlines)
        self.Detector = Detector(sans_instrument)
        
        # setup of beamline
        self.Beamline = Beamline(
            neutron_wavelength=neutron_wavelength, detector_distance=detector_distance
            , neutron_polarisation=neutron_polarisation, magnetic_field=magnetic_field, flipping_ratio=flipping_ratio
            , sample_rotations=sample_rotations, sample_environment_rotations=sample_environment_rotations
            , detector_offset_V=detector_offset_V, detector_offset_W=detector_offset_W, angle_unit=angle_unit
            )
        self._q_unit = q_unit

        # calculate qmap
        self.qmap_disorder = qmap_disorder
        # calculate and set Q_veclist
        self.Q_veclist = self.calc_qmap(qmap_disorder=self.qmap_disorder)

        # neutron polarisation in beamline reference frame (U,V,W)
        #self.neutron_polarisation_UVW = Rotation.from_matrix(self.Beamline._rotation_uvw_UVW).apply(self.Beamline.neutron_polarisation)
        self.neutron_polarisation_UVW = np.matmul( self.Beamline.neutron_polarisation, self.Beamline._rotation_uvw_UVW )
        
        return
        
    
    def calc_qmap(self, qmap_disorder=0.):
        """ 
        Calculate the q-map from detector and beamline settings. 
        
        :Parameters:
        *qmap_disorder* | float between 0 and 0.5 | default = 0
            Disorder parameter to avoid Fourier transform artifacts.
            0.35 is a good choice for moderate disorder.
            
        :Returns:
        *q_list* | (n_Q, 3) | [m^-1, m^-1, m^-1]
        TODO: check format of Q-list...
            List of q vectors for momentum transfer on SANS detector.
        """
        
        #  positions of the pixel centers (assuming detector is centered)
        pixel_center_list_U = np.array([0.])  # in principle, U == detector_distance, but set U == 0 here
        pixel_center_list_V = self.Detector.pixel_size_V * np.arange( -self.Detector.pixel_number_V/2.+0.5, +self.Detector.pixel_number_V/2.+0.5, 1)
        pixel_center_list_W = self.Detector.pixel_size_W * np.arange( -self.Detector.pixel_number_W/2.+0.5, +self.Detector.pixel_number_W/2.+0.5, 1)

        #  if detector has a transverse offset => move values
        pixel_center_list_V = pixel_center_list_V + self.Beamline.detector_offset_V
        pixel_center_list_W = pixel_center_list_W + self.Beamline.detector_offset_W

        # create 2D map of bin centers - order of coordinates is U, V, W
        # old: V, W, U
        pixel_centers = np.meshgrid(pixel_center_list_U, pixel_center_list_V, pixel_center_list_W, indexing = 'ij')

        # calculate which scattering angle the bin centers correspond to...
        # in small-angle approximation the angle distance between the bins should be constant: tan(phi) ~ phi
        q_angle_tangens = np.divide( pixel_centers, self.Beamline.detector_distance_U )
        q_map = ( 2. * np.pi / self.Beamline.neutron_wavelength ) * q_angle_tangens

        # if neccessary, add a small disorder to each pixel
        if qmap_disorder != 0.:
            
            random_offset = np.random.random( np.shape(q_map) )
            random_offset[0, :, :, :] = 0.  # no offset along the beam direction, only in the detector plane
            pixel_prefactor = np.mean([self.Detector.pixel_size_V, self.Detector.pixel_size_W]) / self.Beamline.detector_distance_U
            q_prefactor = 2. * np.pi / self.Beamline.neutron_wavelength
            q_map = q_map + pixel_prefactor * q_prefactor * np.multiply( qmap_disorder, random_offset)

        # transform to a list of 3-vectors (plotting can be easily done with scatter or tricontourf)
        q_list = np.transpose(q_map.reshape((3, np.product(np.shape(q_map)) // 3)))

        # order of elements U, V, W ( U is the "longitudinal z" direction along the beam)
        # q_list[:,-1] = 0 # set U values to zero, but probably not neccessary...

        return q_list


    @staticmethod
    def calc_log_qmap(q_max=1e4, num_q_points=32):
        """
        Calculates a detector map with logarithmic spacing.

        :Parameters:
        *q_max* float | m^-1
            Default 1/(0.1 mm). Maximum q value to consider.
        *num_q_points* int (odd)
            Default 32. Number of log bins to consider in each direction

        :Returns:
        *q_list* ((2*num_q_points+1)^2, 3)
            vectors of q-map with log spacing between points

        TODO: maybe need to update plotting functions in the Experiment class to get non-linear colourmap scaling (maybe add a tag to the Probe class to indicate that a logarithmic qmap is used...)
        """
        # if, and if yes, how to implement disorder?

        q_values = np.hstack( (
            -1 * np.logspace( 0, np.log10( q_max ), num_q_points)[::-1]
            , np.array( [1e-24] ) # do not put a zero in there, it only will make problems...
            , +1 * np.logspace( 0, np.log10( q_max ), num_q_points)
        ) )

        q_list = np.transpose(
            np.reshape(
                np.meshgrid( [0], q_values, q_values )
                , (3, len( q_values ) ** 2)
                )
            )

        return q_list

        
    def plot_qmap(self, figsize = (12,6), q_unit='nm'):
        """
        Plot detector map.
        
        :Parameters:
        *figsize* | (size_x, size_y) | (inch, inch)
            Figure size, default (12, 6)
        *q_unit* | 'nm' or 'Angstrom' 
            Unit for the q, given either in inverse nm or inverse Angstrom.
            Default is 'nm'
        
        :Returns:
            Figure of detector map of scattering angles and evaluated q values.
        """
            
        q_unit_label, q_unit_scaling_factor = self._get_q_axislabel_settings()
        
        sns.set_style('whitegrid') # white, whitegrid
        sns.set_context('talk') # paper, notebook, talk, poster

        fig = plt.figure(figsize = figsize)
        ax0 = plt.subplot(1, 2, 1, aspect = 'equal')
        ax1 = plt.subplot(1, 2, 2, aspect = 'equal')

        ax0.set_title(f'pixel centers at {self.Beamline.detector_distance_U:.1f} m detector distance')
        ax1.set_title(f'evaluated $q$ points (disorder = {self.qmap_disorder:.2f})' )
        
        #plot scattering angles, without disorder
        q_map_nodisorder = self.calc_qmap(qmap_disorder=0)
        q_angle = np.arctan( q_map_nodisorder * self.Beamline.neutron_wavelength / (2 * np.pi )  )
        ax0.scatter(
             np.rad2deg( q_angle[:,1] )
            ,np.rad2deg( q_angle[:,2] )
            ,s = 0.1
            )
        ax0.set_xlabel('$\phi_V$ (deg)')
        ax0.set_ylabel('$\phi_W$ (deg)')

        # plot evaluated q positions (probably with disorder)
        ax1.scatter(
             self.Q_veclist[:,1] / q_unit_scaling_factor
            ,self.Q_veclist[:,2] / q_unit_scaling_factor
            ,s = 0.1
            )
        ax1.set_xlabel(f'$q_V$ ({q_unit_label})')
        ax1.set_ylabel(f'$q_W$ ({q_unit_label})')

        plt.tight_layout()

        return


    def _get_q_axislabel_settings(self):
        """ 
        Return q axis label and conversion factor.
        
        :Parameters:
        *q_unit*: string 
            Specifies unit of q values (unit of q is 1/length)
            - 'nm', 'nanometre', 'nanometer'
            - 'AA', 'angstrom', 'Angstrom'
            - 'm' or None
        
        :Returns:
        *q_unit_label*: string
            Label for plot q axis.
        *q_unit_scaling_factor*: float
            Conversion factor, q values will be divided to get value according to q unit.
        """

        q_unit_label = 'm$^{-1}$'
        q_unit_scaling_factor = 1.

        if (self._q_unit == 'nm') or (self._q_unit == 'nanometre') or (self._q_unit == 'nanometer'):
            q_unit_label = 'nm$^{-1}$'
            q_unit_scaling_factor = 1e9
        elif (self._q_unit == 'AA') or (self._q_unit == 'angstrom') or (self._q_unit == 'Angstrom'):
            q_unit_label = 'Å$^{-1}$'
            q_unit_scaling_factor = 1e10
        elif (self._q_unit == None) or (self._q_unit == 'm'):
            pass
            
        return q_unit_label, q_unit_scaling_factor