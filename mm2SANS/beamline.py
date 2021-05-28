import numpy as np

# for sample rotations
from scipy.spatial.transform import Rotation

class Beamline:
    """ 
    Beamline settings.
    """
    
    #import numpy as np
    #from scipy.spatial.transform import Rotation 
    # to get projections for neutron polarisation, >scipy 1.2 required
    
    
    def __init__(self
                 , neutron_wavelength=6e-10, detector_distance=15.
                 , neutron_polarisation=None, magnetic_field = 0.
                 , sample_rotations=[], sample_environment_rotations=[]
                 , flipping_ratio=1., detector_offset_V=0., detector_offset_W=0
                 , angle_unit = 'deg'
                ):
        """
        Setup of neutron and beamline settings. 
        
        : Parameters :
        *neutron_wavelength*: [float] | [m]
            Default 6 Angstrom.
            Neutron wavelength.
            TODO: what about polychromatic beamlines? What about wavelength spread?
        *detector_distance*: [float] | [m]
            Default 15 m.
            Distance between detector and sample (which is sitting at U=0, i.e. distance is measured along axis U).
        *neutron_polarisation*: [float, float, float] or None | [1]
            Default None
            neutron polarisation, i.e. direction of the guide field at the sample within the sample environment
        *magnetic_field* float | [T]
            Default 0. Magnitude of the magnetic field at the sample position.
            The field sets the neutron polarisation direction, but is usually too weak to affect the sample.
            TODO: For strong fields: implement Zeeman splitting???
        *sample_rotations*: list of zero to three pairs [rotation_axis, roation_angle]
            Default [].
            Rotations to orient sample coordinates (x, y, z) within the sample environment (u, v, w).
            Rotation axis is specified by rotation_type 'yaw', 'pitch', and 'roll'.
        *sample_environment_rotations*: list of zero to three pairs [rotation_axis, roation_angle]
            Default [].
            Rotations to orient sample environment (u, v, w) within the beamline coordinate system (U, V, W).
            Rotation axis is specified by rotation_type 'yaw', 'pitch', and 'roll'.
        *flipping_ratio* float | [1]
            Default 1, otherwise number between 0 and 1.
            Gives the flipping ratio of neutron polarisation.
        *detector_offset_V, detector_offset_W* float | [m]
            Default 0. Offset of detector center along beamline horizontal (V) and vertical (W) direction to extend q range.
            
        : Optional/pre-set parameters :
        *angle_unit*: 'deg' or 'rad'
            Default 'deg'.
            Unit for the sample rotations.

        : Returns :
        Beamline beamline object which stores experimental settings.
        The rotation matrices to transform the coordinate systems are calculated upon initialisation. 
        If angles are re-set, these need to be re-calcualted.
        
        TODO (maybe): 
            - Implement non-perfect flipping ratio?
        """
    
        # setup of beamline
        self.neutron_wavelength = neutron_wavelength
        self.detector_distance_U = detector_distance
        self.detector_offset_V = detector_offset_V
        self.detector_offset_W = detector_offset_W
        self.flipping_ratio = flipping_ratio
        
        # guide field (= neutron polarisation) within sample environment
        self.magnetic_field = magnetic_field
        if (neutron_polarisation is None) or (np.sum(np.abs(neutron_polarisation)) == 0):
            # zero vector
            self.neutron_polarisation = np.zeros(3)
        else:
            # vector with length normalised to one
            self.neutron_polarisation = np.array(neutron_polarisation) / np.linalg.norm(neutron_polarisation)

        # angles to calculate rotation matrices are usually given in degrees, but can be set to radians with 'rad'
        self._angle_unit = angle_unit

        # initialise rotation matrices between the different coordinate systems
        self._rotation_xyz_uvw = np.diag((1,1,1))
        self._rotation_uvw_UVW = np.diag((1,1,1))
        self._rotation_xyz_UVW = np.diag((1,1,1))
        
        # orientation sample coordinates (x, y, z) in sample environment (u, v, w)
        self.sample_rotations = sample_rotations
        # orientation sample environment (u, v, w) in beamline (U, V, W)
        self.sample_environment_rotations = sample_environment_rotations
        
        # calculate the respective rotation matrices
        self.calc_rotation_matrices()
        
        return
    
    
    def calc_rotation_matrices(self):
        """ Calculate roation matrices between sample, sample environment, and beamline """

        # 2021-05-25 re-set rotation matrices (otherwise each re-run will lead to differnt results!)
        _rotation_xyz_uvw = Rotation.from_matrix( np.diag((1,1,1)) )
        _rotation_uvw_UVW = Rotation.from_matrix( np.diag((1,1,1)) )
        _rotation_xyz_UVW = Rotation.from_matrix( np.diag((1,1,1)) )

        # orientation of sample in sample environment
        for rotation in self.sample_rotations:            
            rotation_type, rotation_angle = self._test_rotation_input(rotation)
            _rotation_xyz_uvw = self._get_rotation_matrix(rotation_type, rotation_angle, self._angle_unit) * _rotation_xyz_uvw

        # orientation of sample environment in beamline
        for rotation in self.sample_environment_rotations:
            rotation_type, rotation_angle = self._test_rotation_input(rotation)
            _rotation_uvw_UVW = self._get_rotation_matrix( rotation_type, rotation_angle,
                                                                self._angle_unit ) * _rotation_uvw_UVW

        # oriantation of sample in beamline
        _rotation_xyz_UVW = _rotation_uvw_UVW * _rotation_xyz_uvw

        # save to class, and set all rotations as rotation matrices
        self._rotation_xyz_uvw = _rotation_xyz_uvw.as_matrix()
        self._rotation_uvw_UVW = _rotation_uvw_UVW.as_matrix()
        self._rotation_xyz_UVW = _rotation_xyz_UVW.as_matrix()

        # print( 'xyz => uvw', self._rotation_xyz_uvw, np.linalg.det( self._rotation_xyz_uvw ) )
        # print( 'uvw => UVW', self._rotation_uvw_UVW, np.linalg.det( self._rotation_uvw_UVW ) )
        # print( 'xyz => UVW',  self._rotation_xyz_UVW, np.linalg.det(self._rotation_xyz_UVW) )
        #  print('Rotation matrices between coordinate systems (x,y,z), (u,v,w) and (U,V,W) calculated.')

        return


    @staticmethod
    def _test_rotation_input(rotation):
        """ Helper function: Do a quick test if rotation input is correctly given. Swa values, if neccessary. """
        a, b = rotation  # one should be string, the other a float val
        test_1 = (isinstance(a, str) and isinstance(b, str))
        test_2 = (not(isinstance(a, str)) and not(isinstance(b, str)) )
        if test_1 or test_2:
            print(f'Input {rotation} incorrect: Required format is (str(rotation_type), float(rotation_angle))')
            print('\tNo rotation applied to data.')
            rotation_type, rotation_angle = 'roll', 0.
        elif isinstance(a, str):
            rotation_type, rotation_angle = a, b
        else:
            rotation_type, rotation_angle = b, a

        return rotation_type, rotation_angle


    @staticmethod
    def _get_rotation_matrix(rotation_type, rotation_angle, angle_type):
        """
        Helper function:
        Returns rotation matrix for rotation_angle (either in angle_type = 'deg' or 'rad')
        around different axes according to rotation_type = 'yaw', 'pitch', 'roll'.

        Will output a scipy.spatial.Rotation object (instead of a matrix...)
        """

        if angle_type == 'deg':
            angle = np.deg2rad(rotation_angle)
        else:
            angle = rotation_angle

        # rotation matrix for pitch, yaw, roll
        if rotation_type == 'yaw':  # rotation around vertical z axis (W axis)
            rotation_matrix = Rotation.from_rotvec( rotation_angle * np.array([0, 0, 1]))
        elif rotation_type == 'pitch': # rotation around horizontal y axis (V axis)
            rotation_matrix = Rotation.from_rotvec( rotation_angle * np.array( [0, 1, 0] ) )
        elif rotation_type == 'roll':  # rotation around x axis (or U along beam)
            rotation_matrix = Rotation.from_rotvec( rotation_angle * np.array( [1, 0, 0] ) )
        else:
            rotation_matrix = Rotation.from_matrix( np.diag([1,1,1]) )
            print('Rotation type neither yaw, pitch, or roll - unity matrix returned!')

        return rotation_matrix


    def print_beamline_settings(self):
        """ Print short summary of beamline properties. """

        # wavelength and detector distance
        info_string = f'Neutron wavelength = {np.round(self.neutron_wavelength*1e10, 1):.1f} Angstrom, detector distance = {self.detector_distance_U} m'
        if np.sum(np.abs([self.detector_offset_V, self.detector_offset_W])) > 0:
            info_string +=f'\nDetector offset along V = {self.detector_offset_V} m, along W = {self.detector_offset_W} m'

        # neutron polarisation
        if (np.sum(np.abs(self.neutron_polarisation)) > 0):
            info_string += f'\nNeutron polarisation set to {self.neutron_polarisation} in sample environment coordinate system (u, v, w), '
            #info_string += f'\nNeutron polarisation is {np.round(np.matmul(self.rotation_uvw_UVW, self.neutron_polarisation), 3)} in the beamline coordinate system (U, V, W).'
        else:
            info_string += '\nNo neutron polarisation set.'

        return print(info_string)


    def print_orientation_overview(self):
        """ Print short summary of rotation matrices. """

        info_string = ''

        if len(self.sample_rotations)==0 and len(self.sample_environment_rotations)==0:
            info_string += f'All coordinate systems (i.e. sample, sample environment, and beamline) are co-aligned.'
        else:
            if len(self.sample_rotations)>0:
                info_string += f'Rotations between sample coordinates (x, y, z) and sample environment (u, v, w): {self.sample_rotations}\n'
            if len(self.sample_environment_rotations)>0:
                info_string += f'Rotations between sample environment (u, v, w) and beamline (U, V, W): {self.sample_environment_rotations}\n'

            info_string += f'Transformation matrices:\n'
            str_mat_xyz_uvw = np.array2string(np.around(self._rotation_xyz_uvw,decimals=3)).replace('\n', '')
            str_mat_uvw_UVW = np.array2string(np.around(self._rotation_uvw_UVW,decimals=3)).replace('\n', '')
            str_mat_xyz_UVW = np.array2string(np.around(self._rotation_xyz_UVW,decimals=3)).replace('\n', '')
            info_string += f'xyz => uvw {str_mat_xyz_uvw}\nuvw => UVW {str_mat_uvw_UVW}\nxyz => UVW {str_mat_xyz_UVW}'

        return print(info_string)


