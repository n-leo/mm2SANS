from mm2SANS.sample import Sample
from mm2SANS.probe import Probe

import numpy as np
import pandas as pd

# for sample rotations
#from scipy.spatial.transform import Rotation

import scipy.constants
# TODO: sign for magnetic scattering length positive or negative?
b_H = (  # prefactor for magnetic scattering length *density* = -2.7 fm/mu_B = -2.906e8 (Am)^{-1}
        # magnetic scattering length = b_H * |M|, with moment M in multiples of mu_Bohr
        scipy.constants.value( 'neutron mag. mom. to nuclear magneton ratio' ) # [1]
        * scipy.constants.value( 'classical electron radius' ) # [m]
        / (2. * scipy.constants.value( 'Bohr magneton' )) # [J/T = Am^2]
    ) / 1e14 # to transform it to be the same as the structural SLD [10-6 / Angstrom^2]

import matplotlib.pyplot as plt # data plotting
from mpl_toolkits.axes_grid1 import AxesGrid

import seaborn as sns
sns.set_style('whitegrid') # white, whitegrid
sns.set_context('talk') # paper, notebook, talk, poster

# nicer colourmaps: https://matplotlib.org/cmocean/
# but replace colourmaps if the package is not installed
# https://stackoverflow.com/questions/1051254/check-if-python-package-is-installed
import matplotlib.colors as colors
try:
    import cmocean
    cmap_int = cmocean.cm.ice_r
    colorbar_crop_percent = 12.5 # 7.5 before
    cmap_pm  = cmocean.tools.crop_by_percent(cmocean.cm.balance, colorbar_crop_percent, which='both', N=None)
except ImportError as e:
    print('cmocean is not installed - but it makes nice colourmaps :-)')
    cmap_int = plt.cm.plasma
    cmap_pm  = plt.cm.bwr
    pass


class Experiment:
    """
    Experiment class to calculate SANS scattering patterns.

    All internal calculations are done in SI units, i.e. SLDs are transformed from their usual 10-6/Angstrom^2 to 1/m^2.
    Output values are given in neutron-usual units, i.e.
        - [fm = 1e-15 m] for scattering lengths
        - [barn = 100 * fm^2 = 1e-28 m^2] for scattering cross-sections / intensities
    """

    def __init__(self
               # neccessary parameters
               , input_Sample, input_Probe
               # optional parameters
               , print_diagnostics=True
               ):
        """
        Initialise a SANS experiment/scattering pattern simulation.

        :Parameters:
        *Sample* mm2SANS Sample object
            Describing the density and magnetisation distribution of the scattering object.
        *Probe* mm2SANS Probe object
            Detailing information about the SANS detector and the beamline setup.
        *print_diagnostics*:
            Default False. Print status messages from calculations of scattering patterns.

        :Returns:
        Experiment object, including the calculated scattering pattern, and differnt plot functions.

        Colums of self.data, a structured pandas dataframe:
            - q_V, q_W: momentum transfer in detector plane; q_U should be zero
            - sld_struct, sld_mag_U, sld_mag_V, sld_mag_W: structural and magnetic scattering lengths
            - I_pp, I_mm , I_pm, I_mp: non-spin-flip and spin-flip scattering cross sections
            - I_p, I_m, I_sum, I_dif: cross sections for half-polarised SANS measurements
            - asym: spin asymmetry
        """

        self.print_diagnostics = print_diagnostics

        # keep probe object (includes all beamline settings, neutron polarisation, and the Q_veclist)
        self.Probe = input_Probe
        # plotting options for q axes
        self._q_unit_label, self._q_unit_scaling_factor = self.Probe._get_q_axislabel_settings()

        # transform the coordinates and moments into the beamline coordinate system (U,V,W)
        # TODO: rotations come out completely wrong!!! Is rotation matrix correctly defined? Is matmul correct here???
        R_veclist = np.matmul(input_Sample.R_veclist, self.Probe.Beamline._rotation_xyz_UVW)
        M_veclist = np.matmul(input_Sample.M_veclist, self.Probe.Beamline._rotation_xyz_UVW)
        periodicity = np.abs( np.matmul( input_Sample.periodicity, self.Probe.Beamline._rotation_xyz_UVW ) )
        #rotation_xyz_UVW = Rotation.from_matrix( self.Probe.Beamline._rotation_xyz_UVW )
        #R_veclist = rotation_xyz_UVW.apply(input_Sample.R_veclist)
        #M_veclist = rotation_xyz_UVW.apply(input_Sample.M_veclist)
        #periodicity = rotation_xyz_UVW.apply(input_Sample.periodicity)
        self.Sample = Sample(
              positions=R_veclist
            , moments=M_veclist
            , saturation_magnetisation=1. # input moments already multiples of mu_B
            , periodicity=periodicity
            , voxel_volumes=input_Sample.R_volumes
            , scattering_length_density=input_Sample.scattering_length_density
            , print_diagnostics=self.print_diagnostics
            )

        # create output data frame
        self.data = pd.DataFrame()

        # set q columns
        self.data['q_U'] = self.Probe.Q_veclist[:, 0] # longitudinal direction along beam, should be (almost) zero
        self.data['q_V'] = self.Probe.Q_veclist[:, 1] # detector map, horizontal direction
        self.data['q_W'] = self.Probe.Q_veclist[:, 2] # detector map, vertical direction
        
        # angle phi [deg]: 0 = right horizontal axis, counts counterclockwise (i.e. mathmatically correct notation)
        # TODO: check that above angle definition is correct, after interchange of U and W!!!
        self.data['q_phi'] = np.rad2deg( np.arctan2( self.data['q_V'], self.data['q_W'] ) ) + 180.
        self.data['q_abs'] = np.linalg.norm( self.Probe.Q_veclist, axis=-1 )

        # set data columns for scattering components
        column_names = [
            'sld_struct', 'sld_mag_U', 'sld_mag_V', 'sld_mag_W'
            , 'I_pp', 'I_pm', 'I_mp', 'I_mm', 'I_m', 'I_p'
            , 'I_sum', 'I_dif', 'asym'
            ]
        for col in column_names:
            self.data[col] = np.nan

        # calculate scattering patterns
        #self.calc_scattering_pattern( print_diagnostics=print_diagnostics )

        return


    def _calc_form_factor(self):
        """
        Calculate site form factors, multiplied with structural scattering length.
        Either considers point scatters (if R_volume=None) or spherical scatterers.

        Output:
        Returns site-dependent scattering length, [m], array with shape (n_Q, n_R).
        """

        # calculate the form factors for spherical cells, if point scatterer are not assumed

        # TODO: is uniform form factor justified for homogeneous material?
        # uniform form factor if the material is uniform
        # # i.e. from a finite-differnences simulation (regular_grid = True) of a bulk material (scalar SLD)
        # cond1 = (self.Sample._regular_grid is True)
        # cond2 = isinstance(self.Sample.scattering_length_density, float) or isinstance(self.Sample.scattering_length_density, complex)
        # if (cond1 is True) and (cond2 is True):
        #     form_factor = np.ones((len(self.Probe.Q_veclist), self.Sample.number_of_points))
        #     form_factor = np.mean(self.Sample.scattering_length_density) * np.mean(self.Sample.R_volumes) * self.Sample.number_of_points * form_factor
        if True is True:
            # sphere form factor, with q = |Q| and r = |r| calculate from spherical voxel volumes
            # form_factor_sphere = 3 * V * SLD * [ sin(qr) - (qr)*cos(qr) ] / (qr)^3, see 2005Foerster
            if isinstance(self.Sample.R_volumes, float):
                R_volumes = np.array( [ self.Sample.R_volumes for _ in range(self.Sample.number_of_points) ] )
            else:
                R_volumes = self.Sample.R_volumes
            # sphere radii
            R_radii = np.power( (3 * R_volumes) / (4. * np.pi), 1./3. )

            QR_array = np.outer( np.linalg.norm(self.Probe.Q_veclist, axis=-1), R_radii)
            form_factor = np.multiply(
                        np.subtract(
                            np.sin(QR_array), np.multiply(QR_array, np.cos(QR_array))
                            )
                        ,
                        np.multiply(
                             3. * self.Sample.scattering_length_density # * self.Sample.R_volumes
                             , np.reciprocal( np.power(QR_array, 3.) )
                             )
                        )

        # form factor is a scattering length [m]
        return form_factor


    def _calc_Fourier_argument(self):
        """
        Calculate the phase argument for the Fourier transforms.

        :Returns:
        Array (n_Q, n_R).
        """

        return np.exp(1j * np.linalg.multi_dot([self.Probe.Q_veclist, np.transpose(self.Sample.R_veclist)]))


    def _calc_FF_lattice(self, uc_repetitions=None):
        """ Calculate the Fourier factor associated with the periodic repetition of the lattice. """

        # get number of repetitions
        if uc_repetitions is None:
            repetition_uc = []  # no repetitions along U
            for i, direction in enumerate( ['U', 'V', 'W'] ):
                # only need in-plane repetitions
                Q_diff = np.mean( np.abs( np.diff( self.data[f'q_{direction}'] ) ) )
                if self.Sample.periodicity[i] > 0:
                    repetition = int( np.ceil( 1. / (Q_diff * self.Sample.periodicity[i] + 1) + 1 ) )
                else:
                    repetition = 10
                # the 10 is to prevent from repetitions to become too large...
                # TODO: there are Fourier fringes appearing, depending on the number of repetitions
                repetition_uc.append( min( repetition, 10 ) )
        else:
            repetition_uc = uc_repetitions
        if self.print_diagnostics is True:
            print(f'Reptitions of unit cell {self.Sample.periodicity} considered along U, V, W: {repetition_uc}.')

        # construct list of lattice points to evaluate
        lattice_points = np.reshape(
            np.transpose(
                np.meshgrid(
                    range( repetition_uc[0] )
                    , range( repetition_uc[1] )
                    , range( repetition_uc[2] )
                )
            ), (np.prod( repetition_uc ), 3)
        ) * self.Sample.periodicity
        lattice_points = lattice_points - np.mean( lattice_points, axis=0 )

        # calculate Fourier transform for each lattice point, and average
        periodicity_of_unitcell = np.mean(
            np.exp(
                1j * np.linalg.multi_dot(
                    [self.Probe.Q_veclist, np.transpose( lattice_points )]
                )
            ), axis=-1
            )

        return periodicity_of_unitcell


    def _calc_FF_struct(self):
        #R_veclist, Q_veclist, R_volume=None, structural_scattering_length=None):
        """
        Calculate strctural scattering length via Fourier transform of spatial coordinates.
        Takes into account cells with spherical form factor (or, in a simplified approach, point scatterers).
        Can also take into account position-dependent scattering lengths, for material mixtures.

        Input:
        R_veclist: positions in real space, array of shape (n_R, 3)
        Q_veclist: evaluated q vectors in reciprocal space, array of shape (n_Q, 3)
        R_volume: Cell volume for form factor, either None, scalar, or list (n_R)
        structural_scattering_length: neutron scattering length, either None (=1), scalar, or list (n_R)

        Output:
        Structual scattering length in reciprocal space, [m], list with shape (n_Q)
        If structural_scattering_length == None: Fourier transform of the density, [m^3], list with shape (n_Q)
        """

        # calculate spatial Fourier transform, shape (n_Q) after summation over real-space sites
        N_FF = np.mean( # mean or sum, as otherwise the density argument is destroyed?
                np.multiply(
                    self._calc_form_factor()
                    , self._calc_Fourier_argument()
                    # np.multiply(
                    #     self.Sample.R_volumes * self.Sample.scattering_length_density
                    #     , self._calc_Fourier_argument()
                    #     )
                    )
                , axis=1  # sum over R axis
                )

        return N_FF


    def _calc_Mperp(self):
        """
        Calculate (real-space) magnetisation components perpendicular to scattering vector.
        \vec{M}_perp(\vec{Q}, \vec{R}) = \hat{Q} (\hat{Q} \cdot \vec{M}(\vec{R}) ) - \vec{M}(\vec{R})
        \hat{Q} is the normalised vector of Q: \hat{Q} = \vec{Q} / |\vec{Q}|

        Input:
        M_veclist: position-dependent magnetisation \vec{M}(\vec{R}), [Am^2], list of shape (n_R, 3)
        Q_veclist: evaluated \vec{Q}, [m^{-1}], list of shape is (n_Q, 3)
        Vectors in both list must be given in the same reference frame, i.e. the beamline reference frame.

        Output:
        Perpendicular magnetisation, [in multiples of mu_B], array of shape (n_Q, n_R, 3).
        """

        # norm of Q-vectors, shape = ( len(Q_veclist), 3 )
        q_norm_list = np.transpose(
            np.true_divide(np.transpose(self.Probe.Q_veclist), np.linalg.norm(self.Probe.Q_veclist, axis=-1))
            )

        # $$\vec{M}_\perp( \vec{Q}, \vec{R} ) = \hat{Q} \left( \hat{Q} \cdot \vec{M}(\vec{R}) \right) - \vec{M}(\vec{R})$$
        M_perp_QR3 = np.subtract(
                np.multiply( # dot (?) product between q_norm and M_R
                    np.stack(  # q norm for multiplication
                        [q_norm_list for _ in range(len(self.Sample.M_veclist))]
                        , axis=1)
                    ,
                    np.stack(
                        [np.inner(q_norm_list, self.Sample.M_veclist) for _ in range(3)]
                        , axis=2)
                    )
                ,
                np.stack(  # M for substraction
                    [self.Sample.M_veclist for _ in range(len(self.Probe.Q_veclist))]
                    , axis=0)
            )

        # unit of M_perp in reciprocal space: [M_perp] = [M] = A/m (like saturation magnetisation)
        return M_perp_QR3


    def _calc_FF_magnetic(self):
        #R_veclist, M_veclist, Q_veclist):
        """
        Reciprocal-space magnetisation scattering length via Fourier transform of \vec{M}\perp\vec{Q}.

        Input:
        R_veclist: moment positions, [m], shape (n_R, 3)
        M_veclist: moments, [Am^2], shape (n_R, 3)
        Q_veclist: scattering vectors to evalute, [m^{-1},] shape (n_Q, 3)

        Output:
        Array with three-dimensional magnetic scattering length in reciprocal space, [m], shape (n_Q, 3)
        """

        M_perp_QR3 = self._calc_Mperp()
        Fourier_argument = self._calc_Fourier_argument()

        #  reciprocal-space components of magnetisation, summed over all positions
        M_perp_FF = (
            np.moveaxis(np.array([
                  np.mean(np.multiply(M_perp_QR3[:, :, 0], Fourier_argument), axis=1)
                , np.mean(np.multiply(M_perp_QR3[:, :, 1], Fourier_argument), axis=1)
                , np.mean(np.multiply(M_perp_QR3[:, :, 2], Fourier_argument), axis=1)
            ])
                , 0, 1  # swapping of axes results in matrix with shape (n_Q, 3)
            )
        )

        # unit: [M_perp_FF] =  A/m (still)
        return b_H * M_perp_FF


    def _calc_scattering_channels(self, struct_FF, M_perp_FF, print_diagnostics=False):
        """
        Calculate mixing of nuclear and magnetic scattering

        Returns:
        Non-spin-flip channels Ipp, Imm = T1 + T3 \pm T2
        Spin-flip channels: Ipm, Imp = T4 - T3 \mp i T5

        Output:
        [Ipp, Imm, Ipm, Imp], all lists with shape (n_Q)
        """

        # stack density_FF so that is has the same shape as Mperp_FF, for easier calculation
        N_FF_3 = np.stack( [ struct_FF for _ in range(3)], axis=1 )

        """ T1 = |density_FF|^2 """
        T1 = np.power(np.abs(struct_FF), 2)

        """ initialise other scattering terms """
        T2 = np.zeros_like( T1 )
        T3 = np.zeros_like( T1 )
        T4 = np.zeros_like( T1 )
        T5 = np.zeros_like( T1 )

        # calculate magnetic scattering contributions
        if self.Sample.is_magnetic is True:

            """ T2 = neutron_polarisation * [ density_FF * Mperp_FF ] """
            T2 = np.inner(
                self.Probe.neutron_polarisation_UVW,
                np.add(
                    np.multiply(np.conjugate(N_FF_3), M_perp_FF)
                    , np.multiply(N_FF_3, np.conjugate(M_perp_FF))
                    )
                )

            """T3 = | neutron_polarisation * M_perp_FF |^2 """
            T3 = np.power(np.abs(
                    np.inner(self.Probe.neutron_polarisation_UVW, M_perp_FF)
                ), 2)

            """ T4 = scalar product of M_perp """
            T4 = np.power(np.linalg.norm(M_perp_FF, axis=-1), 2)

            """ T5 = neutron_polarisation * ( cross prouct of M_perp )"""
            #  maybe there is a faster version for the cross product over the list than a for loop?
            T5 = np.inner(
                self.Probe.neutron_polarisation_UVW
                , np.array([np.cross(M, np.conjugate(M)) for M in M_perp_FF])
                )

        """ Non-spin-flip channels Ipp, Imm = T1 + T3 \pm T2 """
        I_pp = T1 + T3 + T2
        I_mm = T1 + T3 - T2

        """ Spin-flip channels: Ipm, Imp = T4 - T3 \mp i T5 """
        I_pm = T4 - T3 - 1j * T5
        I_mp = T4 - T3 + 1j * T5

        """ Test validity of results -- all results need to be non-complex"""
        if print_diagnostics == True:
            self._print_diagnostics_calc_scattpatt(T1, T2, T3, T4, T5)

        # all scattering intensities need to be real and positive
        # explicitly return real components only, otherwise matplotlib will complain
        return np.real(I_pp), np.real(I_mm), np.real(I_pm), np.real(I_mp)


    def calc_scattering_pattern(self, print_diagnostics=False, uc_repetitions=None):
        """
        Calculate Fourier-transform of structural density and M_perp, and calculate SANS patterns.
        Saves a structured pandas data frame with the result to the Experiment object.

        Scattering lengths are calculated in 10e-6/Angsgrom^2.
        Scattering cross sections are calculated in barn=10e-28 m^2.
        """

        # factor to take into account the periodicity of the structure
        periodicity_of_unitcell = self._calc_FF_lattice(uc_repetitions=uc_repetitions)
        #periodicity_of_unitcell = np.ones_like(periodicity_of_unitcell)

        # calculate structural scattering length
        # [sld_struct] = 1/m^2 * m^3 = m
        # multiply with scattering length density (either scalar or list); the volume is contained in the sphere form factor
        # SASview link: http://www.sasview.org/docs/user/models/sphere.html
        sld_struct = np.multiply(
                self._calc_FF_struct(), periodicity_of_unitcell
            ) / self.Sample.number_of_unit_cells

        # calcualte magnetic scattering length, by taking into account the M components perpendicular to Q only
        # [sld_mag] = m/(Am^2) * Am^2 (mu_B) = m
        sld_magn = np.transpose(
            np.multiply(
                np.transpose(self._calc_FF_magnetic()), periodicity_of_unitcell
                )
            ) / self.Sample.number_of_unit_cells

        # calculate scattering patterns (list of real scalars)
        I_pp, I_mm, I_pm, I_mp = self._calc_scattering_channels( sld_struct, sld_magn, print_diagnostics=print_diagnostics )

        # update data frame - scattering lengths in [fm], cross sections in [barn]
        prefactor_sl = 1. #1./1e15 # m to fm
        prefactor_xs  = 1. #1./1e28 # m^2 to barn
        self.data['sld_struct'] = prefactor_sl * sld_struct # complex
        self.data['sld_mag_U'] = prefactor_sl * sld_magn[:, 0] # complex
        self.data['sld_mag_V'] = prefactor_sl * sld_magn[:, 1] # complex
        self.data['sld_mag_W'] = prefactor_sl * sld_magn[:, 2] # complex
        self.data['I_pp'] = prefactor_xs * I_pp
        self.data['I_mm'] = prefactor_xs * I_mm
        self.data['I_pm'] = prefactor_xs * I_pm
        self.data['I_mp'] = prefactor_xs * I_mp
        # TODO: take into account flipping ratio?
        self.data['I_p'] = self.data['I_pp'] + self.data['I_pm']
        self.data['I_m'] = self.data['I_mm'] + self.data['I_mp']
        self.data['I_sum'] = self.data['I_p'] + self.data['I_m']
        self.data['I_dif'] = self.data['I_p'] - self.data['I_m']
        # spin asymmetry
        self.data['asym'] = self.data['I_dif'] / self.data['I_sum']

        return


    @staticmethod
    def _test_complex_list(complex_list, print_diagnostics=False):
        """ Test if numbers of a list of scalars contains real and/or imaginary components, or are zero. """

        # check if format of input list is correct
        if len(np.shape(complex_list)) != 1:
            if print_diagnostics == True:
                return print('\tInput is not a list of complex scalar numbers!')

        # check for real and imaginary components
        if np.sum(np.abs(complex_list)) == 0:
            statement = 'only zero'
        elif np.sum(np.abs(np.real(complex_list))) == 0:
            statement = 'only imaginary'
        elif np.sum(np.abs(np.imag(complex_list))) == 0:
            statement = 'only real'
        else:
            statement = 'real and imaginary'

        if print_diagnostics is True:
            print(f'\tThe list contains {statement} components.')

        return


    def _print_diagnostics_calc_scattpatt(self, T1, T2, T3, T4, T5):

        print('T1: |nuclear density|^2')
        self._test_complex_list(T1)

        print('T2: neutron_polarisation * mix term')
        self._test_complex_list(T2)

        print('T3: neutron_polarisation * M_perp')
        self._test_complex_list(T3)

        print('T4: | M_perp |^2')
        self._test_complex_list(T4)

        print('T5: neutron_polarisation * (cross product of M_perp)')
        self._test_complex_list(T5)

        print('NOTE: Only T5 should contain imaginary components, the rest should be real or zero!')

        return


    @staticmethod
    def _get_axis_title(column_title):
        """
        Get LaTeX-formatted axis title for a given data frame column, to print in a matplotlib figure.
        """

        # use unicode for the subscripts, otherwise the subscript minus sign may be clipped off in matplotlib
        # TODO: Also add label for the colourmap, to be used for the general plotting function?
        # i.e. either scattering length [m] or scattering cross section [m^2]

        # q columns
        if column_title == 'q_U':
            axis_title = '$q_U$'
        elif column_title == 'q_V':
            axis_title = '$q_V$'
        elif column_title == 'q_W':
            axis_title = '$q_W$'

        # structural scattering lengths
        elif column_title == 'sld_struct':
            axis_title = '$b_N(\\vec{q})$'

        # magnetic scattering lengths
        elif column_title == 'sld_mag_U':
            axis_title = '$b_{M_{\perp,U}}(\\vec{q})$'
        elif column_title == 'sld_mag_V':
            axis_title = '$b_{M_{\perp,V}}(\\vec{q})$'
        elif column_title == 'sld_mag_W':
            axis_title = '$b_{M_{\perp,W}}(\\vec{q})$'

        # net scattering cross sections (using subscripted characters)
        elif column_title == 'I_pp':
            axis_title = '$I₊₊$'
        elif column_title == 'I_mm':
            axis_title = '$I₋₋$'
        elif column_title == 'I_pm':
            axis_title = '$I₊₋$'
        elif column_title == 'I_mp':
            axis_title = '$I₋₊$'
        elif column_title == 'I_p':
            axis_title = '$I₊$'
        elif column_title == 'I_m':
            axis_title = '$I₋$'
        elif column_title == 'I_sum':
            axis_title = '$\Sigma = I₊ + I₋$'
        elif column_title == 'I_dif':
            axis_title = '$\Delta = I₊ - I₋$'
        elif column_title == 'asym':
            axis_title = 'spin asymmetry $\Delta/\Sigma$'

        # column name not recognised
        else:
            axis_title = '???'

        return axis_title


    def _get_data_to_plot(self, column_name, plot_imag=False):
        """ Returns real or imaginary part of given data column """

        if plot_imag == False:
            data_to_plot = np.real( self.data[column_name] )
        else:
            data_to_plot = np.imag( self.data[column_name] )

        return data_to_plot


    def plot_property(self, column_name, ax=None, title='', limit=None, plot_imag=False, contours=True):
        """
        Plot a specified q-dependent property.

        :Parameters:
        *column_name*: string, Name of data column to plot.
            real values: 'I_pp', 'I_mm', 'I_pm', 'I_mp', 'I_p', 'I_m', 'I_sum', 'I_dif', 'asym'
            complex values: 'sld_struct', 'sld_magn_i' with i = U, V, W
        *ax*: matplotlib axis. Axis to plot on.
            Default None. If None, a new figure will be created.
        *title*: string. Axis title.
        *limit*: float. Limit for colourbar. If None, autoscaling values are used.
        *plot_real* Boolean. Default False to plot the real part of the selected data column.
            If Real, the imaginary part will be plottet. Only relevant for complex Fourier transforms of density or magnnetisation.
        *contours* Bool
            Default True. If True, contour lines will be plotted.

        :Returns:
        Axis with plotted data.
        """

        # initialise figure
        plot_cbar = False
        if ax == None:
            fig = plt.figure()
            grid = AxesGrid( fig, 111,
                             nrows_ncols=(1,1),
                             axes_pad=0.45,
                             cbar_mode='single',
                             cbar_location='right',
                             cbar_pad=0.25
                             )
            ax = grid.axes_row[0][0]
            plot_cbar = True
            return_value = fig, ax
        else:
            return_value = None
        ax.axis('equal')

        # figure title
        if len(title) == 0:
            title = self._get_axis_title(column_name)
        ax.set_title(title, y=1.)

        # q axis
        q_limit = np.max(np.abs(np.array(self.data[['q_V', 'q_W']]))) / self._q_unit_scaling_factor
        ax.set_xlim([-q_limit, +q_limit])
        ax.set_ylim([-q_limit, +q_limit])
        ax.set_xlabel(f'{self._get_axis_title("q_V")} ({self._q_unit_label})')
        ax.set_ylabel(f'{self._get_axis_title("q_W")} ({self._q_unit_label})')

        # data to plot and data limits
        data_to_plot = self._get_data_to_plot(column_name, plot_imag=plot_imag)
        if limit == None:
            limit = np.quantile(np.abs(data_to_plot), 0.95)
        elif property == 'asym':
            limit = +1.

        # continously filled contours
        cbar_ref_data = ax.tricontourf(
              self.data['q_V'] / self._q_unit_scaling_factor
            , self.data['q_W'] / self._q_unit_scaling_factor
            , data_to_plot
            , cmap=cmap_pm
            , levels=100
            , norm = colors.Normalize(vmin=-1 * limit, vmax=+1 * limit, clip=False)
            , zorder=1
            )
        # contour levels
        if contours is True:
            ax.tricontour(
                  self.data['q_V'] / self._q_unit_scaling_factor
                , self.data['q_W'] / self._q_unit_scaling_factor
                , data_to_plot
                , levels=11  # less levels do not wash out the spot positions
                , linewidths=0.5, colors='k'
                , zorder=2
                )

        # plot color bar (only if figure & axis have been initialised within function)
        if plot_cbar is True:
            cax = grid.cbar_axes[0]
            # TODO add labels for colorbars!
            # cax.set_ylabel( '$\sigma$', rotation=90 )
            if property == 'asym':
                # add normalisation to -1/+1 for spin asymmetry
                cbar_ref = ax.scatter(
                    [0, 0], [0, 0]
                    , c = [+1., -1.]
                    , cmap = cmap_pm
                    , norm = colors.Normalize( vmin=-1 * limit, vmax=+1 * limit, clip=False )
                    , zorder = -5
                    )
                plt.colorbar( cbar_ref, cax=cax )
            else:
                plt.colorbar( cbar_ref_data, cax=cax )
        return


    def plot_scattering_patterns(self, halfpol=False, figure_title='', subplot_side_length=3., contours=True):
        """
        Overview plot of all channels to the the net scattering cross section.

        :Parameters:
        *halfpol* Bool
            Default False. Plots all scattering channels, plus their sum and difference.
            If True, only the flipper on/off and sum/difference plots are shown.
        *figure_title* String.
            Default ''.
            Figure title.
        *subplot_side_length*: float.
            Default 3.
            Side lenght of subplot axes, in inch.
        *contours* Bool.
            Default True. Whether contour lines should be plotted.

        :Returns:
        None
        TODO: save figure if an output fileneame is given
        """

        # general figure settings
        sns.set_context('talk')
        sns.set_style('whitegrid')

        # if sample is nonmagnetic plot the sum intensity only!
        if self.Sample.is_magnetic is False:
            self.plot_property('I_sum', title=figure_title, contours=contours)

        # plot magnetic scattering patterns
        else:

            # how many subaxes need to be plotted?
            number_cols, quantity_list = 4, ['I_pp', 'I_pm', 'I_p', 'I_sum', 'I_mm', 'I_mp', 'I_m', 'I_dif']
            if halfpol is True:
                number_cols, quantity_list= 2, ['I_p', 'I_sum', 'I_m', 'I_dif']

            # initialise figure
            fig = plt.figure(figsize=(number_cols * subplot_side_length, 2 * subplot_side_length))
            if len(figure_title) > 0:
                fig.suptitle(figure_title)
            grid = AxesGrid(fig, 111,
                            nrows_ncols=(2, number_cols),
                            axes_pad=0.45,
                            cbar_mode='single',
                            cbar_location='right',
                            cbar_pad=0.25
                            )

            # plot the different columns
            limit = np.quantile(np.abs(self.data['I_sum']), 0.975)
            for n, column_name in enumerate(quantity_list):
                ax = grid.axes_row[n // number_cols][n % number_cols]
                self.plot_property(column_name, ax=ax, limit=limit, contours=contours)

            # colour bar from dummy plot
            cbar_ref = ax.scatter(
                [0, 0], [0, 0], c=[-limit, +limit], s=0
                , cmap=cmap_pm
                , vmin=-1 * limit, vmax=+1 * limit
                , zorder=-10
                )
            cax = grid.cbar_axes[0]
            plt.colorbar( cbar_ref, cax=cax )
            # TODO: units of scattering cross section?
            # cbar.ax.set_ylabel('$\sigma$ (??? m$^2$/f.u.)', rotation=90)
            cax.set_ylabel( '$\sigma$', rotation=90 )

        return


    def plot_scattering_lengths(self, plot_imag=True, figure_title='', subplot_side_length=3., contours=False):
        """
        Overview plot of all scattering lengths.

        :Parameters:
        *plot_imag* Bool
            Default False. Plots all scattering lengths including their imaginary parts.
            If True, only the flipper on/off and sum/difference plots are shown.
        *figure_title* String.
            Default ''.
            Figure title.
        *subplot_side_length*: float.
            Default 3.
            Side lenght of subplot axes, in inch.
        *contours* Bool.
            Default True. Whether contour lines should be plotted.

        :Returns:
        None
        TODO: save figure if an output fileneame is given
        """

        # general figure settings
        sns.set_context('talk')
        sns.set_style('whitegrid')

        # how many subaxes need to be plotted?
        number_cols, number_rows , quantity_list= 4, 2, ['sld_struct', 'sld_mag_U', 'sld_mag_V', 'sld_mag_W']
        if plot_imag is False:
            number_rows = 1

        # initialise figure
        fig = plt.figure(figsize=(number_cols * subplot_side_length, 2 * subplot_side_length))
        if len(figure_title) > 0:
            fig.suptitle(figure_title)
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(number_rows, number_cols),
                        axes_pad=0.45,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.25
                        )

        # plot the different columns
        limit = np.quantile(np.abs(self.data['sld_struct']), 0.975)
        for n, column_name in enumerate(quantity_list):
            ax = grid.axes_row[0][n]
            self.plot_property(column_name, ax=ax, limit=limit, contours=contours, plot_imag=False)
            ax.set_title(f'real({self._get_axis_title(column_name)})')
            if plot_imag is True:
                ax = grid.axes_row[1][n]
                self.plot_property(
                    column_name, ax=ax, limit=limit, contours=contours, plot_imag=True
                    )
                ax.set_title( f'imag({self._get_axis_title( column_name )})' )

        # colour bar from dummy plot
        cbar_ref = ax.scatter(
            [0, 0], [0, 0], c=[-limit, +limit], s=0
            , cmap=cmap_pm
            , vmin=-1 * limit, vmax=+1 * limit
            , zorder=-10
            )
        cax = grid.cbar_axes[0]
        plt.colorbar( cbar_ref, cax=cax )
        cax.set_ylabel( '$b$ m/f.u.)', rotation=90 )

        return


    def _get_binned_averages(self, data_column_name, data_imag=False
                             , average_column_name='q_abs', step_delta=None, num_bins=20
                             , lim_min=None, lim_max=None
                             ):
        """
        Get radial or angular binned data of given data column.

        :Parameters:
        *data_column_name*
            Data column to average over.
        *average_column_name* Column to average over.
            Default 'q_abs' for radial average. Set 'q_phi' for angular average.
        *step_delta and num_bins*
            Step width or number of bins. Default 20 bins for |q| average, and 5° for angular average.
        *lim_min and lim_max*
            Default None. Set limits of angular (radial) data range for radial |q| (angular) average.

        :Returns:
        [bin_centers, bin_mean, bin_std]
        """

        # data for average
        subdata = self.data[['q_abs', 'q_phi', data_column_name]].copy()
        if data_imag == False:
            subdata.loc[:, data_column_name] = np.real(subdata[data_column_name])
        else:
            subdata.loc[:, data_column_name] = np.imag(subdata[data_column_name])

        # limit data range, if desired
        limit_column_name = 'q_phi'
        if average_column_name == 'q_phi':
            limit_column_name = 'q_abs'
        if lim_min != None:
            subdata.loc[:,:] = self.data[self.data[limit_column_name] >= lim_min]
        if lim_max != None:
            subdata.loc[:,:] = self.data[self.data[limit_column_name] <= lim_max]

        # get bins  - either of given width, or with a specified number
        bin_limits = []
        if average_column_name == 'q_abs':
            if step_delta is None:
                bin_limits = np.linspace(0, subdata['q_abs'].max(), num_bins)
                step_delta = np.abs(np.mean(np.diff(bin_limits)))
            else:
                bin_limits = np.arange(0., subdata['q_abs'].max() + step_delta, step_delta)
        elif average_column_name == 'q_phi':
            if step_delta is None:
                step_delta = 5.
            bin_limits = np.arange(0., 360. + step_delta, step_delta)

        # get bins, and values and standard deviation of the values
        bin_index = [
            np.argwhere((bin_limits[i] <= np.array(subdata[average_column_name])) & (np.array(subdata[average_column_name]) < bin_limits[i + 1])).ravel()
            for i in range(len(bin_limits) - 1)
            ]

        bin_centers = bin_limits[:-1] + step_delta / 2.
        bin_mean = np.array([np.nanmean(subdata.loc[bin_index[i], data_column_name]) for i in range(len(bin_index))])
        bin_std = np.array([np.nanstd(subdata.loc[bin_index[i], data_column_name], ddof=0) for i in range(len(bin_index))])

        return [bin_centers, bin_mean, bin_std]


    def plot_angular_average(self,
                             column_name, delta_phi=10.
                             , ax=None, title=''
                             , q_min=None, q_max=None, data_imag=False
                             ):
        """
        Plot angular average of a chosen data column.

        :Parameters:
        *column_name* string.
            Column name of quantity to plot.
        *delta_phi*  float.
            Default 10°. Size of angular bins.
        *ax* matplotlib axis
            Default None. Axis to plot data on. If None, a new figure with polar projection will be created.
        *title* string.
            Default ''. Title for axis. If empty string, the column name is used.
        *q_min and q_max* floats.
            Data limits on absolute values of |q|.
        *data_imag* Boolean.
            Default False. If True, the imaginary part of the data will be averaged.

        :Returns:
        Returns plot, respectively axis.
        """

        # TODO: limit q range to vertical/horizontal scale (i.e. same number of data poins in all directions?)

        bin_centers, bin_mean, bin_std = self._get_binned_averages( column_name, average_column_name='q_phi', step_delta=delta_phi, data_imag=data_imag, lim_min=q_min, lim_max=q_max )

        if ax == None:
            fig = plt.figure()
            ax = plt.subplot(projection='polar')
            output_value = fig
            x_data = np.deg2rad(bin_centers)
        else:
            x_data = bin_centers
            output_value = ax

        if title == '':
            title = self._get_axis_title(column_title=column_name)
        ax.set_title(title)

        # plot values (positive and negative)
        for i, sign in enumerate([+1., -1.]):

            selection = np.argwhere(
                np.sign(sign * bin_mean) == +1.
                ).ravel()
            ax.errorbar(
                x_data[selection]
                , sign * bin_mean[selection]
                , bin_std[selection]
                , marker='.'
                , linewidth=0
                , elinewidth=1
                , color=['red', 'blue'][i]
            )

        return #output_value


    def plot_radial_average(self, column_name, num_bins=20, delta_q=None, ax=None, title='', phi_min=None, phi_max=None, data_imag=False):
        """
        Plot radial average of a chosen data column.

        :Parameters:
        *column_name* string.
            Column name of quantity to plot.
        *num_bins or delta_q*  Integer or float
            Number of bins (default 20), or width of q-bins (if step size is given it overrides the number of bins).
        *ax* matplotlib axis
            Default None. Axis to plot data on. If None, a new figure will be created.
        *title* string.
            Default ''. Title for axis. If empty string, the column name is used.
        *q_min and q_max* floats.
            Data limits on absolute values of |q|.
        *data_imag* Boolean.
            Default False. If True, the imaginary part of the data will be averaged.

        :Returns:
        Returns plot, respectively axis.
        TODO: Errorbars do not really represent uncertainty of less data points in the corners.
        """

        bin_centers, bin_mean, bin_std = self._get_binned_averages(
            column_name, average_column_name='q_abs', step_delta=delta_q, num_bins=num_bins
            ,data_imag=data_imag, lim_min=phi_min, lim_max=phi_max)

        if ax == None:
            fig = plt.figure()
            ax = plt.subplot()
            output_value = fig
        else:
            output_value = ax

        if title == '':
            title = self._get_axis_title(column_title=column_name)
        ax.set_title(title)
        ax.set_xlabel(f'|q| ({self._q_unit_label})')
        ax.set_xlim(0, self.data['q_abs'].max() / self._q_unit_scaling_factor)

        # plot values
        ax.errorbar(
            bin_centers / self._q_unit_scaling_factor
            , bin_mean
            , bin_std
            , marker='.'
            , linewidth=0
            , elinewidth=1
            #, color='black'
        )

        return #output_value

