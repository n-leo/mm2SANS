import numpy as np

class Detector:
    """
    Geometry of SANS detector.
    """

    def __init__(self, sans_instrument=None):
        """ 
        Properties of SANS pixel detector.
        
        :Parameters:
        *sans_instrument*: None or string
            *None* or *PSI_SANS1*: 128 x 128 pixels measuring 7.5 mm x 7.5 mm
            
        :Returns:
        Detector object which stores detector settings.
        Determines q-map together with neutron wavelength and detector distance.
        """
        
        self.instrument = sans_instrument

        # [int] number of pixels along the beam direction (equals one for a 2D detector)
        self.pixel_number_U = 1
        
        # detector settings (usually not changed for the beamlines)
        if (sans_instrument == None) or (sans_instrument == 'PSI_SANS1'):
            
            self.pixel_size_V = self.pixel_size_W = 7.5e-3 # [float in m] horizontal and vertical pixel size
            self.pixel_number_V = self.pixel_number_W = 128  # [int] number of pixels in horizontal and vertical direction

        elif sans_instrument == 'test':
            self.pixel_size_V = self.pixel_size_W = 2 * 7.5e-3  # [float in m] horizontal and vertical pixel size
            self.pixel_number_V = self.pixel_number_W = 128 / 2 # [int] number of pixels in horizontal and vertical direction

        return
    
    
    def print_detector_info(self):
        """ Print short summary of detector properties. """
        
        info_string = f'{self.instrument} detector has {self.pixel_number_V} x {self.pixel_number_W} pixels'
        info_string += f' with a size of {np.round(self.pixel_size_V*1e3,2):.2f} mm x {np.round(self.pixel_size_W*1e3,2):.2f} mm.'
        
        return print(info_string)
    
    