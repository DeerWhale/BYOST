"""
BYOT (build your own spectral template)

Author: Jing Lu (lujingeve158@gmail.com)

This is a package that can build a spectral template based on given spectra and 2 attached conditions. 
The method is initially developed for the NIR spectral template of type Ia supernovae, see 
https://arxiv.org/abs/2211.05998

PS:I really wanted to name it noodle since I LOVE noodles! Maybe Nir spectrOscOpic Diversity tempLatE (NOODLE)?
"""

from .build import *
from .template import *
from .merge_spec import *
from .visualize import *
