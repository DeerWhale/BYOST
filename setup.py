from setuptools import setup, find_packages

VERSION = '0.0.3' 
DESCRIPTION = 'BYOST (Build Your Own Spectral Template)'
LONG_DESCRIPTION = 'Using Pricipal Component Analysis and Gaussian Process Regression\
                     to construct spectral template given 2 conditions.\
                     see https://arxiv.org/abs/2211.05998 for details.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="BYOST", 
        version=VERSION,
        author="Jing Lu",
        author_email="lujingeve158@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'astropy',
            'pandas',
            'scikit-learn',
            'tqdm'
        ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Topic :: Scientific/Engineering :: Astronomy"
        ]
)
