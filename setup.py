from setuptools import setup

description = 'Calculation of SANS scattering patterns from micromagnetic simulations'
date_version_update = '2021-07-06'
version = '0.1.1'

# open README file
def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='mm2SANS',
    version=version,
    packages=['mm2SANS'],
    url='https://github.com/n-leo/mm2SANS',
    license='GPL 3.0',
    author='Naëmi leo',
    author_email='naemi.leo@alumni.ethz.ch',
    description=description,
	keywords='small angle neutron scattering, magnetism, micromagnetic simulations',
    classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3 :: Only'
		],
	long_description=readme(),
	long_description_content_type='text/markdown',
    install_requires=[
		'numpy >= 1.15',
		'pandas >= 0.20',
		'scipy >= 1.2',
		'matplotlib >= 3.3',
		'seaborn >= 0.9',
		],
)
