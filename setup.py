from setuptools import setup, find_packages

setup(
    name='crtemplate_analysis',
    version='0.0.3',
    url='https://github.com/matthiasplum/cosmicray-analysis-tools',
    author='Matthias Plum',
    author_email='matthias.plum@sdsmt.edu',
    description='Description of my package',
    packages=['crtemplate_analysis'],
    install_requires=['numpy >= 1.11.1','scipy >= 1.9.0', 'matplotlib >= 3.0.0', 'iminuit >= 2.16.0'],
)
