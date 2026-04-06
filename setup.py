from setuptools import setup, find_packages

setup(
    name='binaryneat',                    # Consider renaming to 'sneat' or 'binary-sneat' since the code is heavily modified from sNEAT
    version='1.0.1',
    packages=find_packages(),
    package_data={'binaryneat': ['default_config.ini']},
    
    install_requires=[
        'numpy>=1.24.0',          # Broad but safe range
        'networkx>=3.1',
        'matplotlib>=3.7.0',
        'tabulate>=0.9.0',
        'tqdm>=4.65.0',
    ],
    
    extras_require={
        'examples': [
            'gymnasium>=1.0.0a2',
            # gymnasium[box2d], [classic_control], [mujoco] are usually installed separately by users
        ]
    },
    
    python_requires='>=3.9',      # Add this — good practice
    description='Modified sNEAT with binary/gate-based neural networks',
    # author, url, license, etc. recommended
)