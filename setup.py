from setuptools import setup, find_packages

setup(
    name='binaryneat',
    version='1.0.3',
    packages=find_packages(),
    package_data={'binaryneat': ['default_config.ini']},
    install_requires=[
        'matplotlib==3.9.0',
        'networkx==3.3',
        'numpy==1.26.4',
        'tabulate==0.9.0',
        'tqdm==4.66.4'
    ],
    extras_require={
        'examples': [
            'gymnasium==1.0.0a2',
            'gymnasium[box2d]',
            'gymnasium[classic_control]',
            'gymnasium[mujoco]'
        ]
    }
)