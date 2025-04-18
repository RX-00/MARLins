from setuptools import setup, find_packages

setup(
    name='gym_kilobots',
    version='0.0.1',
    install_requires=[
        # gymnasium and stable-baselines3 are required via pip
        'box2d-py',
        'numpy',
        'scipy',
        'pygame',
        'matplotlib'
    ],
    packages=find_packages()
)