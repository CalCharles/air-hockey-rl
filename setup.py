from setuptools import setup, find_packages

setup(
    name='airhockey', 
    version='0.1.0', 
    author='Michael Munje',  
    author_email='michaelmunje@utexas.edu',  
    description='Environment for Air Hockey', 
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown', 
    url='http://github.com/michaelmunje/airhockey-rl', 
    packages=find_packages(),  
    python_requires='>=3.7',
    install_requires=[
        'mujoco',
        'mujoco-py',
        'box2d-py',
        'gymnasium',
        'gym==0.25.1'
        'box2d-py==2.3.8',
        'tqdm',
        'h5py',
        'xmltodict',
        'robosuite',
    ],
    extras_require={
        'train': [
            'opencv-python',
            'tensorboard',
            'wandb',
            'stable-baselines3==2.2.1',
            'pyrallis',
            'torch',
            'scikit-learn',
        ],
    },
)
