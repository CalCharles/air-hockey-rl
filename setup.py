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
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
    install_requires=[
        'mujoco',
        'mujoco-py',
        'box2d-py',
        'gymnasium',
        'box2d-py==2.3.8',
        'tqdm',
        'h5py',
    ],
    extras_require={
        'train': [
            'opencv-python',
            'tensorboard',
            'wandb',
            'stable-baselines3==2.2.1',
            'pyrallis',
            'torch',
        ],
    },
)
