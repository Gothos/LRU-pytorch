
from distutils.core import setup
setup(
  name = 'LRU-pytorch',         
  packages = ['LRU_pytorch'],  
  version = '0.1.2',     
  license='MIT',       
  description = 'Linear Recurrent Unit (LRU) - Pytorch',  
  author = 'Vishnu Jaddipal',                  
  author_email = 'zeus.vj2003@gmail.com',     
  url = 'https://github.com/Gothos/LRU-pytorch',   
  download_url = 'https://github.com/Gothos/LRU-pytorch/archive/refs/tags/v0.1.1-alpha.tar.gz',  
  keywords = ['Artificial Intelligence', 'Deep Learning', 'Recurrent Neural Networks'],   
  install_requires=[            
          'torch>=1.13'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',    
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    
  ],
)
