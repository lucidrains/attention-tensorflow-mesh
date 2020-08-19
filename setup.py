from setuptools import setup, find_packages

setup(
  name = 'attention-tensorflow-mesh',
  packages = find_packages(),
  version = '0.0.2',
  license='MIT',
  description = 'A bunch of attention related functions, for constructing transformers in tensorflow mesh',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/attention-tensorflow-mesh',
  keywords = ['transformers', 'artificial intelligence'],
  install_requires=[
      'mesh-tensorflow',
      'tensorflow-gpu>=1.15'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)