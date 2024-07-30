#!/usr/bin/env python

from setuptools import setup, find_packages
setup(name='latentvs',
      version='1.0',
      description='Code for latent space based visual servoing methods, presented in Samuel Felton\'s thesis: "Deep latent representations for visual servoing"',
      author='Samuel Felton',
      packages=find_packages(),
      #package_dir={'latentvs': '.'},
      author_email='samuel.felton@irisa.fr',
)