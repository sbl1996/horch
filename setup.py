from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "horch._cy",
        ["horch/_cy.pyx"],
    )
]

setup(name='horch',
      version='0.1',
      description='',
      url='http://github.com/sbl1996/horch',
      author='HrvvI',
      author_email='sbl1996@126.com',
      license='MIT',
      packages=['horch'],
      zip_safe=False,
      ext_modules = cythonize(extensions),
      extra_compile_args=['-O2', '-march=native'],
      extra_link_args=['-O2', '-march=native'],
)