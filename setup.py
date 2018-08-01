from setuptools import setup

setup(name='im2txt',
      version='0.1',
      description='An image captioning framework using tensorflow',
      url='http://github.com/brandontrabucco/im2txt',
      author='Brandon Trabucco',
      author_email='brandon@btrabucco.com',
      license='MIT',
      packages=['im2txt', 'im2txt.ops'],
      zip_safe=False)
