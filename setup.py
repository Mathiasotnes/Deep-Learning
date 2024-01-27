from setuptools import setup, find_packages

setup(
    name='brain_of_mathias',
    version='0.1',
    packages=find_packages(),
    description='The brain of Mathias encoded into a neural net',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Mathias Otnes',
    author_email='mathias.otnes@gmail.com',
    url='https://github.com/mathiasotnes/Deep-Learning',
    install_requires=[
        'numpy',
        'matplotlib.pyplot',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
