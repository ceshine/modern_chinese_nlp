from distutils.core import setup

setup(
    name='dekisugi',
    version='0.0.1',
    packages=[
        'dekisugi',
    ],
    install_requires=[
        'sentencepiece',
        'torch>=0.4.1',
        'tqdm',
        'opencc-python-reimplemented',
        'click',
        'pandas'
    ],
    license='MIT',
    long_description=open('README.md').read(),
)
