from setuptools import setup, find_packages

setup(
    name="tigeralgorithmexample",
    version="0.0.1",
    author="Xulin Chen",
    author_email="chenxl@cellsvision.com",
    packages=find_packages(),
    license="LICENSE.txt",
    install_requires=[
        #"numpy==1.20.2",
        "tqdm==4.62.3",
        'yacs'
    ],
)
