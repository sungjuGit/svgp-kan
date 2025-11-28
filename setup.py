from setuptools import setup, find_packages

setup(
    name="svgp_kan",
    version="0.1.0",
    description="Scalable Probabilistic Kolmogorov-Arnold Networks using Sparse Variational GPs",
    author="Open Source Contributor",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "matplotlib",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
