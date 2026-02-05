from setuptools import setup, find_packages

setup(
    name="tibetan_hubert",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "pyworld>=0.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.12.0",
        "scikit-learn>=1.2.0",
    ],
    python_requires=">=3.8",
)
