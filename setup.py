# Following https://awaywithideas.com/the-optimal-python-project-structure/
import setuptools

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="cell_counter", packages=["cell_counter"], install_requires=install_requires
)
