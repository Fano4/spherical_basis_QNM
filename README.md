# spherical_basis_QNM

This code computes the quasi-normal modes of an ensemble of spherical nanoparticles

# To build and upload the package

$ rm -r dist/

$ python3 -m build

$ python3 -m twine upload --verbose --repository testpypi dist/*

# To install the package

python3 -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps quantum-dyn==0.0.1.2
