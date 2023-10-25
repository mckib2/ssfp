
# Remove any existing distribution archives
rm -rf dist

# Generate distribution archives
python -m pip install --upgrade build setuptools wheel
python -m build

# Upload
python -m pip install --upgrade twine
python -m twine upload dist/*
