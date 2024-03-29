SCOPE's Website Development Guide
---------

## Prerequisite

Clone the source code of the scope into your device:

```bash
git clone git@github.com:abess-team/scope.git
cd scope
```

Install dependent libraries:

```bash
cd docs
pip install -r requirements.txt
# conda install -c conda-forge pandoc # it works well on windows11, python3.9 without pandoc
```

## Generate Website's Content  

Conduct the following command to generate the content of website:

```bash
make html
```

You can see the website's contents by opening `index.html` file
in the directory `docs/build/html` with your browser.

For more components and details, please refer to [pydata-sphinx-theme](pydata-sphinx-theme.readthedocs.io)
