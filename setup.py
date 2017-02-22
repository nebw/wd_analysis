from distutils.core import setup
from pip.req import parse_requirements


install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]

setup(
    name='wd_analysis',
    version='0.1.0',
    description='',
    entry_points={
        'console_scripts': [
            'wda = wda.scripts.wda:main',
        ]
    },
    install_requires=reqs,
    dependency_links=dep_links,
    extras_require={
    },
    packages=[
        'wda',
    ],
    package_data={
    }
)
