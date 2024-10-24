from setuptools import find_packages, setup

setup(
    name="dagster_scorecard",
    packages=find_packages(exclude=["dagster_scorecard_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
