from setuptools import setup

package_name = 'sim_data_collection'
submodules = ['sim_data_collection/data_collector', 'sim_data_collection/analysis']

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name] + submodules,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joseph Agrane',
    maintainer_email='josephagrane@gmail.com',
    description='This package serializes ROS messages from simulation runs into a database in which the relationships between emssages are directly exposed.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_collector = sim_data_collection.data_collector.main:main',
            'health_check = sim_data_collection.analysis.health_check_main:main',
        ],
    },
)
