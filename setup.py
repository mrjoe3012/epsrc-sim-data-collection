from setuptools import setup

package_name = 'sim_data_collection'
submodules = [
    'sim_data_collection/data_collector',
    'sim_data_collection/analysis',
    'sim_data_collection/perception_model'
]

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name] + submodules,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/good.json', 'models/realistic.json', 'models/poor.json']),
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
            'integrity_check = sim_data_collection.analysis.integrity_check:main',
            "analysis = sim_data_collection.analysis.analysis_main:main",
            "perception_model = sim_data_collection.perception_model.main:main",
        ],
    },
)
