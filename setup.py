from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros2_bebop_ibvs'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'models'), glob('models/****/***/**/*')),
        (os.path.join('share', package_name, 'models','parrot_bebop_2'), glob('models/parrot_bebop_2/model.sdf')),
        (os.path.join('share', package_name, 'models','parrot_bebop_2'), glob('models/parrot_bebop_2/model.config')),
        (os.path.join('share', package_name, 'models','parrot_bebop_2','meshes'), glob('models/parrot_bebop_2/meshes/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jr',
    maintainer_email='juliordzcer@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "ibvs_sim = ros2_bebop_ibvs.ibvs_sim:main",
            "ibvs_real = ros2_bebop_ibvs.ibvs_real:main",
            "ibfc_sim = ros2_bebop_ibvs.ibfc_sim:main",
            "sim_control = ros2_bebop_ibvs.sim_control:main",
        ],
    },
)
