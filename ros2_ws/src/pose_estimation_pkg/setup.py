from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'pose_estimation_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Samanth Krishna',
    maintainer_email='samanthkrishna2001@gmail.com',
    description='6-DoF pose estimation using FoundationPose and SAM2 with ROS2 Jazzy',
    license='MIT',
    entry_points={
        'console_scripts': [
            'foundationpose_node = pose_estimation_pkg.foundationpose_node:main',
            'sam2_node = pose_estimation_pkg.sam2_node:main',
            'benchmark_node = pose_estimation_pkg.benchmark_node:main',
        ],
    },
)