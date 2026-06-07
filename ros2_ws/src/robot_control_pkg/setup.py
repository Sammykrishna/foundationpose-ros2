from setuptools import setup
import os
from glob import glob

package_name = 'robot_control_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf.xacro')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Samanth Krishna',
    maintainer_email='samanthkrishna2001@gmail.com',
    description='MoveIt2 grasp executor for FoundationPose pipeline',
    license='MIT',
    entry_points={
        'console_scripts': [
            'grasp_executor = robot_control_pkg.grasp_executor:main',
        ],
    },
)