from setuptools import setup

package_name = 'object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml',
                                   'launch/object_detection.launch.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rshima',
    maintainer_email='rintarohshima2023@u.northwestern.edu',
    description='This package wraps YOLOv7 into a ROS2 node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection = object_detection.object_detection:main'
        ],
    },
)
