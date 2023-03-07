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
        ('lib/' + package_name + '/models/',['models/experimental.py', 'models/common.py',
                                             'models/yolo.py']),
        ('lib/' + package_name + '/utils/',['utils/general.py', 'utils/torch_utils.py',
                                             'utils/plots.py', 'utils/datasets.py',
                                             'utils/google_utils.py', 'utils/activations.py',
                                             'utils/add_nms.py', 'utils/autoanchor.py',
                                             'utils/loss.py', 'utils/metrics.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marno Nel & rshima',
    maintainer_email='marthinusnel2023@u.northwestern.edu & rintarohshima2023@u.northwestern.edu',
    description='A ROS 2 package to deploy YOLOv7 trained models on',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection = object_detection.object_detection:main'
        ],
    },
)
