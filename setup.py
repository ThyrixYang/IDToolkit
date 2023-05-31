from distutils.core import setup

setup(name="inverse design benchmark",
      version='0.1',
      description='',
      author='',
      author_email='',
      url='',
      include_package_data=True, 
      packages=['inverse_design_benchmark', 
                'inverse_design_benchmark.utils', 
                'inverse_design_benchmark.envs', 
                'inverse_design_benchmark.algorithms', 
                'inverse_design_benchmark.assets', 
                'inverse_design_benchmark.parameter_space', 
                'inverse_design_benchmark.algorithms.nn_models', 
                ],
      # package_dir={
      #       'inverse_design_benchmark': 'inverse_design_benchmark'
      #       },
      install_requires=[
            # "python>=3.7", "rich"
      ],
)
