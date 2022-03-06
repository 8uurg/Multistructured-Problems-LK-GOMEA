project('MO_GOMEA', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'])

rectangular_lsap_dep = dependency('rectangular_lsap', fallback : ['rectangular_lsap', 'rectangular_lsap_dep'])

traceback_deps = []

executable('MO_GOMEA',
           'MO_GOMEA.cpp',
           dependencies: [rectangular_lsap_dep] + traceback_deps)
