project('rectangular_lsap', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++14'])

rectangular_lsap_inc = include_directories('.')
rectangular_lsap_lib = static_library('rectangular_lsap', 'rectangular_lsap.cpp')

rectangular_lsap_dep = declare_dependency(link_with: rectangular_lsap_lib, include_directories: rectangular_lsap_inc) 