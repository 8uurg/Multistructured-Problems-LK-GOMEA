project('getopt', 'c')

getopt_inc = include_directories('include')
getopt_src = ['src/getopt.c']

getopt_lib = static_library('getopt', sources: getopt_src, include_directories: getopt_inc)
getopt_dep = declare_dependency(link_with: getopt_lib, include_directories: getopt_inc)