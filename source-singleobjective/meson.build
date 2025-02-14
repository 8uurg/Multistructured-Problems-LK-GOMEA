#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

project('wGOMEA', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3',
                     'cpp_std=c++17'])

include_dir_self = include_directories('include_GOMEA')

include_dirs = [include_dir_self]

pymod = import('python')
python = pymod.find_installation()
python_dep = python.dependency(embed: true)
message('Compiling using Python @0@'.format(python_dep.version()))

gomea_dependencies = [python_dep]

# Windows does not have getopt.h, use the included code
# that implements getopt for Windows.
if host_machine.system() == 'windows'
  getopt_dep = dependency('getopt', fallback : ['getopt', 'getopt_dep'], optional: true)
  gomea_dependencies += [getopt_dep]
endif

# Note: meson dislikes globbing (as it is slow). As such, glob ahead of time
# using `update_files.sh` (srcfiles don't change too often!) and 
# store it in a file that is tracked.
files = run_command('cat', files('files')).stdout().strip().split('\n')

exe = executable('GOMEA', 
    files, 
    include_directories: include_dirs, 
    dependencies: gomea_dependencies)
