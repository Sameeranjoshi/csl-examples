# The SDK runs in a container with a different python environment than the host.
# This script installs a python package in the container's python environment.

# Running from your application workdir:
# local_py_path=`which python`
# bash /path/to/sdk_install_python_package.sh ./$local_py_path my_pip_package_name
# bash sdk_install_python_package.sh ~/csl/sparse_csl/venv_new/bin/ pyamg

py_path=$(realpath $1)
package_name=$2
SINGULARITYENV_PYTHONPATH="$(realpath $py_path)"
cs_python -c "
import subprocess
import sys
package='$2'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--target=$py_path', package])
"
# echo -e "Please do\nexport SINGULARITYENV_PYTHONPATH=\"$SINGULARITYENV_PYTHONPATH\"\nbefore using cs_python"

echo -e "Please do\nexport SINGULARITYENV_PYTHONPATH=\"$SINGULARITYENV_PYTHONPATH:/cbcore/py_root:/cbcore/py_root/cerebras\"\n before using cs_python"

