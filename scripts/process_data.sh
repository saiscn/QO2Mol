proj_root=$(dirname "$(dirname "$(readlink -f "$0")")")
python "$proj_root/src/dataset/qo2mol/qo2mol.py"
