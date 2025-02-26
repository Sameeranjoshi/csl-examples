

import os
from typing import Optional
from pathlib import Path
import shutil
import subprocess
import random
import numpy as np
from scipy.sparse.linalg import eigs
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
from cmd_parser import parse_args
from util import (
    hwl_2_oned_colmajor,
    oned_to_hwl_colmajor,
    laplacian,
    csr_7_pt_stencil,
)
from cg import conjugateGradient
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg
import matplotlib.pyplot as plt
import utilities as ut
from dataclasses import dataclass
import json
from sklearn.datasets import make_sparse_spd_matrix

@dataclass
class CompileCoreArgs:
    cslc: str
    pe_rows: int
    pe_cols: int
    file_config: str
    elf_dir: str
    fabric_width: int
    fabric_height: int
    core_fabric_offset_x: int  # fabric-offsets of the core
    core_fabric_offset_y: int
    use_precompile: bool
    arch: Optional[str]
    channels: int
    width_west_buf: int
    width_east_buf: int
    
def calculate_fabric_dimensions(args, width, height, width_west_buf, width_east_buf):
  # fabric-offsets = 1,1
  fabric_offset_x = 1
  fabric_offset_y = 1
  # starting point of the core rectangle = (core_fabric_offset_x, core_fabric_offset_y)
  # memcpy framework requires 3 columns at the west of the core rectangle
  # memcpy framework requires 2 columns at the east of the core rectangle
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf #4
  core_fabric_offset_y = fabric_offset_y  #1
  # (min_fabric_width, min_fabric_height) is the minimal dimension to run the app
  min_fabric_width = (core_fabric_offset_x + width + 2 + 1 + width_east_buf)
  min_fabric_height = (core_fabric_offset_y + height + 1)

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  return fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y

def parse_and_get_arguments():
	args, dirname = parse_args()
	# Get params from compile metadata
	with open(f"{dirname}/out.json", encoding='utf-8') as json_file:
			compile_data = json.load(json_file)
	pe_rows = int(compile_data['params']['pe_rows'])
	pe_cols = int(compile_data['params']['pe_cols'])
	
	code_csl = "./src/layout_amg.csl"
	if args.driver is not None:
			cslc = args.driver
	else:
			cslc = "cslc"
	channels = args.channels
	assert channels <= 16, "only support up to 16 I/O channels"
	assert channels >= 1, "number of I/O channels must be at least 1"
	width_west_buf = args.width_west_buf
	width_east_buf = args.width_east_buf  
	fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y = calculate_fabric_dimensions(args, pe_cols, pe_rows, width_west_buf, width_east_buf)

	
	layout_arguments = CompileCoreArgs(
			cslc, pe_cols, pe_rows, code_csl, dirname,
			fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y,  # fabric details
			args.run_only, args.arch, channels,
			width_west_buf, width_east_buf
	)
	return layout_arguments

def host_calculations(input_data, layout_arguments):
  A = input_data["A"]
  x = input_data["x"]
  b = input_data["b"]

  # Compute the residual
  residual_host = b - A @ x
  return residual_host

def main():
  random.seed(127)
  ############################################################
  # Parse arguments
  ############################################################
  layout_arguments = parse_and_get_arguments()
  print(layout_arguments)
  
  ############################################################
  # Input data 
  ############################################################
  matrix_rows = 4
  matrix_cols = 4
  M = matrix_rows
  N = matrix_cols
  
  A = np.arange(M*N, dtype=np.float32).reshape(M, N)
  x = np.full(shape=N*1, fill_value=1.0, dtype=np.float32)
  b = np.zeros(shape=M*1, dtype=np.float32)

  print("Input Matrix Shape:", A.shape)
  print("Input b_1d Shape:", b.shape)
  print("Input x_1d Shape:", x.shape)
  ut.visualize_matrix_or_vector(A, title="A Matrix", filename="images/A.png")
  ut.visualize_matrix_or_vector(x, title="x Vector", filename="images/x.png")
  ut.visualize_matrix_or_vector(b, title="b Vector", filename="images/b.png")
  
  input_data = {
    "A": A,
    "x": x,
    "b": b,
  }

  ############################################################
  # HOST CALCULATIONS
  ############################################################
  residual_host = host_calculations(input_data, layout_arguments)
  print("Residual Host:", residual_host)
  ############################################################
  # DEVICE CALCULATIONS
  ############################################################
  

 
  
if __name__ == "__main__":
  main()
