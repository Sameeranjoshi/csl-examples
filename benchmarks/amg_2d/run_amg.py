

import os
import argparse
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
import amg as amg 

@dataclass
class CompileCoreArgs:
    cslc: str
    layout_file: str
    arch: str
    fabric_width: int
    fabric_height: int
    core_fabric_offset_x: int
    core_fabric_offset_y: int
    pe_rows: int
    pe_cols: int
    M: int
    N: int
    R_M: int
    R_N: int
    channels: int
    width_west_buf: int
    width_east_buf: int
    elf_folder: str
    extra_args: str
 
def calculate_fabric_dimensions(width, height, width_west_buf, width_east_buf):
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
  
  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  return fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y

def parse_and_get_arguments(run_args, logs_dir):
  # args, logs_dir = parse_args()
  # Get params from compile metadata
  cslc = "cslc"  
  layout_file = "./src/layout_amg.csl"
  
  with open(f"{logs_dir}/out.json", encoding='utf-8') as json_file:
      compile_data = json.load(json_file)
  pe_rows = int(compile_data['params']['pe_rows'])
  pe_cols = int(compile_data['params']['pe_cols'])
  M = int(compile_data['params']['M'])
  N = int(compile_data['params']['N'])
  R_M = int(compile_data['params']['R_M'])
  R_N = int(compile_data['params']['R_N'])
  # do check that M should be equal to N, to be square matrix
  assert M == N, "M should be equal to N, to be square matrix"
  
  channels = 1
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"
  width_west_buf = 0
  width_east_buf = 0
  fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y = calculate_fabric_dimensions(pe_cols, pe_rows, width_west_buf, width_east_buf)


  layout_arguments = CompileCoreArgs(
      run_args, logs_dir, cslc, pe_cols, pe_rows, M, N, R_M, R_N, layout_file, logs_dir,
      fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y,  # fabric details
      run_args.run_only, run_args.arch, channels,
      width_west_buf, width_east_buf
  )
  return layout_arguments

def compile_csl_core(layout_arguments: CompileCoreArgs):
    
    args = []
    args.append(layout_arguments.cslc) # command
    args.append(layout_arguments.layout_file)
    args.append(f"--arch={layout_arguments.arch}")
    args.append(f"--fabric-dims={layout_arguments.fabric_width},{layout_arguments.fabric_height}")
    args.append(f"--fabric-offsets={layout_arguments.core_fabric_offset_x},{layout_arguments.core_fabric_offset_y}")    
    args.append(f"--params=pe_rows:{layout_arguments.pe_rows},pe_cols:{layout_arguments.pe_cols},M:{layout_arguments.M},N:{layout_arguments.N},R_M:{layout_arguments.R_M},R_N:{layout_arguments.R_N}")
    args.append("--memcpy")
    args.append(f"--channels={layout_arguments.channels}")
    args.append(f"--width-west-buf={layout_arguments.width_west_buf}")
    args.append(f"--width-east-buf={layout_arguments.width_east_buf}")
    args.append(layout_arguments.extra_args)
    args.append(f"-o={layout_arguments.elf_folder}")
    
    print(f"subprocess.check_call(layout_args = {args})")
    subprocess.check_call(args)
    
def logs(run_args, logs_dir):

  if run_args.cmaddr is None:
    # move simulation log and core dump to the given folder
    dst_log = Path(f"{logs_dir}/sim.log")
    src_log = Path("sim.log")
    if src_log.exists():
      shutil.move(src_log, dst_log)

    dst_trace = Path(f"{logs_dir}/simfab_traces")
    src_trace = Path("simfab_traces")
    if dst_trace.exists():
      shutil.rmtree(dst_trace)
    if src_trace.exists():
      shutil.move(src_trace, dst_trace)
    
    # Move all files and directories starting with "sim" to the logs_dir
    for item in Path().glob('sim*'):
        dest = Path(logs_dir) / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(logs_dir))


    
def host_calculations(v_cycle_data):

  # unpack v_cycle_data
  ml = v_cycle_data["ml"]
  setup_config = v_cycle_data["setup_config"]
  x_level = v_cycle_data["x_level"]
  b_level = v_cycle_data["b_level"]

  A, P, R, x, b = ml.levels[0].A, ml.levels[0].P, ml.levels[0].R, x_level[0], b_level[0]
  b_coarse_host, x_coarse_host = amg.each_layer_solver(A, b, x, R, ml, setup_config, 0)
  return b_coarse_host, x_coarse_host

def device_calculations(v_cycle_data):
  run_args, logs_dir = parse_args()
  
  ############################################################
  # unpack data
  ############################################################
  # unpack the input data
  ml = v_cycle_data["ml"]
  setup_config = v_cycle_data["setup_config"]
  x_level = v_cycle_data["x_level"]
  b_level = v_cycle_data["b_level"]

  ############################################################
  # CREATE DYNAMIC LAYOUT
  ############################################################
  A, P, R, x, b = ml.levels[0].A.toarray(), ml.levels[0].P.toarray(), ml.levels[0].R.toarray(), x_level[0], b_level[0]
  [M,N] = A.shape
  [R_M, R_N] = R.shape
  pe_cols = 1
  pe_rows = 1
  layout_arguments = generate_dynamic_layout(run_args, pe_cols_=pe_cols, pe_rows_=pe_rows, M_=M, N_=N, R_M_=R_M, R_N_=R_N)
  
  ############################################################
  # Setup simulator
  ############################################################
  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  simulator = SdkRuntime(run_args.elffolder, cmaddr=run_args.cmaddr)

  # data accessible from host and device.
  symbol_residual = simulator.get_id("residual")
  symbol_b_coarse = simulator.get_id("b_coarse")
  symbol_A = simulator.get_id("A")
  symbol_x = simulator.get_id("x")
  symbol_b = simulator.get_id("b")
  symbol_R = simulator.get_id("R")
  symbol_x_smooth = simulator.get_id("x_smooth")

  simulator.load()
  simulator.run()

  ############################################################
  # H2D
  ############################################################
  simulator.memcpy_h2d(symbol_x, x, 0, 0, pe_cols, pe_rows, N*1, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
  simulator.memcpy_h2d(symbol_A, A.flatten(order='C'), 0, 0, pe_cols, pe_rows, M*N, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
  simulator.memcpy_h2d(symbol_R, R.flatten(order='C'), 0, 0, pe_cols, pe_rows, R_M*R_N, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
  simulator.memcpy_h2d(symbol_b, b, 0, 0, pe_cols, pe_rows, M*1, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
  ############################################################
  # Kernel
  ############################################################
  simulator.launch('jacobi', nonblock=False)
  simulator.launch('compute', nonblock=False)

  ############################################################
  # D2H
  ############################################################
  x_smooth_device = np.zeros([N*1], dtype=np.float32)
  simulator.memcpy_d2h(x_smooth_device, symbol_x_smooth, 0, 0, 1, 1, N*1, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
  residual_device = np.zeros([M*1], dtype=np.float32)
  simulator.memcpy_d2h(residual_device, symbol_residual, 0, 0, 1, 1, M*1, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
  b_coarse_device = np.zeros([R_M*1], dtype=np.float32)
  simulator.memcpy_d2h(b_coarse_device, symbol_b_coarse, 0, 0, 1, 1, R_M*1, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
  ############################################################
  # Cleanup
  ############################################################
  simulator.stop()
  logs(run_args, logs_dir)
  return residual_device, b_coarse_device, x_smooth_device

def generate_dynamic_layout(run_args, pe_cols_=1, pe_rows_=1, M_=4, N_=4, R_M_=2, R_N_=4):
    print("Precompile disabled, compiling based on problem size.")
    cslc, layout_file, arch = "cslc", "./src/layout_amg.csl", "wse2"
    pe_cols, pe_rows, M, N, R_M, R_N = pe_cols_, pe_rows_, M_, N_, R_M_, R_N_
    channels, width_west_buf, width_east_buf = 1, 0, 0
    fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y = calculate_fabric_dimensions(pe_cols, pe_rows, width_west_buf, width_east_buf)
    elf_folder = run_args.elffolder
    extra_args = "--max-inlined-iterations=1000000"
    
    layout_arguments = CompileCoreArgs(
      cslc, layout_file, arch,
      fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y,  # fabric details
      pe_cols, pe_rows, M, N, R_M, R_N,
      channels, width_west_buf, width_east_buf,
      elf_folder, extra_args
    )
    compile_csl_core(layout_arguments)
    if (run_args.compile_only):
      print("Compilation complete, exiting.")
      exit(0)
    else:
      return layout_arguments


def main():
  random.seed(127)

  print("############################################################")
  print("# INPUT DATA")
  print("############################################################")
  M = 5
  N = 5
  
  # data
  A0 = np.arange(M*N, dtype=np.float32).reshape(M, N)  # 2D
  x0 = np.full(shape=N*1, fill_value=1.0, dtype=np.float32)  # 1D
  b0 = np.zeros(shape=M*1, dtype=np.float32)
  nrm_b = np.linalg.norm(b0, 2)
  eps = 1.e-4
  relative_tol = eps * nrm_b # relative tolerance

  # print data
  print("Input problem size:", M, N)
  print("Input Matrix Shape:", A0.shape)
  print("Input b_1d Shape:", b0.shape)
  print("Input x_1d Shape:", x0.shape)
  print("Relative Tolerance:", relative_tol)
  ut.visualize_matrix(A0, title="A Matrix", filename="images/A.png")
  
  # wrap into a dictionary
  input_data = {
    "A0": A0,
    "x0": x0,
    "b0": b0,
    "tol": relative_tol
  }
  print("############################################################")
  print("# AMG SETUP")
  print("############################################################")
  print("\tPerforming AMG Setup")
  solver_callable_host = amg.scipy_direct_solver
  ml, setup_config = amg.smooth_aggregate_setup_only(input_data["A0"], x=input_data["x0"], b=input_data["b0"], 
                                                     solver=solver_callable_host, max_level=10, max_coarse=2)
  print(ml)
  print(amg.print_table_shapes(ml.levels))
  print(amg.print_table_data(ml.levels))
  V_levels = len(ml.levels)
  x_level = [None] * V_levels
  b_level = [None] * V_levels
  x_level[0] = np.copy(x0)
  b_level[0] = np.copy(b0)  
  
  v_cycle_data = {
    "ml": ml,
    "x_level": x_level,
    "b_level": b_level,
    "setup_config": setup_config
  }
  print("############################################################")
  print("# HOST CALCULATIONS")
  print("############################################################")
  b_coarse_host, x_coarse_host = host_calculations(v_cycle_data)
  print("\tb_coarse Host:", b_coarse_host.ravel())
  print("\tx_coarse Host:", x_coarse_host.ravel())
  print("############################################################")
  print("# DEVICE CALCULATIONS")
  print("############################################################")
  residual_device, b_coarse_device, x_smooth_device = device_calculations(v_cycle_data)
  print("Residual Device:", residual_device.ravel())
  print("b_coarse Device:", b_coarse_device.ravel())
  print("x_smooth Device:", x_smooth_device.ravel())
  print("############################################################")
  print("# COMPARISON")
  print("############################################################")
  # assert np.allclose(residual_host, residual_device, atol=1e-6), "Residual do not match!"
  assert np.allclose(b_coarse_host, b_coarse_device, atol=1e-6), "b_coarse do not match!"
  print("Results Match!")




if __name__ == "__main__":
  main()
