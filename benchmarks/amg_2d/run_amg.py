

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
    args: argparse.Namespace
    dirname: str
    cslc: str
    pe_rows: int
    pe_cols: int
    M: int
    N: int
    R_M: int
    R_N: int
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

def parse_and_get_arguments(layout_file):
  args, dirname = parse_args()
  # Get params from compile metadata
  with open(f"{dirname}/out.json", encoding='utf-8') as json_file:
      compile_data = json.load(json_file)
  pe_rows = int(compile_data['params']['pe_rows'])
  pe_cols = int(compile_data['params']['pe_cols'])
  M = int(compile_data['params']['M'])
  N = int(compile_data['params']['N'])
  R_M = int(compile_data['params']['R_M'])
  R_N = int(compile_data['params']['R_N'])
  # do check that M should be equal to N, to be square matrix
  assert M == N, "M should be equal to N, to be square matrix"
  code_csl = layout_file
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
      args, dirname, cslc, pe_cols, pe_rows, M, N, R_M, R_N, code_csl, dirname,
      fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y,  # fabric details
      args.run_only, args.arch, channels,
      width_west_buf, width_east_buf
  )
  return layout_arguments

def logs(input_data, layout_arguments):
  args = layout_arguments.args
  dirname = layout_arguments.dirname

  if args.cmaddr is None:
    # move simulation log and core dump to the given folder
    dst_log = Path(f"{dirname}/sim.log")
    src_log = Path("sim.log")
    if src_log.exists():
      shutil.move(src_log, dst_log)

    dst_trace = Path(f"{dirname}/simfab_traces")
    src_trace = Path("simfab_traces")
    if dst_trace.exists():
      shutil.rmtree(dst_trace)
    if src_trace.exists():
      shutil.move(src_trace, dst_trace)

def host_calculations(input_data, layout_arguments):
  A = input_data["A"]
  x = input_data["x"]
  b = input_data["b"]
  R = input_data["R"]

  solver_callable_host = amg.scipy_direct_solver
  # convert A to CSR
  x_level = amg.AMG_only_solve(A, b0=b, x0=x, tol=1e-6, max_ite=1, max_levels=10, max_coarse=2, solver=solver_callable_host)
  # Compute the residual
  residual_host = b - A @ x
  print("residual level host:", residual_host)
  b_coarse_host = R @ residual_host
  return residual_host, b_coarse_host

def device_calculations(input_data, layout_arguments):
  ############################################################
  # unpack data
  ############################################################
  # unpack the input data
  A = input_data["A"]
  R = input_data["R"]
  x = input_data["x"]
  b = input_data["b"]
  # unpack the layout arguments
  args = layout_arguments.args
  dirname = layout_arguments.dirname
  pe_rows = layout_arguments.pe_rows
  pe_cols = layout_arguments.pe_cols
  M = np.int32(layout_arguments.M)
  N = np.int32(layout_arguments.N)
  R_M = np.int32(layout_arguments.R_M)
  R_N = np.int32(layout_arguments.R_N)

  # 1. amg setup
  # 2. all data in levels
  # for all levels:
  #   3. x,y to allocate PEs= smart_analysis(of_level)(dynamic technique)
  #   4. allocate layout PE(x,y)
  #   5. copy data to PEs
  #   6. run computation/solve phase
  #   7. copy data back to host


  ############################################################
  # Setup simulator
  ############################################################
  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)

  # data accessible from host and device.
  symbol_residual = simulator.get_id("residual")
  symbol_b_coarse = simulator.get_id("b_coarse")
  symbol_A = simulator.get_id("A")
  symbol_x = simulator.get_id("x")
  symbol_b = simulator.get_id("b")
  symbol_R = simulator.get_id("R")

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
  simulator.launch('compute', nonblock=False)

  ############################################################
  # D2H
  ############################################################
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
  logs(input_data, layout_arguments)
  return residual_device, b_coarse_device


def main():
  random.seed(127)
  ############################################################
  # Parse arguments
  ############################################################
  layout_arguments = parse_and_get_arguments(layout_file="./src/layout_amg.csl")
  print(layout_arguments)

  ############################################################
  # Input data
  ############################################################
  M = np.int32(layout_arguments.M)
  N = np.int32(layout_arguments.N) # make it explicit 32 bit
  R_M = np.int32(layout_arguments.R_M)
  R_N = np.int32(layout_arguments.R_N)

  A = np.arange(M*N, dtype=np.float32).reshape(M, N)  # 2D
  R = np.full(shape=R_M*R_N, fill_value=1.0, dtype=np.float32).reshape(R_M, R_N)  #2D
  x = np.full(shape=N*1, fill_value=1.0, dtype=np.float32)  # 1D
  b = np.zeros(shape=M*1, dtype=np.float32)
  

  print("Input Matrix Shape:", A.shape)
  print("Input R Matrix Shape:", R.shape)
  print("Input b_1d Shape:", b.shape)
  print("Input x_1d Shape:", x.shape)
  ut.visualize_matrix_or_vector(A, title="A Matrix", filename="images/A.png")
  ut.visualize_matrix_or_vector(R, title="R Matrix", filename="images/R.png")
  ut.visualize_matrix_or_vector(x, title="x Vector", filename="images/x.png")
  ut.visualize_matrix_or_vector(b, title="b Vector", filename="images/b.png")

  input_data = {
    "A": A,
    "R": R,
    "x": x,
    "b": b,
  }

  ############################################################
  # HOST CALCULATIONS
  ############################################################
  residual_host, b_coarse_host = host_calculations(input_data, layout_arguments)
  print("Residual Host:", residual_host)
  print("b_coarse Host:", b_coarse_host)
  ############################################################
  # DEVICE CALCULATIONS
  ############################################################
  residual_device, b_coarse_device = device_calculations(input_data, layout_arguments)
  print("Residual Device:", residual_device)
  print("b_coarse Device:", b_coarse_device)
  ############################################################
  # COMPARISON
  ############################################################
  assert np.allclose(residual_host, residual_device, atol=1e-6), "Residual do not match!"
  assert np.allclose(b_coarse_host, b_coarse_device, atol=1e-6), "b_coarse do not match!"
  print("Results Match!")




if __name__ == "__main__":
  main()
