

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
import copy

from mapper_layouts.different_layouts import *

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
    args.append("--verbose")
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

  ############################################################
  # Perform V-Cycle
  ############################################################
  residual_host = []
  
  #### Reference testing
  # TODO: REMOVE THIS AS EXPENSIVE
  b_level_copy = copy.deepcopy(b_level)
  x_level_copy = copy.deepcopy(x_level)
  ml_copy = copy.deepcopy(ml)
  setup_config_copy = copy.deepcopy(setup_config)
  solver_callable_host = amg.scipy_direct_solver
  ref_b_coarsest, ref_x_coarsest, ref_A_coarsest = amg.AMG_test(ml_copy, setup_config_copy, x_level_copy, b_level_copy, max_level=10, max_coarse=2, max_ite=1, solver=solver_callable_host)
    
  
  ## 
  if len(ml.levels) == 0:
    raise RuntimeError("No levels in the multilevel hierarchy")
    exit(1)

  # V down
  for i, level in enumerate(ml.levels[:-1]):
    level = ml.levels[i]  # current level
    b_coarse_layer, x_coarse_layer = amg.each_layer_solver(level, x_level, b_level, ml, setup_config, i, residual_host)
    b_level[i + 1] = b_coarse_layer  # Directly store in the next level
    x_level[i + 1] = x_coarse_layer  # Directly store in the next level

  # V coarse
  b_coarsest = b_level[-1]
  x_coarsest = x_level[-1]
  A_coarsest = ml.levels[-1].A
  amg.debugprint(ml.levels, b_level, x_level)
  
  
  # check ref* and x_coarsest
  assert np.allclose(ref_b_coarsest, b_coarsest, rtol=1e-5), "ref_b_coarsest do not match!"
  assert np.allclose(ref_x_coarsest, x_coarsest, rtol=1e-5), "ref_x_coarsest do not match!"
  assert np.allclose(A_coarsest.toarray(), ref_A_coarsest.toarray(), rtol=1e-6), "A_coarsest do not match!"
  print("Success: b_coarsest, x_coarsest, A_coarsest match with reference")
  
  return b_coarsest, x_coarsest, residual_host

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
  residual_device_log = []

  layer_coordinates_map = s_and_p_500      
  layer_param_map, layer_coordinates_map = generate_dynamic_layout(run_args, ml, layer_coordinates_map, filename="images/s_and_p_500.png")

  ############################################################
  # Setup simulator
  ############################################################
  simulator = SdkRuntime(run_args.elffolder, cmaddr=run_args.cmaddr)
  simulator.load()
  simulator.run()
  
  ############################################################
  # VARIABLES
  ############################################################
  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  # data accessible from host and device.
  symbol_residual = simulator.get_id("residual")
  symbol_b_coarse = simulator.get_id("b_coarse")
  symbol_A = simulator.get_id("A")
  symbol_x = simulator.get_id("x")
  symbol_b = simulator.get_id("b")
  symbol_R = simulator.get_id("R")

  level_index = 0
  while(level_index < len(ml.levels) - 1):  # not the coarsest level
    print(f"Solving Layer {level_index}:")
    
    ############################################################
    # GET LAYER DATA
    ############################################################
    A, P, R, x, b = ml.levels[level_index].A.toarray(), ml.levels[level_index].P.toarray(), ml.levels[level_index].R.toarray(), x_level[level_index], b_level[level_index]
    [M,N] = A.shape
    [R_M, R_N] = R.shape
    pe_cols = 1
    pe_rows = 1
    px, py, w, h = layer_coordinates_map[level_index]['layer_start_x'], layer_coordinates_map[level_index]['layer_start_y'], layer_coordinates_map[level_index]['layer_pe_cols'], layer_coordinates_map[level_index]['layer_pe_rows']
    
    print(f"data before passing to layer {level_index}:")
    print(f"A: {A.shape}")
    print(f"R: {R.shape}")
    print(f"x: {x.shape}")
    print(f"b: {b.shape}")
    print(f"px: {px}, py: {py}, w: {w}, h: {h}")
    
    ############################################################
    # H2D
    print(f"\t1. Copying data to device")
    ############################################################
    simulator.memcpy_h2d(symbol_x, x, px, py, w, h, N*1, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
    simulator.memcpy_h2d(symbol_A, A.flatten(order='C'), px, py, w, h, M*N, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
    simulator.memcpy_h2d(symbol_R, R.flatten(order='C'), px, py, w, h, R_M*R_N, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
    simulator.memcpy_h2d(symbol_b, b, px, py, w, h, M*1, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
    ############################################################
    # Kernel
    ############################################################
    omega=amg.get_omega_from_presmoother(setup_config)
    iterations=amg.get_iterations_from_presmoother(setup_config)    
    print(f"\t2. Solving Layer {level_index} with omega={omega}, smoothing_iterations={iterations}")
    
    simulator.launch('compute', np.float32(omega), np.int16(iterations), nonblock=False)

    ############################################################
    # D2H
    print(f"\t3. Copying results to host")
    ############################################################
    residual_device = np.zeros([M*1], dtype=np.float32)
    simulator.memcpy_d2h(residual_device, symbol_residual, px, py, w, h, M*1, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
    b_coarse_device = np.zeros([R_M*1], dtype=np.float32)
    simulator.memcpy_d2h(b_coarse_device, symbol_b_coarse, px, py, w, h, R_M*1, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
    x_coarse_device = np.zeros_like(b_coarse_device)
    # copy x from device to host
    x_temp = np.zeros([N*1], dtype=np.float32)
    simulator.memcpy_d2h(x_temp, symbol_x, px, py, w, h, N*1, streaming=False,
      order=MemcpyOrder.ROW_MAJOR, data_type=memcpy_dtype, nonblock=False)
    print(f"level {level_index} x_smooth: ", x_temp)
    x_level[level_index] = x_temp
    
    
    ############################################################
    # UPDATE DATA
    ############################################################
    b_level[level_index + 1] = b_coarse_device  # Directly store in the next level
    x_level[level_index + 1] = x_coarse_device
    ############################################################
    # some additional logging
    ############################################################
    print(f"\t4. Calculating residual")
    rho = np.dot(residual_device, residual_device)
    residual_device_log.append(rho)
    
    logs(run_args, logs_dir)  
    level_index += 1  #loop induction var
      
  ############################################################
  # Cleanup
  ############################################################
  simulator.stop()
 
         
  b_coarsest_device = b_level[-1]
  x_coarsest_device = x_level[-1]
  A_coarsest_device = ml.levels[-1].A.toarray()
  amg.debugprint(ml.levels, b_level, x_level)
  return b_coarsest_device, x_coarsest_device, residual_device_log

def find_total_pes_used(layer_coordinates_map):
    """
    Selects the best total PE column and row calculation method 
    with the simplest logic.
    """

    # Compute max and sum of layer PE counts
    sum_pe_cols = sum(layer['layer_pe_cols'] for layer in layer_coordinates_map.values())
    sum_pe_rows = sum(layer['layer_pe_rows'] for layer in layer_coordinates_map.values())
    max_pe_cols = max(layer['layer_start_x'] + layer['layer_pe_cols'] for layer in layer_coordinates_map.values())  
    max_pe_rows = max(layer['layer_start_y'] + layer['layer_pe_rows'] for layer in layer_coordinates_map.values())  

    if abs(sum_pe_cols - max_pe_cols) < abs(sum_pe_rows - max_pe_rows):
        return sum_pe_cols, max_pe_rows  # sum, max (Row-wise stacking)
    elif abs(sum_pe_rows - max_pe_rows) < abs(sum_pe_cols - max_pe_cols):
        return max_pe_cols, sum_pe_rows  # max, sum (Column-wise stacking)
    else:
        return sum_pe_cols, sum_pe_rows  # Safe fallback
      
def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Command output:", result.stdout)
        print("Layout compiled successfully.")
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)

def generate_layout_compile_command(layer_param_map, total_pe_rows, total_pe_cols, total_levels, generated_layout_file, run_args):
    # Mostly doesn't change
    cslc, layout_file, arch = "cslc", generated_layout_file , "wse2"
    extra_args = "--max-inlined-iterations=1000000"
    elf_folder = run_args.elffolder
    # Can Change based on problem size.
    channels, width_west_buf, width_east_buf = 1, 0, 0
    fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y = calculate_fabric_dimensions(total_pe_cols, total_pe_rows, width_west_buf, width_east_buf)

    base_command = f"cslc {layout_file} --arch={arch} --fabric-dims={fabric_width},{fabric_height} --fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y} \\\n"
    # Extract global parameters
    global_params = f"--params=total_pe_rows:{total_pe_rows},total_pe_cols:{total_pe_cols},total_levels:{total_levels} \\\n"

    # Extract per-layer parameters from the map
    layer_params_list = []
    for key, params in layer_param_map.items():
        layer_data_shapes = params['layer_data_shapes']
        layer_coordinates = params['layer_coordinates']
        layer_params = f"--params=layer_M_{key}:{layer_data_shapes['layer_M']},layer_N_{key}:{layer_data_shapes['layer_N']},layer_R_M_{key}:{layer_data_shapes['layer_R_M']},layer_R_N_{key}:{layer_data_shapes['layer_R_N']},"
        layer_params += f"layer_start_x_{key}:{layer_coordinates['layer_start_x']},layer_start_y_{key}:{layer_coordinates['layer_start_y']},layer_pe_cols_{key}:{layer_coordinates['layer_pe_cols']},layer_pe_rows_{key}:{layer_coordinates['layer_pe_rows']},"
        layer_params += f"layer_index_{key}:{params['layer_index']}"
        layer_params += f" \\\n"
        layer_params_list.append(layer_params)

    # Combine all layer parameters
    all_layer_params = " ".join(layer_params_list)

    # Other fixed parameters
    fixed_params = f"--memcpy --channels={channels} --width-west-buf={width_west_buf} --width-east-buf={width_east_buf} {extra_args} -o {elf_folder}"

    # Construct the final command
    final_command = f"{base_command} " \
                    f"{global_params} " \
                    f"{all_layer_params} " \
                    f"{fixed_params}"
                    
    # create a data structure to store fabric details and return it.
    fabric_dimensions = [fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y]
    return final_command, fabric_dimensions
 
def create_layer_param_map(ml, layer_coordinates_input=None):
  # create a dictionary to store the metadata of each layer
  layer_params_map = {}
  layer_coordinates_map = {}  # much smaller map.(User defined or auto-generated)
  layer_data_shapes_map = {}
  total_levels = len(ml.levels) - 1
  
  # default mapping.
  if layer_coordinates_input is None:
    print("Layer coordinates not provided, using default linear placement across columns.")
    layer_coordinates_input = {
      i: {
        "layer_start_x": i,  # default linear placement to right/px
        "layer_start_y": 0,
        "layer_pe_cols": 1,  # spread across columns
        "layer_pe_rows": 1,
        } for i in range(total_levels)
    }
    
  for i in layer_coordinates_input.keys():
    layer_coordinates_map[i] = layer_coordinates_input[i]
    # if i not in range(total_levels):
    #   raise ValueError(f"It looks like layer {i} provided is not in input problem layers(most likely more coordinates than AMG Levels size.)")

  # error checks
  for i in range(len(layer_coordinates_input)):
    if layer_coordinates_input[i]['layer_start_x'] < 0:
      raise ValueError(f"Layer {i} should have positive x coordinate.")
    if layer_coordinates_input[i]['layer_start_y'] < 0:
      raise ValueError(f"Layer {i} should have positive y coordinate.")
    # update the map.
    

  # AMG problem specific layers.
  for level_index in range(total_levels):  # not the coarsest level
    A, R = ml.levels[level_index].A.toarray(), ml.levels[level_index].R.toarray()
    [M,N] = A.shape
    [R_M, R_N] = R.shape
    
    layer_data_shapes = {
      "layer_M": M,
      "layer_N": N,
      "layer_R_M": R_M,
      "layer_R_N": R_N,
    }
    layer_params = {  # map         # Layer specific data.
        # problem data.
        "layer_data_shapes": layer_data_shapes,  # map of shapes of data
        # layer coordinates
        "layer_coordinates": layer_coordinates_map[level_index],  # map of coordinates
        # layer kernel to run(e.g. L1_kernel.csl, L2_kernel.csl, ...)
        #"layer_kernel": "kernel_amg.csl", # kernel to run on this layer
        "layer_index": level_index,  # layer index
    }
    layer_params_map[level_index] = layer_params
    layer_data_shapes_map[level_index] = layer_data_shapes
    
  return layer_params_map, layer_coordinates_map

def generate_dynamic_layout(run_args, ml, layer_coordinates_map:Optional[dict]=None, filename="images/layer_mapping.png"):
  # This is a function to generate dynamic layout based on the problem size.
  
  # global data regardless of layers.
  tot_level_minus_one = len(ml.levels) - 1
  
  # This is problem specific.
  layer_param_map, layer_coordinates_map = create_layer_param_map(ml, layer_coordinates_map)
  total_pe_cols, total_pe_rows = find_total_pes_used(layer_coordinates_map)
  
  # generated_layout_file = ut.generate_layout_file_from_template(layer_param_map, run_args, total_pe_cols, total_pe_rows, tot_level_minus_one)
  generated_layout_file = "./src/layout_amg.csl"
  
  print("Precompile disabled, compiling based on problem size.")
  layout_command, fabric_dimensions = generate_layout_compile_command(layer_param_map, total_pe_rows, total_pe_cols, tot_level_minus_one, generated_layout_file, run_args)
  ut.visualize_layout_with_empty(fabric_dimensions, layer_coordinates_map, layer_param_map, total_pe_cols, total_pe_rows, filename)
  print("############################################################")
  print("Generating blueprint layout with :\n")
  print(layout_command)
  print("############################################################")
  run_command(layout_command)
    
  if (run_args.compile_only):
    print("Compilation complete, check the layout. exiting.")
    exit(0)
  else:
    return layer_param_map, layer_coordinates_map


def main():
  random.seed(127)

  print("############################################################")
  print("# INPUT DATA")
  print("############################################################")
  M = 50
  N = 50
  
  # data
  A0 = np.arange(M*N, dtype=np.float32).reshape(M, N)  # 2D
  x0 = np.full(shape=N*1, fill_value=1.0, dtype=np.float32)  # 1D
  # b0 = np.zeros(shape=M*1, dtype=np.float32)
  b0 = np.full(shape=M*1, fill_value=3.0,dtype=np.float32)
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
  
  V_levels = len(ml.levels)
  x_level = [None] * V_levels
  b_level = [None] * V_levels
  x_level[0] = np.copy(x0)
  b_level[0] = np.copy(b0)  
  for i in range(V_levels - 1): # We know this
    x_level[i+1] = np.zeros((ml.levels[i].R.shape[0], 1), dtype=np.float32)
    
  v_cycle_data = {
    "ml": ml,
    "x_level": x_level,
    "b_level": b_level,
    "setup_config": setup_config
  }
  v_cycle_data_host = copy.deepcopy(v_cycle_data)
  v_cycle_data_device = copy.deepcopy(v_cycle_data) # This is expensive
  print("############################################################")
  print("# HOST CALCULATIONS")
  print("############################################################")
  b_coarsest_host, x_coarsest_host, residual_host = host_calculations(v_cycle_data_host)
  print("HOST CALCULATIONS DONE")
  print("\tb_coarsest Host:", b_coarsest_host.ravel())
  print("\tx_coarsest Host:", x_coarsest_host.ravel())
  for i, residual in enumerate(residual_host):
      print(f"\tResidual Host [{i}]:", residual)
  print("############################################################")
  print("# DEVICE CALCULATIONS")
  print("############################################################")
  b_coarsest_device, x_coarsest_device, residual_device = device_calculations(v_cycle_data_device)
  print("\nDEVICE CALCULATIONS DONE")
  print("\tb_coarsest Device:", b_coarsest_device.ravel())
  print("\tx_coarsest Device:", x_coarsest_device.ravel()) 
  for i, residual in enumerate(residual_device):
      print(f"\tResidual Device [{i}]:", residual)   
  print("############################################################")
  print("# COMPARISON")
  print("############################################################")
  # assert np.allclose(residual_host, residual_device, atol=1e-6), "Residual do not match!"
  assert np.allclose(b_coarsest_host, b_coarsest_device, atol=1e-6), "b_coarsest of host and device do not match!"
  print("Results Match!")




if __name__ == "__main__":
  main()
