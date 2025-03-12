import os
import argparse
import time
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
import warnings

from mapper_layouts.different_layouts import *
import time_utils as time_ut
 
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

def time_logs(h, w, time_memcpy_hwl, time_ref_hwl, is_downward, level_index, iteration, timing_map_all_iterations):
    # Get timing data for this layer
    timing_data_per_layer = time_ut.timing_analysis_2d(h, w, time_memcpy_hwl, time_ref_hwl)
    
    # Store timing data based on direction and level
    direction = "down" if is_downward else "up"
    
    # Initialize iteration and direction if not exists
    if iteration not in timing_map_all_iterations:
        timing_map_all_iterations[iteration] = {}
    if direction not in timing_map_all_iterations[iteration]:
        timing_map_all_iterations[iteration][direction] = {}
        
    # Update timing data for this level
    timing_map_all_iterations[iteration][direction][level_index] = timing_data_per_layer

  # 1. memory usage per pe.
  # Calculate memory usage per PE based on array sizes

# The shapes or the data stored on PE might change, this is just a rough estimate.
def find_max_memory_usage(layer_param_map, layer_coordinates_map):
    print("\nMemory allocation(max static) per PE:")
    for level_index in layer_param_map:
        shapes = layer_param_map[level_index]["layer_data_shapes"]
        M = shapes["layer_M"]
        N = shapes["layer_N"] 
        R_M = shapes["layer_R_M"]
        R_N = shapes["layer_R_N"]
        
        mem_A = M*N*4  # A matrix (float32 = 4 bytes)
        mem_R = R_M*R_N*4  # R matrix
        mem_P = R_N*R_M*4  # P matrix
        mem_x = N*4  # x vector
        mem_b = M*4  # b vector
        mem_residual = M*4  # residual vector
        mem_b_coarse = R_M*4  # b_coarse vector
        mem_x_coarse = R_M*4  # x_coarse vector
        
        total_mem_per_pe = mem_A + mem_R + mem_P + mem_x + mem_b + mem_residual + mem_b_coarse + mem_x_coarse
        # Used hypersparse memory usage example, 48KB total per WSE-2, 2KB maybe for instructions.
        assert total_mem_per_pe < 46*1024, "exceed maximum memory capacity (46KB), increase the core rectangle"
        print(f"Level {level_index}:")
        print(f"  Per PE max memory: {total_mem_per_pe/1024:.2f} KB")
        # 2. memory usage per layer.
        coords = layer_coordinates_map[level_index]
        num_pes = coords["layer_pe_cols"] * coords["layer_pe_rows"]
        print(f"  Total memory(wxh): {total_mem_per_pe*num_pes/1024:.2f} KB({coords['layer_pe_cols']}x{coords['layer_pe_rows']})")
    
            
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
    assert channels <= 16, "only support up to 16 I/O channels"
    assert channels >= 1, "number of I/O channels must be at least 1"    
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
  print("############################################################")
  find_max_memory_usage(layer_param_map, layer_coordinates_map)
  print("############################################################")
  if (run_args.compile_only):
    print("Compilation complete, check the layout. exiting.")
    exit(0)
  else:
    return layer_param_map, layer_coordinates_map, total_pe_cols, total_pe_rows

def checkinput(A0, b0, x0, relative_tol, max_iterations):
  A = copy.deepcopy(A0.toarray())
  b = copy.deepcopy(b0)
  x = copy.deepcopy(x0)
  # Compute the rank and condition number of A
  rank_A = np.linalg.matrix_rank(A)
  cond_A = np.linalg.cond(A)

  print("Rank of A:", rank_A)
  print("Condition Number of A:", cond_A)

  # Solve using spsolve on a CSR version of A
  A_csr = sp.csr_matrix(A)
  x_spsolve = spla.spsolve(A_csr, b)
  residual = np.linalg.norm(A @ x_spsolve - b)
  

  # Separate iteration counters
  pyamg_iter = 0
  pyamg_sa_iter = 0
  cg_iter = 0
  def iteration_callback_pyamg(xk):
      nonlocal pyamg_iter
      pyamg_iter += 1
  
  def iteration_callback_pyamg_sa(xk):
      nonlocal pyamg_sa_iter
      pyamg_sa_iter += 1

  def iteration_callback_cg(xk):
      nonlocal cg_iter
      cg_iter += 1
      
  # setup and solve using pyamg.
  ml_pyamg = pyamg.ruge_stuben_solver(A_csr)
  xpyamg = ml_pyamg.solve(b, tol=relative_tol, maxiter=max_iterations, callback=iteration_callback_pyamg)
  residualpyamg = np.linalg.norm(A @ xpyamg - b)

  
  iteration_count = 0
  # solve using smoothed_aggregation_solver from pyamg
  ml_pyamg_SA = pyamg.smoothed_aggregation_solver(A_csr)
  xpyamg_SA = ml_pyamg_SA.solve(b, tol=relative_tol, maxiter=max_iterations, callback=iteration_callback_pyamg_sa)
  residualpyamg_SA = np.linalg.norm(A @ xpyamg_SA - b)
  
  
  # custom smoothed_aggregate_solver with config from amg.py
  solver_callable_host = amg.scipy_direct_solver
  ml_custom, setup_config = amg.smooth_aggregate_setup_only(A_csr, x, b, solver=solver_callable_host, max_level=10, max_coarse=2)
  x_custom = ml_custom.solve(b, x0=x, tol=relative_tol, maxiter=max_iterations)
  residualcustom = np.linalg.norm(A @ x_custom - b)
  
  # custom old amg solver
  x_otheramg, _ = amg.AMG_only_solve(A_csr, b0=b, x0=x, tol=relative_tol, max_ite=max_iterations, max_levels=10, max_coarse=2, solver=solver_callable_host)      # split into 3 phases
  residual_otheramg = np.linalg.norm(A @ x_otheramg - b)
  
  # do a cg solver from scipy spla.
  iteration_count = 0
  x_cg, info = spla.cg(A_csr, b, x0=x, tol=relative_tol, maxiter=max_iterations, callback=iteration_callback_cg)
  residual_cg = np.linalg.norm(A @ x_cg - b)

  # **Print Iteration Counts & Residuals**
  print("Residual SPSolve||Ax - b||:", residual)
  print(f"Residual pyamg RS ||Ax - b||: {residualpyamg} , iterations {pyamg_iter}")
  print(f"Residual pyamg_SA setup-base=solve-base ||Ax - b||: {residualpyamg_SA} , iterations {pyamg_sa_iter}")
  print(f"Residual setup-custom=solve-base||Ax - b||: {residualcustom}")
  print(f"Residual setup-custom=solve-custom ||Ax - b||: {residual_otheramg}")
  print(f"Residual scipy.cg ||Ax - b||: {residual_cg}, Iterations: {cg_iter}")

def generate_input1(M, N):
    # A0 = np.arange(M*N, dtype=np.float32).reshape(M, N)  # 2D
    A0 = np.random.rand(M, N).astype(np.float32)  # Use random values to ensure positive definiteness  
    A0 = np.dot(A0.T, A0) + 1e-6 * np.eye(A0.shape[1])
  # ill condition number
    lambda_reg = 1000  # Adjust this value as needed
    A0 = A0 + lambda_reg * np.eye(A0.shape[0], dtype=A0.dtype)  
    A0 = A0.astype(np.float32)
  
    x0 = np.full(shape=N*1, fill_value=1.0, dtype=np.float32)  # 1D
  # b0 = np.zeros(shape=M*1, dtype=np.float32)
    b0 = np.full(shape=M*1, fill_value=3.0,dtype=np.float32)
    return A0,x0,b0

def generate_input2(M, N):
    A = pyamg.gallery.poisson((M, N), dtype=np.float32, format='csr')  # 2D
    b = np.ones((A.shape[0]))                      # RHS
    x = np.zeros((A.shape[1]))                      # initial guess
    return A, x, b

def host_calculations(v_cycle_data):

  # unpack v_cycle_data
  ml = v_cycle_data["ml"]
  setup_config = v_cycle_data["setup_config"]
  x_level = v_cycle_data["x_level"]
  b_level = v_cycle_data["b_level"]
  tol = v_cycle_data["tol"]
  max_iterations = v_cycle_data["max_iterations"]
  ############################################################
  # Perform V-Cycle
  ############################################################
  solver_callable_host = amg.scipy_direct_solver
  
  if len(ml.levels) == 0:
    raise RuntimeError("No levels in the multilevel hierarchy")
    exit(1)
  # Iteration parameters
  
  for iteration in range(max_iterations):
    # V down
    for i, level in enumerate(ml.levels[:-1]):
      level = ml.levels[i]  # current level
      b_coarse_layer, x_coarse_layer = amg.each_layer_solver_down(level, x_level, b_level, ml, setup_config, i)
      b_level[i + 1] = b_coarse_layer  # Directly store in the next level
      x_level[i + 1] = x_coarse_layer  # Directly store in the next level

    # V coarse
    print("Solving coarse on host")
    b_coarsest = b_level[-1]
    x_coarsest = x_level[-1]
    A_coarsest = ml.levels[-1].A
    x_coarsest[:] = solver_callable_host(A_coarsest, b_coarsest)
    
    for i in reversed(range(len(ml.levels) - 1)):    
      level = ml.levels[i]  # current level
      x_lower_level = x_level[i+1]  # this is data not array
      x_current_updated = amg.each_layer_solver_up(level, x_level, b_level, ml, setup_config, i, x_lower_level)
      x_level[i] = x_current_updated

    # Check convergence
    residual = b_level[0] - ml.levels[0].A @ x_level[0]
    residual_norm = np.linalg.norm(residual)
    if residual_norm <= tol:
      print(f"Converged at iteration {iteration} with residual norm {residual_norm}")
      break
    

  b_solution, x_solution = b_level[0], x_level[0]
  print("After final solver:")
  amg.debugprint(ml.levels, b_level, x_level)
  residual_host= np.linalg.norm(ml.levels[0].A @ x_solution - b_solution)
  return b_solution, x_solution, residual_host

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
    max_iterations = v_cycle_data["max_iterations"]
    tol = v_cycle_data["tol"]

    ############################################################
    # CREATE DYNAMIC LAYOUT
    ############################################################
    layer_coordinates_map = amg_2_layers
    layer_param_map, layer_coordinates_map, total_pe_cols, total_pe_rows = generate_dynamic_layout(run_args, ml, layer_coordinates_map, filename="images/amg_2_layers.png")

    ############################################################
    # Setup simulator
    ############################################################
    simulator = SdkRuntime(run_args.elffolder, cmaddr=run_args.cmaddr)
    start = time.time()
    simulator.load()
    end = time.time()
    print(f"*** Layout Load done in {end-start}s")

    simulator.run()
    
    ############################################################
    # Initialize Hardware Timing
    ############################################################
    print("Initializing hardware timing...")
    print("Step 1: Enable timer")
    simulator.launch("f_enable_timer", nonblock=False)
    
    
    ############################################################
    # VARIABLES
    ############################################################
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR
    
    # Helper function for memcpy operations
    def do_memcpy(symbol, data, px, py, w, h, size, is_h2d=True):
        if is_h2d:
            simulator.memcpy_h2d(symbol, data, px, py, w, h, size, streaming=False,
                                order=memcpy_order, data_type=memcpy_dtype, nonblock=False)
        else:
            result = np.zeros(size, dtype=np.float32)
            simulator.memcpy_d2h(result, symbol, px, py, w, h, size, streaming=False,
                                order=memcpy_order, data_type=memcpy_dtype, nonblock=False)
            return result

    # Cache commonly used symbols
    symbols = {
        'b_coarse': simulator.get_id("b_coarse"),
        'A': simulator.get_id("A"),
        'x': simulator.get_id("x"), 
        'b': simulator.get_id("b"),
        'R': simulator.get_id("R"),
        'x_coarse': simulator.get_id("x_coarse"),
        'P': simulator.get_id("P"),
        'time_buf_u16': simulator.get_id("time_buf_u16"),
        'time_ref_u16': simulator.get_id("time_ref_u16")
    }

    ############################################################
    # AMG V-CYCLE
    ############################################################
    iteration = 0
    residual = float('inf')
    timing_map_all_iterations = {}
    while iteration < max_iterations and residual > tol:
        print("\n" + "="*80)
        print(f"║ ITERATION {iteration:3d}")
        print("="*80)

        ############################################################
        # COMBINED V-CYCLE (DOWN AND UP)
        ############################################################
        one_side_levels = len(ml.levels)
        total_levels = 2 * one_side_levels - 1
        center_index = one_side_levels - 1
        
        for total_level_index in range(total_levels):
            if total_level_index == center_index:
                print("\n" + "-"*40)
                print(f"│ COARSE SOLVE AT LEVEL {one_side_levels-1}")
                print("-"*40)
                x_coarsest_device = x_level[-1]
                A_coarse = ml.levels[-1].A.toarray()
                b_coarse = b_level[-1]

                print("Matrix Dimensions:")
                print(f"  A: {A_coarse.shape}")
                print(f"  x: {x_coarsest_device.shape} | b: {b_coarse.shape}")

                solver_callable_host = amg.scipy_direct_solver
                x_coarsest_device[:] = solver_callable_host(ml.levels[-1].A, b_level[-1])
                continue

            is_downward = total_level_index < center_index
            is_upward = total_level_index > center_index
            
            if is_downward:
                level_index = total_level_index
            elif is_upward:
                level_index = total_levels - total_level_index - 1
                
            if level_index < 0 or level_index >= len(ml.levels) - 1:
                print(f"Warning: Invalid level_index {level_index}, skipping")
                continue

            phase = "DOWN" if is_downward else "UP"
            print("\n" + "-"*40)
            print(f"│ {phase} PHASE - LAYER {level_index}")
            print("-"*40)
            
            ############################################################
            # GET LAYER DATA & VERIFY DIMENSIONS
            ############################################################
            level = ml.levels[level_index]
            if is_downward:
                A, R = level.A.toarray(), level.R.toarray()
                x, b = x_level[level_index], b_level[level_index]
                M, N = A.shape
                R_M, R_N = R.shape
                
                print("Matrix Dimensions:")
                print(f"  A: {A.shape} | R: {R.shape}")
                print(f"  x: {x.shape} | b: {b.shape}")
                
            elif is_upward:
                A, P = level.A.toarray(), level.P.toarray()
                x, b = x_level[level_index], b_level[level_index]
                x_coarse = x_level[level_index + 1]
                M, N = A.shape
                R_N, R_M = P.shape
                
                print("Matrix Dimensions:")
                print(f"  A: {A.shape} | P: {P.shape}")
                print(f"  x: {x.shape} | x_coarse: {x_coarse.shape}")
            
            coords = layer_coordinates_map[level_index]
            px, py = coords['layer_start_x'], coords['layer_start_y']
            w, h = coords['layer_pe_cols'], coords['layer_pe_rows']
            
            print("\nPE Configuration:")
            print(f"  Position: ({px}, {py}) | Size: {w}x{h}")

            ############################################################
            # H2D TRANSFERS
            ############################################################
            print("\nMemory Transfers (H2D):")
            if is_downward:
                print(f"  Vector x    : {N*1} elements")
                print(f"  Matrix A    : {M*N} elements")
                print(f"  Matrix R    : {R_M*R_N} elements")
                print(f"  Vector b    : {M*1} elements")
                do_memcpy(symbols['x'], x, px, py, w, h, N*1, is_h2d=True)
                do_memcpy(symbols['A'], A.flatten(order='C'), px, py, w, h, M*N, is_h2d=True)
                do_memcpy(symbols['R'], R.flatten(order='C'), px, py, w, h, R_M*R_N, is_h2d=True)
                do_memcpy(symbols['b'], b, px, py, w, h, M*1, is_h2d=True)
            elif is_upward:
                print(f"  Matrix P    : {R_N*R_M} elements")
                print(f"  Vector x_c  : {R_M*1} elements")
                do_memcpy(symbols['P'], P.flatten(order='C'), px, py, w, h, R_N*R_M, is_h2d=True)
                do_memcpy(symbols['x_coarse'], x_coarse, px, py, w, h, R_M*1, is_h2d=True)

            ############################################################
            # COMPUTE
            ############################################################
            print("Step 2: Initial sync across PEs in each Layer")
            # simulator.launch("f_sync_layer", nonblock=False)
            print("Step 4: Record initial timestamp (tic)")
            simulator.launch("f_tic", nonblock=True)
            
            print("\nComputation:")
            if is_downward:
                omega = amg.get_omega_from_presmoother(setup_config)
                iterations = amg.get_iterations_from_presmoother(setup_config)
                print(f"Launching v_cycle_down with omega={omega}, iterations={iterations}, level_index={level_index}")
                simulator.launch("v_cycle_down", np.float32(omega), np.int16(iterations), np.int16(level_index), nonblock=False)                
            else:
                omega = amg.get_omega_from_postsmoother(setup_config)
                iterations = amg.get_iterations_from_postsmoother(setup_config)
                print(f"Launching v_cycle_up with omega={omega}, iterations={iterations}, level_index={level_index}")
                simulator.launch("v_cycle_up", np.float32(omega), np.int16(iterations), np.int16(level_index), nonblock=False)                

            print("Step 5: toc() records time_end")
            simulator.launch("f_toc", nonblock=False)
            print("Step 6: prepare (time_start, time_end)")
            simulator.launch("f_memcpy_timestamps", nonblock=False)
            
            ############################################################
            # D2H & UPDATE DATA
            ############################################################
            print("\nMemory Transfers (D2H):")
            if is_downward:
                print("  Copying b_coarse, x_smooth")
                b_coarse_device = do_memcpy(symbols['b_coarse'], None, px, py, w, h, R_M*1, is_h2d=False)
                x_smooth = do_memcpy(symbols['x'], None, px, py, w, h, N*1, is_h2d=False)
                x_coarse_device = np.zeros_like(b_coarse_device)
                
                x_level[level_index] = x_smooth
                b_level[level_index + 1] = b_coarse_device
                x_level[level_index + 1] = x_coarse_device
            elif is_upward:
                print("  Copying x")
                x_level[level_index] = do_memcpy(symbols['x'], None, px, py, w, h, N*1, is_h2d=False)

            ############################################################
            # TIME TRANSFERS
            ############################################################
               
            print("Step 7: Retrieve timing data")
            time_memcpy_hwl_1d = np.zeros(w*h*6, np.uint32)
            simulator.memcpy_d2h(time_memcpy_hwl_1d, symbols['time_buf_u16'], px, py, w, h, 6,
                streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            time_memcpy_hwl = oned_to_hwl_colmajor(h, w, 6, time_memcpy_hwl_1d, np.uint16)
            
            time_ref_1d = np.zeros(w*h*3, np.uint32)
            simulator.memcpy_d2h(time_ref_1d, symbols['time_ref_u16'], px, py, w, h, 3,
                streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            time_ref_hwl = oned_to_hwl_colmajor(h, w, 3, time_ref_1d, np.uint16)
            
            ############################################################
            # LOGGING
            ############################################################
            logs(run_args, logs_dir)
            print("\nAnalyzing timing data...")
            time_logs(h, w, time_memcpy_hwl, time_ref_hwl, is_downward, level_index, iteration, timing_map_all_iterations)
            
        ############################################################
        # CHECK CONVERGENCE
        ############################################################
        residual = np.linalg.norm(ml.levels[0].A @ x_level[0] - b_level[0])
        print("\n" + "="*40)
        print(f"ITERATION {iteration} SUMMARY")
        print("="*40)
        print(f"Residual: {residual:.6e}")
        if residual <= tol:
            print("Convergence achieved!")
        print("="*40 + "\n")
        iteration += 1

    ############################################################
    # Cleanup simulator
    ############################################################
    simulator.stop()        
    ############################################################
    # Final results
    ############################################################
    b_solution_device, x_solution_device = b_level[0], x_level[0]
    print("After final solver:")
    amg.debugprint(ml.levels, b_level, x_level)

    # Save and analyze timing data
    print("\nGenerating timing analysis...")
    df = time_ut.write_timing_data(timing_map_all_iterations, filename="reports/timing_data.csv")
    # time_ut.analyze_timing_data("reports/timing_data.csv")
    # print("\nTiming analysis complete. Open timing_report.html to view the results.")

    residual_device = np.linalg.norm(ml.levels[0].A @ x_solution_device - b_level[0])
    return b_solution_device, x_solution_device, residual_device

def main():
  random.seed(127)

  print("############################################################")
  print("# INPUT DATA")
  print("############################################################")
  N = 7
  M = 7
  eps = 1.e-2
  max_iterations = 20
    
  A0, x0, b0 = generate_input2(M, N)
  A0 = A0.astype(np.float32)
  x0 = x0.astype(np.float32)  # Do this compulsorily to make the data 32bit.
  b0 = b0.astype(np.float32)
  
  nrm_b = np.linalg.norm(b0, 2)
  relative_tol = eps * nrm_b # relative tolerance
  
  checkinput(A0, b0, x0, relative_tol, max_iterations)

  # print data
  print("Input problem size:", M, N)
  print("Input Matrix Shape:", A0.shape)
  print("Input b_1d Shape:", b0.shape)
  print("Input x_1d Shape:", x0.shape)
  print("Norm of b:", nrm_b)
  print("Relative Tolerance:", relative_tol)
  # ut.visualize_matrix(A0, title="A Matrix", filename="images/A.png")
  
  # wrap into a dictionary
  input_data = {
    "A0": A0,
    "x0": x0,
    "b0": b0,
    "tol": relative_tol,
    "max_iterations": max_iterations
  }
  print("############################################################")
  print("# AMG SETUP")
  print("############################################################")
  print("\tPerforming AMG Setup")
  solver_callable_host = amg.scipy_direct_solver
  ml, setup_config = amg.smooth_aggregate_setup_only(input_data["A0"], x=input_data["x0"], b=input_data["b0"], 
                                                     solver=solver_callable_host, max_level=10, max_coarse=2)
  # print(ml)
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
    "setup_config": setup_config,
    "tol": relative_tol,
    "max_iterations": max_iterations
  }
  v_cycle_data_host = copy.deepcopy(v_cycle_data)
  v_cycle_data_device = copy.deepcopy(v_cycle_data) # This is expensive
  print("############################################################")
  print("# HOST CALCULATIONS")
  print("############################################################")
  b_final_host, x_final_host, residual_host = host_calculations(v_cycle_data_host)
  print("HOST CALCULATIONS DONE")
  print(f"\tResidual Host ||AX-b||:", residual_host)
  # print(f"\n x_solution_final Host:", x_final_host.ravel())
  print("############################################################")
  print("# DEVICE CALCULATIONS")
  print("############################################################")
  b_final_device, x_final_device, residual_device = device_calculations(v_cycle_data_device)
  print("\nDEVICE CALCULATIONS DONE")
  print(f"\tResidual Device ||AX-b||:", residual_device)
  # print(f"\t x_solution_final Device:", x_final_device.ravel()) 
  print("############################################################")
  print("# COMPARISON")
  print("############################################################")
  # assert np.allclose(residual_host, residual_device, atol=1e-6), "Residual do not match!"
  assert np.allclose(b_final_host, b_final_device, atol=1e-6), "b_final of host and device do not match!"
  assert np.allclose(x_final_host, x_final_device, atol=1e-6), "x_final of host and device do not match!"
  print("Results Match!")


if __name__ == "__main__":
  main()
