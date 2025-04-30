import time
from typing import Optional
from pathlib import Path
import shutil
import subprocess
import random
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
from cmd_parser import parse_args
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg
import amg as amg 
import copy
from mapper_layouts.different_layouts import *
from src.libraries.single_layer.jacobi_only import pad_A, pad_1d, unpad_1d
 
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

def autogenerate_amg_layout(layer_param_map, total_pe_rows, total_pe_cols, total_levels, filename):
    # First prepare data 
    """Generate layout file from layer parameters map."""
    layers_params = []
    for key, params in layer_param_map.items():
        layer_data_shapes = params['layer_data_shapes']
        layer_coordinates = params['layer_coordinates']
        
        params = {
                'M': layer_data_shapes['layer_M'],
                'N': layer_data_shapes['layer_N'],
                'R_M': layer_data_shapes['layer_R_M'],
                'R_N': layer_data_shapes['layer_R_N'],
                'start_x': layer_coordinates['layer_start_x'],
                'start_y': layer_coordinates['layer_start_y'],
                'pe_cols': layer_coordinates['layer_pe_cols'],
                'pe_rows': layer_coordinates['layer_pe_rows'],
                'index': params['layer_index']
        }
        layers_params.append(params)
    
    # Sort layers by index to ensure correct order
    layers_params.sort(key=lambda x: x['index'])
    
    # Call generate_layout_amg with the properly structured data
    generate_layout_amg(total_pe_cols=total_pe_cols, 
                       total_pe_rows=total_pe_rows, 
                       total_levels=total_levels,
                       layers=layers_params, filename=filename)

def generate_layout_compile_command(layer_param_map, total_pe_rows, total_pe_cols, total_levels, generated_layout_file, run_args):
      # Generates below command.
      # cslc ./src/layout_amg.csl --arch=wse2 --fabric-dims=27,12 --fabric-offsets=4,1 \
      # --params=total_pe_rows:10,total_pe_cols:20,total_levels:2 \
      # --params=layer_M_0:50,layer_N_0:50,layer_R_M_0:10,layer_R_N_0:50,layer_start_x_0:0,layer_start_y_0:0,layer_pe_cols_0:10,layer_pe_rows_0:10,layer_index_0:0 \
      # --params=layer_M_1:10,layer_N_1:10,layer_R_M_1:10,layer_R_N_1:10,layer_start_x_1:10,layer_start_y_1:0,layer_pe_cols_1:10,layer_pe_rows_1:10,layer_index_1:1 \
      # --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000 -o out
      
    # Mostly doesn't change
    cslc, layout_file, arch = "cslc", generated_layout_file , "wse2"

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
    extra_args = "--max-inlined-iterations=1000000"
    elf_folder = run_args.elffolder
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
    print("Layer coordinates not provided, using default linear placement across columns(1x1) always works.")
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
  if len(layer_coordinates_input) != total_levels:
    raise ValueError(f"The number of layers provided in layer_coordinates_input should be equal to the total number of levels.")
  for i in range(len(layer_coordinates_input)):
    if layer_coordinates_input[i]['layer_start_x'] < 0:
      raise ValueError(f"Layer {i} should have positive x coordinate.")
    if layer_coordinates_input[i]['layer_start_y'] < 0:
      raise ValueError(f"Layer {i} should have positive y coordinate.")
    if layer_coordinates_input[i]['layer_pe_cols'] != layer_coordinates_input[i]['layer_pe_rows']:
      raise ValueError(f"Layer {i} should have square PE dimensions.")
    # update the map.
    

  # AMG problem specific layers.
  for level_index in range(total_levels):  # not the coarsest level
    A, R = ml.levels[level_index].A, ml.levels[level_index].R
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
  
  generated_layout_file = "./src/auto_layout_amg.csl"
  
  print("Precompile disabled, compiling based on problem size.")
  layout_command, fabric_dimensions = generate_layout_compile_command(layer_param_map, total_pe_rows, total_pe_cols, tot_level_minus_one, generated_layout_file, run_args)
  print("############################################################")
  print("Generating blueprint layout with :\n")
  print("\n1. Autogenerating layout file...")
  autogenerate_amg_layout(layer_param_map, total_pe_rows, total_pe_cols, tot_level_minus_one, generated_layout_file)
  print("\n2. Compiling layout file...")
  print(layout_command)
  run_command(layout_command)
  print("############################################################")
  find_max_memory_usage(layer_param_map, layer_coordinates_map)
  print("############################################################")
  if (run_args.compile_only):
    print("Compilation complete, check the layout. exiting.")
    exit(0)
  else:
    return layer_param_map, layer_coordinates_map, total_pe_cols, total_pe_rows

def generate_input2(M, N, type=np.float32):  
    A = pyamg.gallery.poisson((M, N), dtype=type, format='csr')  # 2D
    b = np.ones((A.shape[0]))                      # RHS
    x = np.zeros((A.shape[1]))                      # initial guess

    A0 = A.astype(type)
    x0 = x.astype(type)  # Do this compulsorily to make the data 32bit.
    b0 = b.astype(type)
    return A0, x0, b0

class simplerMemcpy:
    def __init__(self, simulator, memcpy_order, memcpy_dtype):
        self._simulator = simulator
        self._memcpy_order = memcpy_order
        self._memcpy_dtype = memcpy_dtype
      
    def do_memcpy_h2d_bcast(self, symbol, data, px, py, w, h, size, is_rowbcast=True):
        if is_rowbcast:
            self._simulator.memcpy_h2d_rowbcast(symbol, data, px, py, w, h, size, streaming=False,
                                order=self._memcpy_order, data_type=self._memcpy_dtype, nonblock=False)
        else:
            self._simulator.memcpy_h2d_colbcast(symbol, data, px, py, w, h, size, streaming=False,
                                order=self._memcpy_order, data_type=self._memcpy_dtype, nonblock=False)
        
    def do_memcpy_h2d(self, symbol, data, px, py, w, h, size):
        self._simulator.memcpy_h2d(symbol, data, px, py, w, h, size, streaming=False,
                            order=self._memcpy_order, data_type=self._memcpy_dtype, nonblock=False)

    def do_memcpy_d2h(self, result, symbol, px, py, w, h, size):
        self._simulator.memcpy_d2h(result, symbol, px, py, w, h, size, streaming=False,
                            order=self._memcpy_order, data_type=self._memcpy_dtype, nonblock=False)

def perform_coarse_solve(level, x_level, b_level, solver_callable_host):
    """Handle coarse level solve"""    
    ############################################################
    # Store original dimensions before unpadding - HACK.
    unpadded_x_shape = level.A.shape[1]
    unpadded_b_shape = level.A.shape[0]
    padded_x_shape = x_level[-1].shape[0]
    padded_b_shape = b_level[-1].shape[0]
    # Unpad x and b for coarse level calculations
    x_level[-1] = unpad_1d(x_level[-1], unpadded_x_shape)
    b_level[-1] = unpad_1d(b_level[-1], unpadded_b_shape)
    
    # Convert A to CSR format for coarse solver
    level.A = sp.csr_matrix(level.A)
    ############################################################  
      
    print("\n" + "-"*40)
    print(f"│ COARSE SOLVE")
    print("-"*40)
    
    x_coarsest = x_level[-1]
    b_coarse = b_level[-1]
    A_coarse = level.A
    
    print("Matrix Dimensions:")
    print(f"  A: {A_coarse.shape}")
    print(f"  x: {x_coarsest.shape} | b: {b_coarse.shape}")
    
    x_coarsest[:] = solver_callable_host(A_coarse, b_coarse)
    
    ############################################################
    # Pad x_coarsest to its original dimensions. HACK
    x_coarsest = pad_1d(x_coarsest, padded_x_shape, variable_name="x_coarsest")
    b_coarse = pad_1d(b_coarse, padded_b_shape, variable_name="b_coarse")
    ############################################################
    
    return x_coarsest

def copy_layer_on_device(simple_memcpy, simulator,symbols, level_index, level, coords, x_level, b_level, setup_config, iteration, deviceprofiling, hardwareTimer):
    """Handle downward pass operations for a single level"""
    print("\n" + "-"*40)
    print(f"│ DOWN PHASE - LAYER {level_index}")
    print("-"*40)
    
    # Extract coordinates
    px, py = coords['layer_start_x'], coords['layer_start_y']
    w, h = coords['layer_pe_cols'], coords['layer_pe_rows']
    
    # Get matrices and vectors
    A, R = level.A, level.R
    M, N = A.shape
    R_M, R_N = R.shape
    x = x_level[level_index]
    b = b_level[level_index]
    
    # Calculate per-PE dimensions
    per_pe_rows = M // h
    per_pe_cols = N // w
    per_pe_restrict_rows = R_M // h
    per_pe_restrict_cols = R_N // w
    
    # Setup solver parameters
    omega = np.zeros(h*w, dtype=np.float32)
    iterations = np.zeros(h*w, dtype=np.int32)
    omega[:] = amg.get_omega_from_presmoother(setup_config)
    iterations[:] = amg.get_iterations_from_presmoother(setup_config)
    
    print(f"  A: {A.shape} | R: {R.shape}")
    if (level_index == 0):
        print(f"  b: {b.shape}")
    print(f"  x: {x.shape}")
    print(f"  omega: {omega.shape} | iterations: {iterations.shape}")
    print("PE Configuration:")
    print(f"  Position: ({px}, {py}) | Size: {w}x{h}")    
    print(f"Per PE Data Sizes:")
    print(f"  x: {per_pe_rows}x{1} | b: {per_pe_cols}x{1}")
    print(f"  A: {per_pe_rows}x{per_pe_cols} | R: {per_pe_restrict_rows}x{per_pe_restrict_cols}")
    print("Step 0: Collect layer data")
                          
    ############################################################
    # TRANSFORM THE DATA TO MAP ON DEVICE.( single_layer_run.py ) 
    ############################################################
    # x is across rows and b is across cols (row major)
    # A is across rows and cols (row major)
    # R is across rows and cols (row major)
    # P is across rows and cols (row major)
    # b_next is across last col (row major)
    
    # As an example, consider A[4, 4], mapped onto a 2x2 grid of PEs:
    #
    #   Matrix A on host            2 x 2 PE grid, row major A submatrices
    # +----+----+----+----+         +----------------+----------------+
    # | 0  | 1  | 2  | 3  |         | PE (0, 0):     | PE (1, 0):     |
    # +----+----+----+----+         |  0,  1,  4,  5 |  2,  3,  6,  7 |
    # | 4  | 5  | 6  | 7  |         |                |                |
    # +----+----+----+----+   --->  +----------------+----------------+
    # | 8  | 9  | 10 | 11 |         | PE (0, 1):     | PE (1, 1):     |
    # +----+----+----+----+         |  8,  9, 12, 13 | 10, 11, 14, 15 |
    # | 12 | 13 | 14 | 15 |         |                |                |
    # +----+----+----+----+         +----------------+----------------+
    #
    # So our input array for memcpy_h2d must be ordered as follows after ROW_MAJOR copy ordering.
    # [ 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15 ]    
    # Transform data for device layout
    print("Step 1: Transform data for device layout")
    A_transformed = np.stack(np.split(np.stack(np.split(A, h, axis=1)), w, axis=1)).ravel()
    R_transformed = np.stack(np.split(np.stack(np.split(R, h, axis=1)), w, axis=1)).ravel()
    x_transformed = x.flatten(order='C')
    if (level_index == 0):
      b_transformed = b.flatten(order='C')
    

    start_time = time.time()
    print("Step 2: H2D Transfers")
    simple_memcpy.do_memcpy_h2d(symbols['A'], A_transformed, px, py, w, h, per_pe_rows*per_pe_cols)
    simple_memcpy.do_memcpy_h2d(symbols['R'], R_transformed, px, py, w, h, per_pe_restrict_rows*per_pe_restrict_cols)
    simple_memcpy.do_memcpy_h2d_bcast(symbols['x'], x_transformed, px, py, w, h, per_pe_cols*1, is_rowbcast=False)
    # Only first layer has b0
    if (level_index == 0):
        # simple_memcpy.do_memcpy_h2d_bcast(symbols['b'], b_transformed, px, py, w, h, per_pe_rows*1, is_rowbcast=True)
        simple_memcpy.do_memcpy_h2d(symbols['b'], b_transformed, px, py, 1, h, per_pe_rows*1)
    simple_memcpy.do_memcpy_h2d(symbols['omega'], omega, px, py, w, h, 1)
    simple_memcpy.do_memcpy_h2d(symbols['iterations'], iterations, px, py, w, h, 1)
    h2d_time = time.time() - start_time


def device_calculations_dataflow(v_cycle_data):
    run_args, logs_dir = parse_args()
    
    ############################################################
    # Get the layout for AMG layers given by user.
    ############################################################
    layer_coordinates_map = amg_2_layers
    ############################################################
    # TODO: When doing sparse remove this.
    # Pad the data and make it dense.
    ############################################################
    pad_and_make_dense(layer_coordinates_map, v_cycle_data)
        
    ############################################################
    # unpack data
    ############################################################
    # unpack the input data
    ml = v_cycle_data["ml"] # Changed to dense and padded.
    setup_config = v_cycle_data["setup_config"]
    x_level = v_cycle_data["x_level"] # Changed to dense and padded.
    b_level = v_cycle_data["b_level"] # Changed to dense and padded.
    max_iterations = v_cycle_data["max_iterations"]
    tol = v_cycle_data["tol"]
    
    ############################################################
    # CREATE DYNAMIC LAYOUT
    ############################################################
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
    # VARIABLES
    ############################################################    
    # Cache commonly used symbols
    symbols = {
        'A': simulator.get_id("A"),
        'R': simulator.get_id("R"),
        'P': simulator.get_id("P"),
        'x': simulator.get_id("x"), 
        'b': simulator.get_id("b"),
        'b_next': simulator.get_id("b_next"),
        'x_coarse': simulator.get_id("x_coarse"),
        'omega': simulator.get_id("omega"),
        'iterations': simulator.get_id("iterations")
    }
    iteration = 0
    residual = float('inf')
    timing_map_all_iterations = {}
    solver_callable_host = amg.scipy_direct_solver
    memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR
    simple_memcpy = simplerMemcpy(simulator, memcpy_order, memcpy_dtype)
    ############################################################
    # AMG V-CYCLE
    ############################################################    
    
    while iteration < max_iterations and residual > tol:
        print(f"\n{'='*40}\n│ ITERATION {iteration}\n{'='*40}")
        
        # Downward passes
        for level_index in range(len(ml.levels) - 1):
            copy_layer_on_device(
                simple_memcpy, simulator, symbols, level_index, ml.levels[level_index],
                layer_coordinates_map[level_index], x_level, b_level, setup_config, iteration, deviceprofiling=None, hardwareTimer=None
            )
        print("Step 4: Compute")
        simulator.launch("v_cycle_down", nonblock=False)
        amg.debugprint(ml.levels, b_level, x_level)
        
        # D2H transfers(from last layer and last PE grid.)
        print("Step 6: D2H Transfers")
        coarse_level = ml.levels[len(ml.levels)-1]
        b_coarse_shape = coarse_level.A.shape[0]
        b_coarse_device = np.zeros(total_pe_rows*b_coarse_shape, dtype=np.float32)
        simple_memcpy.do_memcpy_d2h(b_coarse_device, symbols['b_next'], total_pe_cols-1, 0, 1, total_pe_rows, b_coarse_shape*1) # copy from symbol into result.
        print(f"b_coarse_device: {b_coarse_device}")

        iteration += 1
    ############################################################
    # Cleanup simulator
    ############################################################
    simulator.stop()        
    ############################################################
    # Final results
    ############################################################
    b_solution_device, x_solution_device = b_level[0], x_level[0]

    # time_ut.analyze_timing_data("reports/timing_data.csv")
    # print("\nTiming analysis complete. Open timing_report.html to view the results.")

    residual_device = np.linalg.norm(ml.levels[0].A @ x_solution_device - b_level[0])
    return b_solution_device, x_solution_device, residual_device
  
def pad_and_make_dense(layer_coordinates_map, v_cycle_data):
    """Pad matrices and vectors to match PE dimensions and convert sparse to dense"""
    ml = v_cycle_data["ml"]
    x_level = v_cycle_data["x_level"] 
    b_level = v_cycle_data["b_level"]
    
    for level_index in range(len(ml.levels) -1):
        coords = layer_coordinates_map[level_index]
        kernel_rows = coords['layer_pe_rows']
        kernel_cols = coords['layer_pe_cols']
        
        # Pad and convert matrices to dense
        ml.levels[level_index].A = pad_A(ml.levels[level_index].A.toarray(), kernel_rows, kernel_cols, variable_name=f"A_{level_index}")
        if level_index < len(ml.levels)-1:  
            ml.levels[level_index].R = pad_A(ml.levels[level_index].R.toarray(), kernel_rows, kernel_cols, variable_name=f"R_{level_index}")
            ml.levels[level_index].P = pad_A(ml.levels[level_index].P.toarray(), kernel_rows, kernel_cols, variable_name=f"P_{level_index}")
        
        # Pad vectors
        if x_level[level_index] is not None:
            [padded_matrix_rows, padded_matrix_cols] = ml.levels[level_index].A.shape
            x_level[level_index] = pad_1d(x_level[level_index], padded_matrix_cols, variable_name=f"x_{level_index}")
        if b_level[level_index] is not None:
            [padded_matrix_rows, padded_matrix_cols] = ml.levels[level_index].A.shape
            b_level[level_index] = pad_1d(b_level[level_index], padded_matrix_rows, variable_name=f"b_{level_index}")
                

def main():
  random.seed(127)

  print("############################################################")
  print("# INPUT DATA")
  print("############################################################")
  N = 7
  M = 7
  eps = 1.e-5
  max_iterations = 1
    
  A0, x0, b0 = generate_input2(M, N, type=np.float32)
  nrm_b = np.linalg.norm(b0, 2)
  relative_tol = eps * nrm_b # relative tolerance

  # print data
  print("Input problem size:", M, N)
  print("Input Matrix Shape:", A0.shape)
  print("Input b_1d Shape:", b0.shape)
  print("Input x_1d Shape:", x0.shape)
  print("Norm of b:", nrm_b)
  print("Relative Tolerance:", relative_tol)
  
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
    x_level[i+1] = np.zeros(ml.levels[i].R.shape[0], dtype=np.float32) # Changed to 1D array
    
  v_cycle_data = {
    "ml": ml,
    "x_level": x_level,
    "b_level": b_level,
    "setup_config": setup_config,
    "tol": relative_tol,
    "max_iterations": max_iterations
  }
  v_cycle_data_device = copy.deepcopy(v_cycle_data) # This is expensive
  print("############################################################")
  print("# DEVICE CALCULATIONS")
  print("############################################################")
  b_final_device, x_final_device, residual_device = device_calculations_dataflow(v_cycle_data_device) # changed to dataflow
  print("\nDEVICE CALCULATIONS DONE")


if __name__ == "__main__":
  main()
