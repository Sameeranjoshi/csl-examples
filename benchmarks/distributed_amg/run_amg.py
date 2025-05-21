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
import utilities as ut
import amg as amg 
import copy

from mapper_layouts.different_layouts import *
import time_utils as time_ut
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

# New time logs.
def time_logs_new(h, w, hardware_timing, h2d_time, d2h_time, is_downward, level_index, iteration, filename="./reports/v_cycle_up_down.csv", deviceprofiling=None):
    time_memcpy_hwl = hardware_timing["time_memcpy_hwl"]
    time_ref_hwl = hardware_timing["time_ref_hwl"]
    
    perf_metrics = {}
    # Get timing data for this layer
    timing_data_per_layer = time_ut.time_analysis_noref(h, w, time_memcpy_hwl, time_ref_hwl)

    perf_metrics['iteration'] = iteration        
    direction = "down" if is_downward else "up"
    perf_metrics['direction'] = direction
    perf_metrics['level_index'] = level_index
    perf_metrics['PE'] = f"PE{h}_{w}"
    perf_metrics['h2d_time_seconds'] = h2d_time
    perf_metrics['d2h_time_seconds'] = d2h_time
    perf_metrics['cycles'] = timing_data_per_layer['cycles']
    perf_metrics['kernel_time_us'] = timing_data_per_layer['kernel_time_us']
    df = time_ut.write_performance_data(perf_metrics, filename=filename)
    
    
    timing = time_ut.DeviceOperatorTiming(
        h2d_time=perf_metrics['h2d_time_seconds'],
        d2h_time=perf_metrics['d2h_time_seconds'],
        cycles=perf_metrics['cycles'],
        kernel_time_us=perf_metrics['kernel_time_us']
    )
    deviceprofiling.add_timing(level_index, timing, direction)
    return df
  
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
        coords = layer_coordinates_map[0]
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
        print("\tCommand output:", result.stdout)
        print("\tLayout compiled successfully.")
    except subprocess.CalledProcessError as e:
        print("\tError occurred:", e.stderr)

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
      #   cslc ./src/auto_layout_amg.csl --arch=wse2 --fabric-dims=11,6 --fabric-offsets=4,1 \
      #  --params=total_pe_rows:4,total_pe_cols:4,total_levels:2,layer_start_x:0,layer_start_y:0,layer_pe_cols:4,layer_pe_rows:4 \
      #  --params=layer_M_0:52,layer_N_0:52,layer_R_M_0:4,layer_R_N_0:52,layer_index_0:0 \
      #  --params=layer_M_1:4,layer_N_1:4,layer_R_M_1:4,layer_R_N_1:4,layer_index_1:1 \
      #  --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0 --max-inlined-iterations=1000000 -o out
      
      
    # Mostly doesn't change
    cslc, layout_file, arch = "cslc", generated_layout_file , "wse2"

    # Can Change based on problem size.
    channels, width_west_buf, width_east_buf = 1, 0, 0
    assert channels <= 16, "only support up to 16 I/O channels"
    assert channels >= 1, "number of I/O channels must be at least 1"    
    fabric_width, fabric_height, core_fabric_offset_x, core_fabric_offset_y = calculate_fabric_dimensions(total_pe_cols, total_pe_rows, width_west_buf, width_east_buf)

    base_command = f"cslc {layout_file} --arch={arch} --fabric-dims={fabric_width},{fabric_height} --fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y} \\\n"
    # Extract global parameters
    # get 0 index layer coordinates.
    layer_coordinates = layer_param_map[0]['layer_coordinates']
    global_params = f"--params=total_pe_rows:{total_pe_rows},total_pe_cols:{total_pe_cols},total_levels:{total_levels},layer_start_x:{layer_coordinates['layer_start_x']},layer_start_y:{layer_coordinates['layer_start_y']},layer_end_x:{layer_coordinates['layer_start_x'] + layer_coordinates['layer_pe_cols']},layer_end_y:{layer_coordinates['layer_start_y'] + layer_coordinates['layer_pe_rows']} \\\n"

    # Extract per-layer parameters from the map
    layer_params_list = []
    for key, params in layer_param_map.items():
        layer_data_shapes = params['layer_data_shapes']
        layer_coordinates = params['layer_coordinates']
        layer_params = f"--params=layer_M_{key}:{layer_data_shapes['layer_M']},layer_N_{key}:{layer_data_shapes['layer_N']},layer_R_M_{key}:{layer_data_shapes['layer_R_M']},layer_R_N_{key}:{layer_data_shapes['layer_R_N']},"
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
        "layer_coordinates": layer_coordinates_map[0],  # map of coordinates
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
  # print("############################################################")
  # find_max_memory_usage(layer_param_map, layer_coordinates_map)
  # print("############################################################")
    
  generated_layout_file = "./src/auto_layout_amg.csl"
  print("Precompile disabled, compiling based on problem size.")
  layout_command, fabric_dimensions = generate_layout_compile_command(layer_param_map, total_pe_rows, total_pe_cols, tot_level_minus_one, generated_layout_file, run_args)
  # ut.visualize_layout_with_empty_plotly(fabric_dimensions, layer_coordinates_map, layer_param_map, total_pe_cols, total_pe_rows, filename)
  print("############################################################")
  print("Generating blueprint layout with :\n")
  print("\nStep 1: Autogenerating layout file...")
  autogenerate_amg_layout(layer_param_map, total_pe_rows, total_pe_cols, tot_level_minus_one, generated_layout_file)
  print("\nStep 2: Compiling layout file...")
  print("\t" + layout_command)
  run_command(layout_command)
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

def generate_input2(M, N, type=np.float32):  
    A = pyamg.gallery.poisson((M, N), dtype=type, format='csr')  # 2D
    b = np.ones((A.shape[0]))                      # RHS
    x = np.zeros((A.shape[1]))                      # initial guess

    A0 = A.astype(type)
    x0 = x.astype(type)  # Do this compulsorily to make the data 32bit.
    b0 = b.astype(type)
    return A0, x0, b0

def host_calculations(v_cycle_data):
    hostprofiling = time_ut.HostProfiling()
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
            # Set context BEFORE the operation
            hostprofiling.add_other_info(iteration, i, "down")
            
            level = ml.levels[i]  # current level
            b_coarse_layer, x_coarse_layer, operator_timing = amg.each_layer_solver_down(level, x_level, b_level, ml, setup_config, i, hostprofiling)
            b_level[i + 1] = b_coarse_layer
            x_level[i + 1] = x_coarse_layer
        
        # V coarse
        b_coarsest = b_level[-1]
        x_coarsest = x_level[-1]
        A_coarsest = ml.levels[-1].A
        x_coarsest[:] = solver_callable_host(A_coarsest, b_coarsest)
        
        # # V up
        for i in reversed(range(len(ml.levels) - 1)):
            # Set context BEFORE the operation
            hostprofiling.add_other_info(iteration, i, "up")
            
            level = ml.levels[i]
            x_lower_level = x_level[i+1]
            x_current_updated, operator_timing = amg.each_layer_solver_up(level, x_level, b_level, ml, setup_config, i, x_lower_level, hostprofiling)
            x_level[i] = x_current_updated

        ####TODO: Change this to fine layer.
        # Check convergence
        # residual = b_level[0] - ml.levels[0].A @ x_level[0] # after Up cycle
        residual = b_level[len(ml.levels)-1] - ml.levels[len(ml.levels)-1].A @ x_level[len(ml.levels)-1] # after down.
        residual_norm = np.linalg.norm(residual)
        print(f"Residual: {residual_norm:.6e}, tol: {tol:.6e}, iteration: {iteration}")
        if residual_norm <= tol:
            print(f"Converged at iteration {iteration} with residual norm {residual_norm}")
            break
        
        
    print("After final solver:")
    amg.debugprint(ml.levels, b_level, x_level)
    # hostprofiling.print_timing_summary()
    # Checks
    b_solution, x_solution, A_solution = b_level[len(ml.levels)-1], x_level[len(ml.levels)-1], ml.levels[len(ml.levels)-1].A  # last one
    residual_host = b_solution - A_solution @ x_solution
    residual_host_norm = np.linalg.norm(residual_host)
    return b_solution, x_solution, residual_host_norm

class HardwareTimerManager:
  # keep logs of time_memcpy_hwl, time_ref_hwl, h2d_time, d2h_time inside this class.
  
    def __init__(self, simulator):
        self.simulator = simulator
        self.simulator.launch("f_enable_timer", nonblock=False)
        # self.simulator.launch("f_sync", nonblock=False)
        
    def start(self):
        """Start hardware timer with proper synchronization"""
        self.simulator.launch("f_tic", nonblock=True)  # Wait for tic to complete
        
    def stop(self):
        """Stop hardware timer and collect timestamps"""
        self.simulator.launch("f_toc", nonblock=True)  # Wait for toc to complete
        self.simulator.launch("f_memcpy_timestamps", nonblock=False)
        self.simulator.launch("f_reference_timestamps", nonblock=False)
        
    def get_timing_data(self, simple_memcpy, symbols, px, py, w, h):
        """Retrieve timing data with proper synchronization"""
        time_memcpy_1d_f32 = np.zeros(h*w*3, np.float32)
        time_ref_1d_f32 = np.zeros(h*w*2, np.float32)
        
        # Get timing data with synchronization
        simple_memcpy.do_memcpy_d2h(time_memcpy_1d_f32, symbols['time_memcpy'], px, py, w, h, 3)
        simple_memcpy.do_memcpy_d2h(time_ref_1d_f32, symbols['time_ref'], px, py, w, h, 2)
        
        # Reshape data
        time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (h, w, 3), order='C')
        time_ref_hwl = np.reshape(time_ref_1d_f32, (h, w, 2), order='C')
        
        # create a new to store the timing data.
        hardware_timing = {
          "time_memcpy_hwl": time_memcpy_hwl,
          "time_ref_hwl": time_ref_hwl,
        }
        return hardware_timing

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

def perform_coarse_solve(ml, A_coarse, x_coarsest, b_coarse, solver_callable_host):
    """Handle coarse level solve"""    
    ############################################################
    # Store original dimensions before unpadding - HACK.
    unpadded_x_shape = A_coarse.shape[1]
    unpadded_b_shape = A_coarse.shape[0]
    padded_x_shape = x_coarsest.shape[0]
    padded_b_shape = b_coarse.shape[0]
    # Unpad x and b for coarse level calculations
    x_coarsest = unpad_1d(x_coarsest, unpadded_x_shape)
    b_coarse = unpad_1d(b_coarse, unpadded_b_shape)
    
    # Convert A to CSR format for coarse solver
    A_coarse = sp.csr_matrix(A_coarse)
    ############################################################  
      
    print("\n" + "\t" + "-"*40)
    print("\t" + "│ COARSE SOLVE")
    print("\t" + "-"*40)
    
    print("\t" + "Matrix Dimensions:")
    print("\t" + f"  A: {A_coarse.shape}")
    print("\t" + f"  x: {x_coarsest.shape} | b: {b_coarse.shape}")
    
    x_coarsest[:] = solver_callable_host(A_coarse, b_coarse)
    
    ############################################################
    # Pad x_coarsest to its original dimensions. HACK
    x_coarsest = pad_1d(x_coarsest, padded_x_shape, variable_name="x_coarsest")
    b_coarse = pad_1d(b_coarse, padded_b_shape, variable_name="b_coarse")
    ############################################################
    
    return x_coarsest

def copy_x_coarse_on_device(simple_memcpy, simulator, symbols, ml, layer_coordinates_map, x_level, b_level, setup_config, iteration, deviceprofiling, hardwareTimer):
    print("\n" + "\t" + "-"*40)
    print("\t" + "│ COPYING X_COARSE TO DEVICE (column bcast down)")
    print("\t" + "-"*40)

    coords = layer_coordinates_map[0]
    px, py = coords['layer_start_x'], coords['layer_start_y']
    w, h = coords['layer_pe_cols'], coords['layer_pe_rows']

    # Get R_M for all levels except the last (include L2!)
    R_M_sizes = [ml.levels[i].R.shape[0] for i in range(0, len(ml.levels)-1)]
    total_size = sum(R_M_sizes)
    coarsest_size = ml.levels[-2].R.shape[0] // w  # R_M of the last level with R (L2)
    chunk_size = total_size // w

    # X is not padded to be able to map on WSE.
    R_M_before_last = ml.levels[-2].R.shape[0]
    # padded and new copy is created.
    x_coarse_last = pad_1d(x_level[-1], R_M_before_last, variable_name=f"x_coarse_last")
    x_coarse_last_chunks = np.split(x_coarse_last, w)
    

    # For each column, build its chunk: zeros for all but last layer, last part is x_coarse_last_div
    x_chunks = []
    for j in range(w):
        chunk = np.zeros(chunk_size, dtype=np.float32)
        # Place the coarsest x values at the end of the chunk
        chunk[-coarsest_size:] = x_coarse_last_chunks[j]
        print(f"\tx_coarse_host chunk {j} shape: {chunk.shape}, last {coarsest_size} values set to: {x_coarse_last_chunks[j]}")
        x_chunks.append(chunk)

    x_blob = np.concatenate(x_chunks)

    # Broadcast down columns: each row gets the same chunk for its column
    start_time = time.time()
    simple_memcpy.do_memcpy_h2d(symbols['x_coarse'], x_blob, px, py, w, 1, chunk_size)
    h2d_time = time.time() - start_time
    print(f"\tH2D transfer time: {h2d_time:.4f}s")


def copy_all_layers_on_device(simple_memcpy, simulator, symbols, ml, layer_coordinates_map, x_level, b_level, setup_config, iteration, deviceprofiling, hardwareTimer):
    """Copy all layers data as concatenated blobs to device
    
    Example for 4x4 matrices on 2x2 PE grid with 2 layers:
    
    Layer A0:                Layer A1:               PE Grid:
          +                         +
    [0  1 | 2  3]            [16 17 | 18 19]          +----+----+
    [4  5 | 6  7]            [20 21 | 22 23]          |P00 |P01 |
  + ------------- +          + ------------- +      +----+----+
    [8  9 | 10 11]           [24 25 | 26 27]          |P10 |P11 |
    [12 13| 14 15]           [28 29 | 30 31]          +----+----+
          +                                         
    
    1. Split each layer into PE chunks:
       PE(0,0): A0[0,1,4,5]     PE(0,1): A0[2,3,6,7]
       PE(1,0): A0[8,9,12,13]   PE(1,1): A0[10,11,14,15]
       
       PE(0,0): A1[16,17,20,21] PE(0,1): A1[18,19,22,23]
       PE(1,0): A1[24,25,28,29] PE(1,1): A1[26,27,30,31]
    
    2. Concatenate layers for each PE:
       PE(0,0): [0,1,4,5,16,17,20,21]      # Layer0 chunk + Layer1 chunk
       PE(0,1): [2,3,6,7,18,19,22,23]      # Layer0 chunk + Layer1 chunk
       PE(1,0): [8,9,12,13,24,25,28,29]    # Layer0 chunk + Layer1 chunk
       PE(1,1): [10,11,14,15,26,27,30,31]  # Layer0 chunk + Layer1 chunk
    
    3. Final A_blob sent to device:
       [0,1,4,5,16,17,20,21,    # PE(0,0) data
        2,3,6,7,18,19,22,23,    # PE(0,1) data
        8,9,12,13,24,25,28,29,  # PE(1,0) data
        10,11,14,15,26,27,30,31]# PE(1,1) data
    
    Each PE receives its concatenated chunk containing data from all layers.
    """
    print("\n" + "\t" + "-"*40)
    print("\t" + "│ COPYING ALL LAYERS TO DEVICE")
    print("\t" + "-"*40)

    # Get coordinates from first layer as reference
    coords = layer_coordinates_map[0]
    px, py = coords['layer_start_x'], coords['layer_start_y']
    w, h = coords['layer_pe_cols'], coords['layer_pe_rows']

    # Initialize dictionaries to store PE-wise chunks
    pe_chunks_A = {(i, j): [] for i in range(h) for j in range(w)}
    pe_chunks_R = {(i, j): [] for i in range(h) for j in range(w)}
    pe_chunks_P = {(i, j): [] for i in range(h) for j in range(w)}
    pe_chunks_x = {(i, j): [] for i in range(h) for j in range(w)}
    b_transformed = None
    
    # Transform data for all layers
    for level_index, level in enumerate(ml.levels[:-1]):
        # Get matrices and vectors (already padded from pad_and_make_dense)
        A, R, P = level.A, level.R, level.P
        M, N = A.shape
        R_M, R_N = R.shape
        x = x_level[level_index]
        
        # Calculate per-PE dimensions (must be equal for all PEs)
        per_pe_rows = M // h
        per_pe_cols = N // w
        per_pe_restrict_rows = R_M // h
        per_pe_restrict_cols = R_N // w
        
        print(f"\n\tLayer {level_index} sizes (after padding):")
        print(f"\t  A: {A.shape} | R: {R.shape} | P: {P.shape}")
        print(f"\t  x: {x.shape}")
        
        # Transform A matrix using stack/split and store PE-wise
        A_transformed = np.stack(np.split(np.stack(np.split(A, h, axis=1)), w, axis=1))
        # Transform R matrix using stack/split and store PE-wise
        R_transformed = np.stack(np.split(np.stack(np.split(R, h, axis=1)), w, axis=1))  
        # P is a transpose of R.
        P_transformed = np.stack(np.split(np.stack(np.split(P, h, axis=1)), w, axis=1))
        # Transform x vector and store PE-wise (x is already padded)
        x_transformed = x.reshape(h, per_pe_rows)        
        for i in range(h):
            for j in range(w):
                pe_chunks_A[(i,j)].append(A_transformed[i,j].ravel())
                pe_chunks_R[(i,j)].append(R_transformed[i,j].ravel())
                pe_chunks_P[(i,j)].append(P_transformed[i,j].ravel())
                pe_chunks_x[(i,j)].append(x_transformed[i].ravel())
        
        # Handle b vector for first layer only (b is already padded)
        if level_index == 0:
            b = b_level[0]
            b_transformed = b.flatten(order='C')

    # Concatenate PE-wise chunks and prepare final blobs
    A_final_blob = []
    R_final_blob = []
    P_final_blob = []
    x_final_blob = []
    
    for i in range(h):
        for j in range(w):
            # Concatenate all layers for each PE
            A_pe_data = np.concatenate(pe_chunks_A[(i,j)])
            R_pe_data = np.concatenate(pe_chunks_R[(i,j)])
            P_pe_data = np.concatenate(pe_chunks_P[(i,j)])
            x_pe_data = np.concatenate(pe_chunks_x[(i,j)])
                            
            A_final_blob.append(A_pe_data)
            R_final_blob.append(R_pe_data)
            P_final_blob.append(P_pe_data)
            x_final_blob.append(x_pe_data)

            # if (i,j) == (0,0):
            #     print(f"PE({i},{j}) concatenated sizes:")
            #     print(f"  A: {A_pe_data.shape}")
            #     print(f"  R: {R_pe_data.shape}")
            #     print(f"  x: {x_pe_data.shape}")
                              
            #     # print("PE(0,0) data:")
            #     # print(f"A_pe_data: {A_pe_data}")
            #     # print(f"R_pe_data: {R_pe_data}")
            #     # print(f"x_pe_data: {x_pe_data}")

    # Flatten the final blobs
    A_blob = np.concatenate(A_final_blob)
    R_blob = np.concatenate(R_final_blob)
    P_blob = np.concatenate(P_final_blob)
    x_blob = np.concatenate(x_final_blob)

    # Setup solver parameters (same as before)
    omega = np.zeros(h*w, dtype=np.float32)
    iterations = np.zeros(h*w, dtype=np.uint32)
    omega[:] = amg.get_omega_from_presmoother(setup_config)
    iterations[:] = amg.get_iterations_from_presmoother(setup_config)

    print("\n\tFinal blob sizes:")
    print(f"\tA blob: {A_blob.shape}")
    print(f"\tR blob: {R_blob.shape}")
    print(f"\tP blob: {P_blob.shape}")
    print(f"\tx blob: {x_blob.shape}")

    # Copy concatenated data to device
    start_time = time.time()
    simple_memcpy.do_memcpy_h2d(symbols['A'], A_blob, px, py, w, h, A_blob.size//(w*h))
    simple_memcpy.do_memcpy_h2d(symbols['R'], R_blob, px, py, w, h, R_blob.size//(w*h))
    simple_memcpy.do_memcpy_h2d(symbols['P'], P_blob, px, py, w, h, P_blob.size//(w*h))
    simple_memcpy.do_memcpy_h2d_bcast(symbols['x'], x_blob, px, py, w, h, x_blob.size//(w*h), is_rowbcast=False)
    
    if b_transformed is not None:
        simple_memcpy.do_memcpy_h2d(symbols['b'], b_transformed, px, py, 1, h, b_transformed.size//h)
    
    simple_memcpy.do_memcpy_h2d(symbols['omega'], omega, px, py, w, h, 1)
    simple_memcpy.do_memcpy_h2d(symbols['iterations'], iterations, px, py, w, h, 1)
    h2d_time = time.time() - start_time
    print(f"\tH2D transfer time: {h2d_time:.4f}s")

def copy_all_layers_from_device(simple_memcpy, symbols, ml, total_pe_rows, total_pe_cols, level_data, symbol_name, is_b_level=False):
    """Copy data from device and reconstruct layer-wise data
    
    Args:
        simple_memcpy: Memcpy handler
        symbols: Dictionary of device symbols
        ml: Multilevel hierarchy
        total_pe_rows: Total number of PE rows
        total_pe_cols: Total number of PE columns
        level_data: List to store data for each level (e.g., b_level or x_level)
        symbol_name: Name of the symbol to copy from device (e.g., 'b_next' or 'x')
        is_b_level: Whether this is b_level data (affects how we calculate sizes and where to read from)
        
    Returns:
        level_data: Updated list with reconstructed data for each level
    """
    # Calculate total size by summing appropriate dimensions from each layer
    total_size = 0
    layer_sizes = []  # Store sizes for each layer
    for level in ml.levels[:-1]:  # Exclude coarsest level
        if is_b_level:
            # For b_level, use R_M
            size = level.R.shape[0]
        else:
            # For x_level, use N
            size = level.A.shape[1]
        layer_sizes.append(size)
        total_size += size
    
    # Create buffer for entire blob
    data_blob = np.zeros(total_size, dtype=np.float32)
    
    # Copy entire blob from device
    if is_b_level:
        # For b_level, read from last column
        simple_memcpy.do_memcpy_d2h(data_blob, symbols[symbol_name], total_pe_cols-1, 0, 1, total_pe_rows, total_size//total_pe_rows)
    else:
        # For x_level, read from first row
        simple_memcpy.do_memcpy_d2h(data_blob, symbols[symbol_name], 0, 0, total_pe_cols, 1, total_size//total_pe_cols)
    
    # First divide blob into PE chunks
    if is_b_level:
        # For b_level, divide by rows
        pe_chunk_size = total_size // total_pe_rows
        pe_chunks = []
        for i in range(total_pe_rows):
            start_idx = i * pe_chunk_size
            end_idx = start_idx + pe_chunk_size
            pe_chunks.append(data_blob[start_idx:end_idx])
    else:
        # For x_level, divide by columns
        pe_chunk_size = total_size // total_pe_cols
        pe_chunks = []
        for i in range(total_pe_cols):
            start_idx = i * pe_chunk_size
            end_idx = start_idx + pe_chunk_size
            pe_chunks.append(data_blob[start_idx:end_idx])
    
    # Track read offset within each PE chunk
    if is_b_level:
        pe_offsets = [0] * total_pe_rows
    else:
        pe_offsets = [0] * total_pe_cols
    
    # Reconstruct layer-wise data and copy into level_data
    start_idx = 1 if is_b_level else 0
    for layer_idx, size in enumerate(layer_sizes):
        layer_data = []
        if is_b_level:
            size_per_pe = size // total_pe_rows
            num_pes = total_pe_rows
        else:
            size_per_pe = size // total_pe_cols
            num_pes = total_pe_cols
            
        for i, pe_chunk in enumerate(pe_chunks):
            start = pe_offsets[i]
            end = start + size_per_pe
            layer_data.extend(pe_chunk[start:end])
            pe_offsets[i] += size_per_pe  # Move offset
        # Copy reconstructed data into level_data
        level_data[layer_idx + start_idx] = np.array(layer_data)
        # print(f"\nLayer {layer_idx} reconstructed data for {symbol_name} (size {len(layer_data)}):")
        # print(layer_data)
        
    return level_data

def device_calculations_distributed(v_cycle_data):
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
    print(f"\t*** Layout Load done in {end-start}s")

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
    # Initialize device profiler
    deviceprofiling = time_ut.DeviceProfiling()
    # hardwareTimer = HardwareTimerManager(simulator) # enable_timer, sync
    print("Step 3: Copy all layers' data on device")
    copy_all_layers_on_device(simple_memcpy, simulator, symbols, ml, layer_coordinates_map, x_level, b_level, setup_config, iteration, deviceprofiling, hardwareTimer=None)
    
    print("Step 4: Compute V-Cycle Down")    
    simulator.launch("v_cycle_down", nonblock=False)        
    
    # # D2H transfers
    print("Step 5: D2H Transfers x_smoothed and b_next")
    b_level = copy_all_layers_from_device(simple_memcpy, symbols, ml, total_pe_rows, total_pe_cols, b_level, 'b_next', is_b_level=True)
    x_level = copy_all_layers_from_device(simple_memcpy, symbols, ml, total_pe_rows, total_pe_cols, x_level, 'x', is_b_level=False)
    
    ##### COARSE
    # TODO: Perform on device.
    print("Step 6: Coarse Solve")
    b_coarsest = b_level[-1]    # 2 (unpad)
    x_coarsest = x_level[-1]    # 1
    A_coarsest = ml.levels[-1].A    # 1
    # updates x_level
    x_coarsest[:] = perform_coarse_solve(ml, A_coarsest, x_coarsest, b_coarsest, solver_callable_host)


    print("Step 7: Print V-Cycle Down Results")
    amg.debugprint(ml.levels, b_level, x_level)
    
    print("Step 8: Copy x_coarse on device")
    copy_x_coarse_on_device(simple_memcpy, simulator, symbols, ml, layer_coordinates_map, x_level, b_level, setup_config, iteration, deviceprofiling, hardwareTimer=None)
    
    print("Step 9: Print V-Cycle Up")
    simulator.launch("v_cycle_up", nonblock=False)
    
    # # D2H transfers
    print("Step 10: D2H Transfers x_smoothed and b_next")
    x_level = copy_all_layers_from_device(simple_memcpy, symbols, ml, total_pe_rows, total_pe_cols, x_level, 'x', is_b_level=False)
    print(f"x_level: {x_level}")
    
    # debugprint up cycle
    print("Step 11: Debugprint Up Cycle")
    amg.debugprint(ml.levels, b_level, x_level)
    
    ############################################################
    # Cleanup simulator
    ############################################################
    simulator.stop()
    ############################################################
    # Final results
    ############################################################
    # time_ut.analyze_timing_data("reports/timing_data.csv")
    # print("\nTiming analysis complete. Open timing_report.html to view the results.")
        
    # We only need to unpad b as x is not padded.
    A_solution, b_solution_device, x_solution_device = ml.levels[len(ml.levels)-1].A, b_level[len(ml.levels)-1], x_level[len(ml.levels)-1]
    b_solution_device_unpadded = unpad_1d(b_solution_device, A_solution.shape[0])   # Todo: Remove later.
    residual_device = np.linalg.norm(b_solution_device_unpadded - A_solution @ x_solution_device)
    return b_solution_device, x_solution_device, residual_device
  
def pad_and_make_dense(layer_coordinates_map, v_cycle_data):
    """Pad matrices and vectors to match PE dimensions and convert sparse to dense"""
    ml = v_cycle_data["ml"]
    x_level = v_cycle_data["x_level"] 
    b_level = v_cycle_data["b_level"]
    coords = layer_coordinates_map[0]
    
    for level_index in range(len(ml.levels) -1):    
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
  N = 4
  M = 4
  eps = 1.e-5
  max_iterations = 1
    
  A0, x0, b0 = generate_input2(M, N, type=np.float32)
  nrm_b = np.linalg.norm(b0, 2)
  relative_tol = eps * nrm_b # relative tolerance
  
  checkinput(A0, b0, x0, relative_tol, max_iterations)  # Runs solvers from pyamg and scipy.
  # print("A0: ", A0.toarray())
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
    x_level[i+1] = np.zeros(ml.levels[i].R.shape[0], dtype=np.float32) # Changed to 1D array
    
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
  print(f"\t b_solution_final Host:", b_final_host)
  print("############################################################")
  print("# DEVICE CALCULATIONS")
  print("############################################################")
  b_final_device, x_final_device, residual_device = device_calculations_distributed(v_cycle_data_device) # changed to dataflow
  print("\nDEVICE CALCULATIONS DONE")
  print(f"\t Residual Device ||AX-b||:", residual_device)
  print(f"\t b_solution_final Device:", b_final_device)
  print("############################################################")
  print("# COMPARISON")
  print("############################################################")
  
  x_final_device = unpad_1d(x_final_device, x_final_host.shape[0])
  b_final_device = unpad_1d(b_final_device, b_final_host.shape[0])  # HACK remove later.
  print(f"x_final_device(unpadded): {x_final_device}")
  print(f"b_final_device(unpadded): {b_final_device}")
  assert np.allclose(residual_host, residual_device, atol=1e-6), "Residual do not match!"
  assert np.allclose(x_final_host, x_final_device, atol=1e-6), "x_final of host and device do not match!"
  assert np.allclose(b_final_host, b_final_device, atol=1e-6), "b_final of host and device do not match!"
  print("Results Match!(Residual, x_final, b_final)")


if __name__ == "__main__":
  main()
