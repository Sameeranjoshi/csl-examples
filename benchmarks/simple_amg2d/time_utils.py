from dataclasses import dataclass


# AMG timing data class
@dataclass
class OperatorTiming:
    """Timing data for AMG operators"""
    smoothing_time: float = 0.0
    residual_time: float = 0.0
    restriction_time: float = 0.0
    prolongation_time: float = 0.0
    matrix_multiply_time: float = 0.0
    total_time: float = 0.0

@dataclass 
class DeviceOperatorTiming:
    """Timing data for AMG device operators"""
    h2d_time: float = 0.0    # Host to device transfer time in seconds
    d2h_time: float = 0.0    # Device to host transfer time in seconds
    cycles: float = 0.0      # Hardware cycles
    kernel_time_us: float = 0.0     # Time in microseconds
