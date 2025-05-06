import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder
 
simulator = SdkRuntime("out")
simulator.load()
simulator.run()
 
print("Compute")
simulator.launch("v_cycle_down", nonblock=False)

b_next_symbol = simulator.get_id("b_next")
print("D2H transfer")
b_coarse_device = np.zeros(2, dtype=np.float32)
simulator.memcpy_d2h(b_coarse_device, b_next_symbol, 3, 0, 1, 1, 1, streaming=False,
    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
simulator.stop()        

print("DEVICE CALCULATIONS DONE")