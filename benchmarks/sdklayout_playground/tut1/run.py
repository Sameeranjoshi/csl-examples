#!/usr/bin/env cs_python

import argparse
import numpy as np
from cerebras.geometry.geometry import IntVector, IntRectangle
from cerebras.sdk.runtime.sdkruntimepybind import (
    Color,
    Route,
    Edge,
    RoutingPosition,
    SdkLayout,
    SdkTarget,
    SdkRuntime,
    SimfabConfig,
    get_platform,
)
def make_platform(arch: str):
    cfg = SimfabConfig(dump_core=True)
    tgt = SdkTarget.WSE3 if arch == "wse3" else SdkTarget.WSE2
    return get_platform(None, cfg, tgt)

def rect(x, y, w, h):
    return IntRectangle(IntVector(x, y), IntVector(w, h))

def PE(x, y):
    return IntVector(x, y)

def compile_and_run(platform, layout_artifacts):
    ############
    ### Runtime
    ############
    # Create the runtime using the compilation artifacts and the execution platform.
    runtime = SdkRuntime(layout_artifacts, platform, memcpy_required=False)
    runtime.load()
    runtime.run()
    runtime.stop()

    return runtime

def build_one_pe_only(platform):
    value = 550
    layout = SdkLayout(platform)
    # Create a 1x1 code region using 'gv.csl' as the source code.
    # The code region is called 'gv'.
    code = layout.create_code_region('./gv.csl', 'gv', 1, 1)
    # Set the 'value' param on all PEs in the region. In this
    # example 'code' has only one PE.
    # code.set_param_all('value', value)
    code.set_param_range(rect(0, 0, 1, 1), 'value', value)
    # Compile the layout and use 'out' as the prefix for all
    # produced artifacts.
    compile_artifacts = layout.compile(out_prefix='out')
    runtime = compile_and_run(platform, compile_artifacts)

    # verify
    result = runtime.read_symbol(0, 0, 'gv', dtype='uint16')
    assert result == [value]
    print("SUCCESS!")

def build_send_receive_2_pe(platform):
    layout = SdkLayout(platform)
    code = layout.create_code_region('./send_receive.csl', 'send_receive', 2, 1)
    sender_PE = PE(0,0)
    receiver_PE = PE(1,0)
    code.set_param(sender_PE, 'select', 0) # sender PE = select == 0
    code.set_param(receiver_PE, 'select', 1) # receiver PE = select == 1

    # routing
    send_routes = RoutingPosition().set_input([Route.RAMP]).set_output([Route.EAST])
    recv_routes = RoutingPosition().set_input([Route.WEST]).set_output([Route.RAMP])

    # Colors
    c = Color('c')
    # node, color, routes
    code.paint(sender_PE, c, [send_routes])
    code.paint(receiver_PE, c, [recv_routes])
    code.set_param_all(c)

    compile_artifacts = layout.compile(out_prefix='out')
    runtime = compile_and_run(platform, compile_artifacts)

    # verify

    value = np.array([1,2,3,4,5], dtype=np.uint16)
    result = runtime.read_symbol(1, 0, 'buffer', dtype='uint16')
    assert np.array_equal(value, result)
    print("SUCCESS!")

def build_ports(platform):
    layout = SdkLayout(platform)
    ######################
    ### Common invariants
    ######################
    size = 10
    # Routes
    sender_routes = RoutingPosition().set_input([Route.RAMP])
    receiver_routes = RoutingPosition().set_output([Route.RAMP])


    #################################
    ### Sender 1 and port 'tx1_port'
    #################################
    sender1 = layout.create_code_region('./sender.csl', 'sender1', 1, 1)
    # Color 'tx1' is scoped because even though the name of the color is
    # 'tx' for both senders, colors must be globally unique for the
    # compiler to assign different values to them. By scoping colors like
    # this we are effectively uniqueing them since code regions are unique
    # (i.e., no two regions can have the same name).
    tx1 = sender1.color('tx')
    sender1.set_param_all('size', size)
    sender1.set_param_all(tx1)
    # A sender port is created using a color ('tx1'), an edge (in this
    # example the edge doesn't matter since we have a 1x1 code region),
    # a list of routing positions and a size. The routing positions for an
    # output port must not contain output routes (if they do, an error will
    # be emitted). That's because the compiler is free to chose any output
    # route depending on what's globally optimal. Finally, the 'size' is
    # used to verify compatibility between connected ports.
    tx1_port = sender1.create_output_port(tx1, Edge.RIGHT, [sender_routes], size)


    #################################
    ### Sender 2 and port 'tx2_port'
    #################################
    sender2 = layout.create_code_region('./sender.csl', 'sender2', 1, 1)
    tx2 = sender2.color('tx')
    sender2.set_param_all('size', size)
    sender2.set_param_all(tx2)
    tx2_port = sender2.create_output_port(tx2, Edge.RIGHT, [sender_routes], size)


    ############
    ### Add2vec
    ############
    add2vec = layout.create_code_region('./add2vec.csl', 'add2vec', 1, 1)
    rx1 = Color('rx1')
    rx2 = Color('rx2')
    tx = Color('tx')
    add2vec.set_param_all('size', size)
    add2vec.set_param_all(rx1)
    add2vec.set_param_all(rx2)
    add2vec.set_param_all(tx)
    rx1_port = add2vec.create_input_port(rx1, Edge.RIGHT, [receiver_routes], size)
    rx2_port = add2vec.create_input_port(rx2, Edge.RIGHT, [receiver_routes], size)
    tx_port = add2vec.create_output_port(tx, Edge.LEFT, [sender_routes], size)


    #############
    ### Receiver
    #############
    receiver = layout.create_code_region('./receiver.csl', 'receiver', 1, 1)
    rx = Color('rx')
    receiver.set_param_all('size', size)
    receiver.set_param_all(rx)
    rx_port = receiver.create_input_port(rx, Edge.LEFT, [receiver_routes], size,)


    #####################
    ### Port connections
    #####################
    # This is the key part of this example. The ports defined above for
    # each code region, are now connected. The physical location of the
    # ports can be arbitrary because the SdkLayout compiler will find
    # optimal paths automatically.
    layout.connect(tx1_port, rx1_port)
    layout.connect(tx2_port, rx2_port)
    layout.connect(tx_port, rx_port)    

    #########################
    ### Placement of senders
    #########################
    # We place the senders in arbitrary locations in the layout as
    # an example that demonstrates the ability of the framework to automatically
    # produce paths between input and output ports.
    sender1.place(2, 2)
    sender2.place(4, 7)
    add2vec.place(7, 4)
    receiver.place(3, 3)

    compile_artifacts = layout.compile(out_prefix='out')
    runtime = compile_and_run(platform, compile_artifacts)

    expected = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=np.uint16)
    actual = runtime.read_symbol(3, 3, 'data').view(np.uint16)
    assert np.array_equal(expected, actual)
    print("SUCCESS!")
########################################
### Layout, code region and compilation
########################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmaddr', help='IP:port for CS system')
    parser.add_argument(
        '--arch',
        choices=['wse2', 'wse3'],
        default='wse3',
        help='Target WSE architecture (default: wse3)'
    )
    args = parser.parse_args()
    platform = make_platform(args.arch)

    # Tut1
    # build_one_pe_only(platform)
    # build_send_receive_2_pe(platform)
    build_ports(platform)


if __name__ == "__main__":
    main()