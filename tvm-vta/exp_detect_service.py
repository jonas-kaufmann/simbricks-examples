import os
import sys

sys.path.append(str(os.path.dirname(__file__)))

import itertools

import exp_util
import simbricks.orchestration.simulator_utils as sim_utils
import simbricks.orchestration.simulators as sim
from simbricks.orchestration import experiments

experiments = []

# Experiment parameters
host_variants = ["qemu_k", "qemu_i"]
num_clients_opts = [1, 4, 12]
num_servers_opts = [1, 2, 4, 6]
inference_device_opts = [exp_util.TvmDeviceType.CPU, exp_util.TvmDeviceType.VTA]
vta_clk_freq_opts = [100, 400]

# Build experiments for all parameter combinations
for (
    host_var,
    num_clients,
    num_servers,
    inference_device,
    vta_clk_freq,
) in itertools.product(
    host_variants,
    num_clients_opts,
    num_servers_opts,
    inference_device_opts,
    vta_clk_freq_opts,
):
    experiment = exp.Experiment(
        f"detect_service-{inference_device.value}-{host_var}-{num_servers}s-{num_clients}c-{vta_clk_freq}"
    )
    pci_vta_id_start = 3
    sync = False
    if host_var == "qemu_k":
        HostClass = sim.QemuHost
    elif host_var == "qemu_i":
        HostClass = sim.QemuIcountHost
        sync = True

    # Instantiate network
    switch = sim.SwitchNet()
    switch.name = "switch0"
    experiment.add_network(switch)

    # Instantiate load balancer
    tracker = sim_utils.create_basic_hosts(
        experiment,
        1,
        "tvm_tracker",
        switch,
        sim.I40eNIC,
        HostClass,
        exp_util.i40eTvmNode,
        exp_util.TvmTracker,
    )[0]

    # Instantiate & configure inference servers
    servers = sim_utils.create_basic_hosts(
        experiment,
        num_servers,
        "vta_server",
        switch,
        sim.I40eNIC,
        HostClass,
        exp_util.i40eVtaNode,
        object,
        2,
    )
    for i in range(len(servers)):
        app = exp_util.VtaRpcServerWTracker()
        app.tracker_host = tracker.node_config.ip
        app.pci_device_id = f"0000:00:{(pci_vta_id_start):02d}.0"
        servers[i].node_config.app = app

        vta = exp_util.VTADev()
        vta.name = f"vta{i}"
        vta.clock_freq = vta_clk_freq
        servers[i].add_pcidev(vta)
        experiment.add_pcidev(vta)

    # Instantiate & configure clients using inference service
    clients = sim_utils.create_basic_hosts(
        experiment,
        num_clients,
        "tvm_client",
        switch,
        sim.I40eNIC,
        HostClass,
        exp_util.i40eTvmNode,
        object,
        2 + num_servers,
    )
    for client in clients:
        app = exp_util.TvmDetectWTracker()
        app.tracker_host = tracker.node_config.ip
        app.device = inference_device
        client.node_config.app = app
        client.wait = True

    # Set whether simulators should synchronize or not
    for dev in experiment.pcidevs:
        dev.sync_mode = 1 if sync else 0
    for host in experiment.hosts:
        host.node_config.nockp = not experiment.checkpoint
        host.sync = sync
    switch.sync = sync

    experiments.append(experiment)
