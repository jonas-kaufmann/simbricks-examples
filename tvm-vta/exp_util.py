import enum
import os

import simbricks.orchestration.nodeconfig as node
import simbricks.orchestration.simulators as sim

#######################################
# Node configurations
# -------------------------------------


class i40eTvmNode(node.I40eLinuxNode):

    def __init__(self) -> None:
        super().__init__()
        self.disk_image = os.path.abspath("./output-tvm/tvm")
        self.memory = 3 * 1024

    def prepare_pre_cp(self):
        cmds = super().prepare_pre_cp()
        cmds.extend(
            [
                "mount -t proc proc /proc",
                "mount -t sysfs sysfs /sys",
                "cd /root/tvm/",
                "export PYTHONPATH=/root/tvm/python:${PYTHONPATH}",
                "export PYTHONPATH=/root/tvm/vta/python:${PYTHONPATH}",
            ]
        )
        return cmds


class i40eVtaNode(i40eTvmNode):
    def __init__(self) -> None:
        super().__init__()
        self.kcmd_append = " memmap=512M!1G"

    def prepare_pre_cp(self):
        cmds = super().prepare_pre_cp()
        cmds.extend(
            [
                (
                    "echo 1"
                    " >/sys/module/vfio/parameters/enable_unsafe_noiommu_mode"
                ),
                'echo "dead beef" >/sys/bus/pci/drivers/vfio-pci/new_id',
            ]
        )
        return cmds


#######################################
# Application configurations
# -------------------------------------


class TvmDeviceType(enum.Enum):
    VTA = "vta"
    CPU = "cpu"


class TvmTracker(node.AppConfig):

    def __init__(self) -> None:
        super().__init__()
        self.tracker_host = "0.0.0.0"
        self.tracker_port = 9190

    def run_cmds(self, node):
        return [
            # Otherwise, might get warnings of SYN flood, which drops requests
            "sysctl -w net.ipv4.tcp_max_syn_backlog=4096",
            (
                "python3 -m tvm.exec.rpc_tracker"
                f" --host={self.tracker_host} --port={self.tracker_port} &"
            ),
            "sleep infinity",
        ]


class VtaRpcServerWTracker(node.AppConfig):

    def __init__(self) -> None:
        super().__init__()
        self.pci_device_id = "0000:00:00.0"
        self.tracker_host = "10.0.0.1"
        self.tracker_port = 9190

    def run_cmds(self, node):
        cmds = [
            # wait for tracker to start
            "sleep 3",
            (
                f"VTA_DEVICE={self.pci_device_id} python3 -m"
                " vta.exec.rpc_server --key=simbricks-pci"
                f" --tracker={self.tracker_host}:{self.tracker_port} &"
            ),
            "sleep infinity",
        ]
        return cmds


class TvmDetectWTracker(node.AppConfig):

    def __init__(self) -> None:
        super().__init__()
        self.tracker_host = "10.0.0.1"
        self.tracker_port = 9190
        self.device = TvmDeviceType.CPU
        self.test_img = "dog.jpg"
        self.repetitions = 5
        self.debug = False
        """Whether to dump inference result."""

    def config_files(self):
        return {
            "deploy_detection-infer.py": open(
                "./tvm_deploy_detection-infer.py",
                "rb",
            ),
        }

    def run_cmds(self, node):
        cmds = [
            # wait for tracker and RPC servers to start
            "sleep 6",
            f"export TVM_TRACKER_HOST={self.tracker_host}",
            f"export TVM_TRACKER_PORT={self.tracker_port}",
            (
                "python3 /tmp/guest/deploy_detection-infer.py "
                f"/root/darknet {self.device.value} {self.test_img} "
                f"{self.repetitions} {int(self.debug)}"
            ),
        ]
        if self.debug:
            cmds.extend(
                [
                    "echo dump deploy_detection-infer-result.png START",
                    "base64 deploy_detection-infer-result.png",
                    "echo dump deploy_detection-infer-result.png END",
                ]
            )
        return cmds


#######################################
# Simulators
# -------------------------------------


class VTADev(sim.PCIDevSim):

    def __init__(self) -> None:
        super().__init__()
        self.clock_freq = 100
        """Clock frequency in MHz"""

    def run_cmd(self, env):
        cmd = (
            "./vta_src/simbricks/vta_simbricks "
            f"{env.dev_pci_path(self)} {env.dev_shm_path(self)} "
            f"{self.start_tick} {self.sync_period} {self.pci_latency} "
            f"{self.clock_freq}"
        )
        return cmd
