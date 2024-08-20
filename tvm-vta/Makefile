PACKER_VERSION := 1.7.0

IMAGE := output-tvm/tvm
VTA_DIR := vta_src/simbricks
VTA := $(VTA_DIR)/vta_simbricks
PACKER := ../custom-image/packer
IMAGE_PKR_HCL := ../custom-image/image/image.pkr.hcl

all: experiment_requirements

%.raw: %
	qemu-img convert -f qcow2 -O raw $< $@

$(IMAGE): $(PACKER) $(IMAGE_PKR_HCL) image/install.sh image/cleanup.sh
	rm -rf $(dir $@)
	mkdir -p input-tvm
	$(PACKER) build -var="syscores=`nproc`" -var="outname=tvm" \
		-var="http_proxy=$$http_proxy" -var="https_proxy=$$https_proxy" \
		$(IMAGE_PKR_HCL)
	rm -rf input-tvm
	touch $@

$(PACKER):
	$(MAKE) -C ../custom-image packer

.PHONY: clean
clean:
	rm -rf output-tvm input-tvm $(PACKER) packer_cache out
	$(MAKE) -C $(VTA_DIR) clean

$(VTA):
	SIMBRICKS_LIB=/simbricks/lib $(MAKE) -C $(VTA_DIR)

.PHONY: experiment_requirements
experiment_requirements: $(VTA) $(IMAGE)
