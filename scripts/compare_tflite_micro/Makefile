# Makefile for the TFLite Micro inference harness.
#
# Build the same way for host and Pi (cross-compile by overriding CC/CXX/AR
# from the command line). Optimization flags mirror rusty-axon's
# `release-edge` Cargo profile (-Os, -flto, -s) so binary-size and latency
# numbers in docs/COMPARISON.md are measured under matching compiler stances.
#
# Usage:
#   make TFLM_DIR=tflite-micro                       # host build
#   make TFLM_DIR=tflite-micro \
#        CC=aarch64-linux-gnu-gcc \
#        CXX=aarch64-linux-gnu-g++ \
#        AR=aarch64-linux-gnu-ar \
#        target=tflm_mnist_aarch64                   # Pi Zero 2 W cross
#
# Prereqs:
#   - tensorflow/tflite-micro checked out at $(TFLM_DIR), built once via
#     `make -C $(TFLM_DIR) -f tensorflow/lite/micro/tools/make/Makefile microlite`
#     Output ends up at
#     $(TFLM_DIR)/gen/<host_dir>/lib/libtensorflow-microlite.a
#   - python-tests/train_keras_mnist.py has been run (produces
#     mnist_mlp_tflite.h next to this Makefile).

TFLM_DIR  ?= tflite-micro
target    ?= tflm_mnist
CC        ?= cc
CXX       ?= c++
AR        ?= ar

# Locate the prebuilt microlite static library. TFLM names the gen dir
# after the host triple, so glob defensively.
TFLM_LIB := $(firstword $(wildcard $(TFLM_DIR)/gen/*/lib/libtensorflow-microlite.a))

INCLUDES := \
  -I$(TFLM_DIR) \
  -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/gemmlowp \
  -I$(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads/ruy \
  -I.

# Match rusty-axon's release-edge profile: opt-level z, LTO fat, panic=abort
# (no C analogue — TFLM uses bare exceptions-off C++ already), strip.
CFLAGS_OPT := -Os -flto -ffunction-sections -fdata-sections -DNDEBUG

# CRITICAL: must match the defines libtensorflow-microlite.a was built with,
# otherwise TfLiteTensor and friends read at different field offsets in the
# harness vs the lib (silent ABI skew -> garbage `type`/`dims`/`data`,
# segfault on Invoke). TFLM's microlite Makefile applies these unconditionally
# in COMMON_FLAGS; TF_LITE_USE_CTIME is added when TARGET == HOST_OS.
TFLM_DEFINES := -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DTF_LITE_USE_CTIME

# Mirror microlite's compile stance so optimizer-visible attributes
# (function/data sections, no unwind tables, no message-length wrapping)
# don't introduce subtle differences across the lib/harness boundary.
TFLM_COMMON := -fno-unwind-tables -fno-asynchronous-unwind-tables -fmessage-length=0

LDFLAGS_OPT := -Wl,--gc-sections -flto -s

CXXFLAGS := -std=c++17 -fno-exceptions -fno-rtti \
            $(CFLAGS_OPT) $(TFLM_COMMON) $(TFLM_DEFINES) $(INCLUDES)

OBJS := main.o

.PHONY: all clean check_lib

all: check_lib $(target)

check_lib:
	@if [ -z "$(TFLM_LIB)" ]; then \
	  echo "ERROR: libtensorflow-microlite.a not found under $(TFLM_DIR)/gen."; \
	  echo "Run: make -C $(TFLM_DIR) -f tensorflow/lite/micro/tools/make/Makefile microlite"; \
	  exit 1; \
	fi
	@echo "Linking against $(TFLM_LIB)"

$(target): $(OBJS) $(TFLM_LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS_OPT) -o $@ $(OBJS) $(TFLM_LIB) -lm

main.o: main.c mnist_mlp_tflite.h
	$(CXX) $(CXXFLAGS) -c -o $@ main.c

mnist_mlp_tflite.h:
	@echo "ERROR: mnist_mlp_tflite.h missing. Run python-tests/train_keras_mnist.py first."; \
	exit 1

clean:
	rm -f $(OBJS) $(target) $(target)_aarch64
