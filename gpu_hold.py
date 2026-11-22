"""Reserve 40 GiB on the selected GPU until the task requests release."""
import ctypes as C
import datetime
import os
from pathlib import Path
import signal
import time

root = Path(__file__).resolve().parent
stop_file = root / 'logs/gpu_hold.stop'
ready_file = root / 'logs/gpu_hold.ready'
cuda = C.CDLL('libcuda.so.1')


def check(code):
    if code:
        raise RuntimeError(f'CUDA driver returned {code}')


check(cuda.cuInit(0))
device = C.c_int()
check(cuda.cuDeviceGet(C.byref(device), 0))
bus = C.create_string_buffer(64)
check(cuda.cuDeviceGetPCIBusId(bus, len(bus), device))
if bus.value.decode().lower() != '0000:d8:00.0':
    raise RuntimeError(f'Unexpected GPU PCI address: {bus.value!r}')
context = C.c_void_p()
check(cuda.cuCtxCreate_v2(C.byref(context), 0, device))
pointer = C.c_uint64()
check(cuda.cuMemAlloc_v2(C.byref(pointer), C.c_size_t(40 * 1024 ** 3)))
running = True


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
ready_file.write_text(str(os.getpid()))
print(datetime.datetime.now().astimezone().isoformat(), 'ready', os.getpid(),
      bus.value.decode(), 'reserved 40 GiB; no compute kernels; release via task stop file', flush=True)
try:
    while running and not stop_file.exists():
        time.sleep(1)
finally:
    check(cuda.cuMemFree_v2(pointer))
    check(cuda.cuCtxDestroy_v2(context))
    if ready_file.exists() and ready_file.read_text().strip() == str(os.getpid()):
        ready_file.unlink(missing_ok=True)
    print(datetime.datetime.now().astimezone().isoformat(), 'released', flush=True)
