"""Keep our GPU reservation during first-run model downloads for E0."""
import datetime
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
stop_file = root / 'logs/gpu_hold.stop'
ready_file = root / 'logs/gpu_hold.ready'


def log(message):
    print(datetime.datetime.now().astimezone().isoformat(), message, flush=True)


def alive(pid, name):
    proc = Path(f'/proc/{pid}')
    try:
        return proc.stat().st_uid == os.getuid() and name.encode() in (proc/'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False


def using_gpu(pid):
    result = subprocess.run(['nvidia-smi', '-i', '4', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    return str(pid) in {line.strip() for line in result.stdout.splitlines()}


def reserve(label):
    assert not ready_file.exists(), 'Another reservation is already active'
    stop_file.unlink(missing_ok=True)
    stream = (root / f'logs/gpu_hold_{label}.log').open('a')
    hold = subprocess.Popen(['nohup', '/usr/bin/python3', str(root/'gpu_hold.py')],
                            cwd=root, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    stream.close()
    for _ in range(60):
        if ready_file.exists() and ready_file.read_text().strip() == str(hold.pid):
            (root/'logs/gpu_hold.pid').write_text(str(hold.pid)+'\n')
            log(f'Reservation ready for {label}: PID {hold.pid}')
            return hold
        if hold.poll() is not None:
            raise RuntimeError('Reservation failed to start')
        time.sleep(.5)
    raise RuntimeError('Reservation readiness timeout')


for label, name in [('encode_download', 'encode.py'), ('decode_download', 'decode_eval.py')]:
    pid_file = root / ('logs/encode_test.pid' if name == 'encode.py' else 'logs/decode_ceiling.pid')
    hold = None
    while not pid_file.exists():
        if hold is None:
            hold = reserve(label)
        if (root/'logs/run_to_e0.exit').exists():
            log('Pipeline stopped; keeping bounded reservation for review')
            hold.wait()
            raise SystemExit
        time.sleep(1)
    pid = int(pid_file.read_text())
    while alive(pid, name):
        if using_gpu(pid):
            if hold is not None:
                stop_file.touch()
                hold.wait(timeout=30)
            log(f'GPU handed off to {name}, PID {pid}')
            break
        if hold is None:
            hold = reserve(label)
        time.sleep(2)
    else:
        if hold is None:
            hold = reserve(label)
        log(f'{name} exited before GPU use; keeping bounded reservation')
        hold.wait()
        raise SystemExit
    if name == 'encode.py':
        while alive(pid, name):
            time.sleep(1)
log('E0 model-download handoffs complete; normal pipeline retains review reservation')
