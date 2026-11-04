"""Persistent CPU Qwen decoder. No second decode or custom frame sampling."""
import atexit
import contextlib
import json
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


class VideoDecodeError(RuntimeError):
    pass


class QwenDecoder:
    def __init__(self, timeout=120, attempts=3):
        if timeout <= 0 or attempts < 1:
            raise ValueError('decode timeout must be positive; attempts must be >= 1')
        self.timeout, self.attempts = timeout, attempts
        self.proc = None
        self.temp = tempfile.TemporaryDirectory(prefix='qwen-decode-')
        atexit.register(self.close)

    def stop(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            self.proc.stdin.close()
            self.reader.join(timeout=5)
            self.proc.stdout.close()
            self.proc = None

    def close(self):
        self.stop()
        self.temp.cleanup()

    def start(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', FORCE_QWENVL_VIDEO_READER='decord')
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, env=env)
        responses = queue.Queue()
        self.responses = responses
        stream = self.proc.stdout

        def read_responses():
            for line in stream:
                responses.put(line)
            responses.put('')

        self.reader = threading.Thread(target=read_responses, daemon=True)
        self.reader.start()
        try:
            ready = self.receive()
            if ready.get('status') != 'ready':
                raise RuntimeError('Qwen decoder initialization failed: ' + str(ready))
        except Exception:
            self.stop()
            raise

    def receive(self):
        line = self.responses.get(timeout=self.timeout)
        if not line:
            raise RuntimeError('Qwen decode worker exited unexpectedly')
        return json.loads(line)

    def __call__(self, messages, **kwargs):
        output = str(Path(self.temp.name) / 'result.pkl')
        errors = []
        for attempt in range(1, self.attempts + 1):
            if self.proc is None:
                self.start()  # Configuration/import errors must stop the run.
            try:
                self.proc.stdin.write(json.dumps(dict(messages=messages, kwargs=kwargs,
                                                      output=output)) + '\n')
                self.proc.stdin.flush()
                response = self.receive()
                if response['status'] != 'ok':
                    raise RuntimeError(response.get('error', str(response)))
                # Only load the private local file written by our own child.
                with open(output, 'rb') as handle:
                    return pickle.load(handle)
            except (queue.Empty, RuntimeError, BrokenPipeError) as exc:
                reason = '120s timeout' if isinstance(exc, queue.Empty) else str(exc)
                if isinstance(exc, queue.Empty):
                    reason = f'{self.timeout}s timeout'
                errors.append(reason)
                print(f'[decode] attempt {attempt}/{self.attempts} failed: {reason}', flush=True)
                self.stop()
            finally:
                Path(output).unlink(missing_ok=True)
        raise VideoDecodeError('; '.join(errors))


_decoder = None


def configure_decoder(timeout=120, attempts=3):
    global _decoder
    if _decoder is not None:
        _decoder.close()
    _decoder = QwenDecoder(timeout, attempts)
    _decoder.start()


def process_vision_info(messages, **kwargs):
    if _decoder is None:
        configure_decoder()
    return _decoder(messages, **kwargs)


def get_video_duration(video_path):
    return process_vision_info([], _duration=str(video_path))


def read_duration(path):
    try:
        import decord
        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        fps = vr.get_avg_fps()
        if fps and fps > 0:
            return float(len(vr) / fps)
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                if fps > 0 and count > 0:
                    return float(count / fps)
        finally:
            cap.release()
    except Exception:
        pass
    return 0.0


def worker():
    protocol = sys.stdout
    # Keep library prints off the JSON protocol channel.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from qwen_vl_utils import process_vision_info as process
            import qwen_vl_utils.vision_process as vision
            readers = vision.VIDEO_READER_BACKENDS
            if 'decord' not in readers or 'torchvision' not in readers:
                raise RuntimeError('Unsupported qwen_vl_utils backend API')
            import decord  # Fail at initialization if dependency is missing.

            def no_fallback(*args, **kwargs):
                raise RuntimeError('Decord failed; torchvision fallback is disabled')

            readers['torchvision'] = no_fallback
            vision.get_video_reader_backend = lambda: 'decord'
        except Exception as exc:
            protocol.write(json.dumps({'status': 'init_error', 'error': repr(exc)}) + '\n')
            protocol.flush()
            return
        protocol.write(json.dumps({'status': 'ready'}) + '\n')
        protocol.flush()
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if '_duration' in request['kwargs']:
                    result = read_duration(request['kwargs']['_duration'])
                else:
                    result = process(request['messages'], **request['kwargs'])
                with open(request['output'], 'wb') as handle:
                    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                del result
                response = {'status': 'ok'}
            except Exception as exc:
                response = {'status': 'error', 'error': repr(exc)}
            protocol.write(json.dumps(response) + '\n')
            protocol.flush()


if __name__ == '__main__':
    worker()
