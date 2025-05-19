#!/usr/bin/env python3
"""
multi_segment_recorder_async.py – Raspberry Pi 5 / libgpiod 2.x

Records multiple MJPEG segments and, after each segment:
  • renames the raw file   (…_primming.mjpeg / …_dispensing.mjpeg)
  • analyses it in a background thread
  • RETURNS a list of (idx, mode_changed, result) when run() exits
"""

import asyncio
import os
import pprint
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Tuple

import gpiod
from gpiod.line import Direction, Edge, Bias, Value

# ───────── analysis imports ────────────────────────────────────────
from detect_clogs import (
    threshold_video_movement as threshold_video_movement_primming, 
    homography as homography_primming, 
    classify_nozzles as classify_nozzles_primming
)
from detect_clogs_plate import (
    threshold_video_movement as threshold_video_movement_dispensing, 
    homography as homography_dispensing, 
    classify_nozzles as classify_nozzles_dispensing
)
# ───────────────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "btn": 26,
    "mode_pin": 23,
    "chip_path": "/dev/gpiochip0",
    "width": 760,
    "height": 540,
    "framerate": 30,
    "ev": -1.0,
    "lens_pos": 8.0,
    "dur": 30,
    "debounce": 0.02,
    "min_rec": 0.6,
    "seg_dir": "tmp",
    "final_name": "video.mjpeg",
}


class Recorder:
    """Segmented MJPEG recorder with dual-switch logic + background analysis."""

    def __init__(self, **overrides: Any) -> None:
        self.cfg = cfg = {**DEFAULTS, **overrides}
        base_dir = Path(__file__).resolve().parent
        self.seg_dir = (base_dir / cfg["seg_dir"]).resolve()
        self.seg_dir.mkdir(parents=True, exist_ok=True)

        self.seg_n = 1
        self.proc: subprocess.Popen | None = None
        self.start_ts: float | None = None
        self.window_timer: asyncio.TimerHandle | None = None
        self.record_timer: asyncio.TimerHandle | None = None
        self.mode_changed = False

        self.executor = ThreadPoolExecutor(max_workers=2)
        self.results: List[Tuple[int, bool, Any]] = []

        self._peek_initial_state()

    # ───── initial state ───────────────────────────────────────────
    def _peek_initial_state(self) -> None:
        with gpiod.request_lines(
            self.cfg["chip_path"],
            consumer="peek",
            config={
                self.cfg["btn"]: gpiod.LineSettings(direction=Direction.INPUT,
                                                   bias=Bias.PULL_UP),
                self.cfg["mode_pin"]: gpiod.LineSettings(direction=Direction.INPUT,
                                                         bias=Bias.PULL_UP),
            },
        ) as peek:
            if (
                peek.get_value(self.cfg["btn"]) == Value.ACTIVE
                or peek.get_value(self.cfg["mode_pin"]) == Value.ACTIVE
            ):
                print("Waiting for initial TOUCH …")
                while (
                    peek.get_value(self.cfg["btn"]) == Value.ACTIVE
                    or peek.get_value(self.cfg["mode_pin"]) == Value.ACTIVE
                ):
                    time.sleep(0.01)

    # ───── helpers ────────────────────────────────────────────────
    def _seg_path(self, idx: int | None = None) -> Path:
        return self.seg_dir / f"segment_{idx or self.seg_n}.mjpeg"

    def _stable(self, expect_rising: bool, gpio: int | None = None) -> bool:
        time.sleep(self.cfg["debounce"])
        val = self.req.get_value(gpio or self.cfg["btn"])
        return (val is Value.ACTIVE) if expect_rising else (val is Value.INACTIVE)

    # ───── recording control ───────────────────────────────────────
    def _start_record(self) -> None:
        self.mode_changed = False
        p = self.cfg
        cmd = (
            f"libcamera-vid -t {p['dur'] * 1000} --framerate {p['framerate']} "
            f"--codec mjpeg --flush -o {self._seg_path()} "
            f"--width {p['width']} --height {p['height']} "
            f"--ev {p['ev']} --autofocus-mode manual --lens-position {p['lens_pos']}"
        )
        self.proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
        self.start_ts = time.time()
        print(f"[seg {self.seg_n}] ▶  recording ({p['dur']} s max)")

        loop = asyncio.get_event_loop()
        self.record_timer = loop.call_later(p["dur"], self._on_timeout)

    def _on_timeout(self) -> None:
        if self.proc:
            print(f"[seg {self.seg_n}] ⏰  timeout – stopping")
            self._stop_record()

    # ───── analysis routine ───────────────────────────────────────
    @staticmethod                             # FIX: make static to remove implicit *self*
    def check_nozzles_primming(cycles) -> dict[str, list[bool]]:
        """Analyse primming cycles returned by detect_clogs.threshold_video_movement."""
        nozzle_status: dict[str, list[bool]] = {}
        for i, cycle in enumerate(cycles):
            thresh = cycle["thresholded_image"]
            fid = cycle["fiducial_coordinates"]

            warped_b = homography_primming(fid, thresh, "B")
            warped_f = homography_primming(fid, thresh, "A")

            rep_b, mean_b = classify_nozzles_primming(warped_b, section="B")
            rep_f, mean_f = classify_nozzles_primming(warped_f, section="A")

            if mean_f < 0.1 and mean_b < 0.1:
                continue
            if mean_f > mean_b:
                print(f"Cycle {i}: front (A) wins – ratio {mean_f:.3f} > {mean_b:.3f}")
                nozzle_status.update(rep_f)
            else:
                print(f"Cycle {i}: back (B) wins – ratio {mean_b:.3f} > {mean_f:.3f}")
                nozzle_status.update(rep_b)
        return nozzle_status

    @staticmethod
    def check_nozzles_dispensing(cycles) -> dict[str, list[bool]]:
        """Analyse dispensing cycles (plate) returned by detect_clogs_plate.threshold…"""
        nozzle_status: dict[str, list[bool]] = {}
        for i, cycle in enumerate(cycles):
            thresh = cycle["thresholded_image"]
            fid = cycle["fiducial_coordinates"]

            warped_b = homography_dispensing(fid, thresh, "B")
            warped_f = homography_dispensing(fid, thresh, "A")

            rep_b, mean_b = classify_nozzles_dispensing(warped_b, section="B")
            rep_f, mean_f = classify_nozzles_dispensing(warped_f, section="A")

            if mean_f < 0.1 and mean_b < 0.1:
                continue
            if mean_f > mean_b:
                print(f"Cycle {i}: front (A) wins – ratio {mean_f:.3f} > {mean_b:.3f}")
                nozzle_status.update(rep_f)
            else:
                print(f"Cycle {i}: back (B) wins – ratio {mean_b:.3f} > {mean_f:.3f}")
                nozzle_status.update(rep_b)
        return nozzle_status

    # ───── per-segment analysis dispatcher ────────────────────────
    @staticmethod
    def _analyse_segment(seg_path: Path, mode_changed: bool) -> Any:
        """Return nozzle-check dictionary for this segment."""
        if mode_changed:                                             # dispensing
            print(f"[{seg_path.name}] 🔎 dispensing analysis …")
            cycles = threshold_video_movement_dispensing(str(seg_path))
            return Recorder.check_nozzles_dispensing(cycles)         # FIX
        print(f"[{seg_path.name}] 🔎 primming analysis …")
        cycles = threshold_video_movement_primming(str(seg_path))    # FIX
        return Recorder.check_nozzles_primming(cycles)               # FIX
    
    # ───── finalise one segment ────────────────────────────────────
    def _schedule_finalize(self) -> None:
        proc, raw_path = self.proc, self._seg_path()
        made_change = self.mode_changed
        start_ts = self.start_ts or time.time()
        seg_idx = self.seg_n

        def _finish() -> None:
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass
                proc.wait()

            if raw_path.exists():
                suffix = "dispensing" if made_change else "primming"
                keep_path = raw_path.with_name(
                    f"{raw_path.stem}_{suffix}{raw_path.suffix}"
                )
                if keep_path.exists():
                    keep_path.unlink()
                raw_path.rename(keep_path)

                loop = asyncio.get_event_loop()

                future = loop.run_in_executor(
                    self.executor,
                    Recorder._analyse_segment,
                    keep_path,
                    made_change,
                )

                # store (idx, mode, result) once analysis completes
                def _store(fut: asyncio.Future) -> None:
                    try:
                        res = fut.result()
                    except Exception as exc:              # noqa
                        print(f"[{keep_path.name}] ⚠️  analysis failed: {exc}")
                        res = None
                    self.results.append((seg_idx, made_change, res))
                    print(
                        f"[seg {seg_idx}] ■  saved → {keep_path.name}"
                        f" | mode_changed = {made_change}"
                    )

                future.add_done_callback(_store)

                self.seg_n += 1  # advance only when file kept
            else:
                print(
                    f"[seg {seg_idx}] ■  no file produced – skipped"
                    f" | mode_changed = {made_change}"
                )

        delay = max(0.0, self.cfg["min_rec"] - (time.time() - start_ts))
        asyncio.get_event_loop().call_later(delay, _finish)

        if made_change:
            self.session_changed = True

    def _stop_record(self) -> None:
        if self.record_timer:
            self.record_timer.cancel()
            self.record_timer = None
        self._schedule_finalize()
        self.proc = self.start_ts = None

    # ───── GPIO edge callback ──────────────────────────────────────
    def _on_edge(self) -> None:
        for evt in self.req.read_edge_events():
            gpio = evt.line_offset
            if gpio == self.cfg["btn"]:
                rising = evt.event_type is evt.Type.RISING_EDGE
                if not self._stable(rising):
                    continue
                if rising and self.proc is None:              # start
                    if self.window_timer:
                        self.window_timer.cancel()
                        self.window_timer = None
                    self._start_record()
                elif (not rising) and self.proc:              # stop
                    self._stop_record()
                    loop = asyncio.get_event_loop()
                    self.window_timer = loop.call_later(3, loop.stop)

            elif gpio == self.cfg["mode_pin"] and self.proc:
                if self.mode_changed:                         # already flagged
                    continue
                if evt.event_type is evt.Type.RISING_EDGE and self._stable(True, gpio):
                    self.mode_changed = True
                    print("Mode switch released → mode_changed = True")

    # ───── main run loop ───────────────────────────────────────────
    def run(self) -> List[Tuple[int, bool, Any]]:
        """Run until the event loop stops; return results for all segments."""
        print("Ready – release to start recording")
        loop = asyncio.get_event_loop()

        settings = gpiod.LineSettings(
            direction=Direction.INPUT, edge_detection=Edge.BOTH, bias=Bias.PULL_UP
        )
        cfg_dict = {self.cfg["btn"]: settings, self.cfg["mode_pin"]: settings}

        with gpiod.request_lines(
            self.cfg["chip_path"], consumer="recorder", config=cfg_dict
        ) as req:
            self.req = req
            loop.add_reader(req.fd, self._on_edge)
            try:
                loop.run_forever()
            finally:
                if self.proc:
                    self._stop_record()
                self.executor.shutdown(wait=True)  # wait for analyses to finish
        return self.results


# ───── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    segments = Recorder().run()
    print("\n=== SESSION SUMMARY ===")
    for idx, is_disp, res in segments:
        mode = "dispensing" if is_disp else "primming"
        print(f"Segment {idx} ({mode}) → {pprint.pformat(res, compact=True)}")


