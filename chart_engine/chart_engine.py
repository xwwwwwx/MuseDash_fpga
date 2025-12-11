from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


# ==== chart_check (from chart_engine/check.py) ====
_EVENT_PATTERN = re.compile(r"^\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)$")
_ALLOWED_TYPES = {"tap", "hold_start", "hold_mid"}
_ALLOWED_TRACES = {"0", "1"}


def _resolve_chart_path(chart_name: str, chart_path: Optional[Path]) -> Path:
    if chart_path is not None:
        return chart_path
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "charts" / chart_name / f"{chart_name}.txt"


def chart_check(chart_name: str, chart_path: Optional[Path] = None) -> bool:
    target_path = _resolve_chart_path(chart_name, chart_path)
    if not target_path.exists():
        print(f"[chart_check] 文件不存在: {target_path}")
        return False

    try:
        lines = target_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:  # pragma: no cover
        print(f"[chart_check] 读取文件失败: {target_path} ({exc})")
        return False

    if not lines:
        print("[chart_check] 文件为空")
        return False

    header = lines[0].strip()
    if not header.startswith("bpm="):
        print("[chart_check] 第一行必须为 bpm=<整数>")
        return False

    bpm_value = header.split("=", 1)[1]
    if not bpm_value.isdigit():
        print("[chart_check] BPM 必须为整数")
        return False

    last_time: Optional[int] = None
    last_by_trace: Dict[str, Optional[Tuple[int, str]]] = {"0": None, "1": None}

    for idx, raw_line in enumerate(lines[1:], start=2):
        line = raw_line.strip()
        if not line:
            break

        match = _EVENT_PATTERN.match(line)
        if not match:
            print(f"[chart_check] 第 {idx} 行格式错误，应为 (time,type,trace): {line}")
            return False

        time_str, evt_type, trace_str = match.groups()
        time_str = time_str.strip()
        evt_type = evt_type.strip()
        trace_str = trace_str.strip()

        if evt_type not in _ALLOWED_TYPES:
            print(f"[chart_check] 第 {idx} 行 type 非法: {evt_type}")
            return False

        if trace_str not in _ALLOWED_TRACES:
            print(f"[chart_check] 第 {idx} 行 trace 仅允许 0/1: {trace_str}")
            return False

        if not time_str.lstrip("-").isdigit():
            print(f"[chart_check] 第 {idx} 行 time 必须为整数: {time_str}")
            return False

        time_val = int(time_str)
        if time_val < 0:
            print(f"[chart_check] 第 {idx} 行 time 不得为负: {time_val}")
            return False

        if last_time is not None and time_val < last_time:
            print(f"[chart_check] 时间需整体单调不减：第 {idx} 行 {time_val} < 上一行 {last_time}")
            return False
        last_time = time_val

        prev = last_by_trace[trace_str]
        if prev is not None:
            prev_time, prev_type = prev
            if prev_time >= time_val:
                print(f"[chart_check] 同轨时间需严格递增：轨道 {trace_str} 第 {idx} 行 {time_val} <= 上一事件 {prev_time}")
                return False

            if evt_type == "hold_mid":
                if prev_type not in {"hold_start", "hold_mid"} or prev_time != time_val - 1:
                    print(f"[chart_check] hold_mid 需紧接前一拍同轨 hold_start/hold_mid：第 {idx} 行")
                    return False
            else:
                if prev_type == "hold_start":
                    print(f"[chart_check] hold_start 后必须跟随连续 hold_mid：轨道 {trace_str} 第 {idx} 行")
                    return False
        else:
            if evt_type == "hold_mid":
                print(f"[chart_check] hold_mid 前必须有 hold_start：第 {idx} 行")
                return False

        last_by_trace[trace_str] = (time_val, evt_type)

    for trace, prev in last_by_trace.items():
        if prev is not None and prev[1] == "hold_start":
            print(f"[chart_check] 轨道 {trace} 的 hold_start 未闭合")
            return False

    return True


# ==== generate_random_chart (from chart_engine/random_gen.py) ====
def generate_random_chart(output_dir, name="Random", bpm=120, length_seconds=60, seed=None):
    if seed is not None:
        random.seed(seed)

    output_path = Path(output_dir) / f"{name}.txt"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8"):
            pass
    except Exception as exc:
        print(f"错误：无法创建文件 {output_path}: {exc}")
        return None

    maxlength = bpm * length_seconds / 60
    n = 0
    output_bpm = bpm if bpm is not None else 120
    lines = [f"bpm={output_bpm}"]

    while n < maxlength:
        x = random.randint(2, 6)
        type_val = random.randint(1, 100)
        trace = random.randint(1, 3)

        if type_val <= 85:
            if trace == 1:
                lines.append(f"({n+x},tap,0)")
            elif trace == 2:
                lines.append(f"({n+x},tap,1)")
            elif trace == 3:
                lines.append(f"({n+x},tap,0)")
                lines.append(f"({n+x},tap,1)")
            n = n + x
        else:
            length = random.randint(3, 8)
            head = random.randint(1, 20)
            tail = random.randint(1, 10)
            for m in range(length):
                if m == 0:
                    if trace == 1:
                        lines.append(f"({n+x},hold_start,0)")
                        if head == 1:
                            lines.append(f"({n+x},tap,1)")
                    elif trace == 2:
                        lines.append(f"({n+x},hold_start,1)")
                        if head == 1:
                            lines.append(f"({n+x},tap,0)")
                    elif trace == 3:
                        lines.append(f"({n+x},hold_start,0)")
                        lines.append(f"({n+x},hold_start,1)")
                elif m == length - 1:
                    if trace == 1:
                        lines.append(f"({n+x+m},hold_mid,0)")
                        if tail == 1:
                            lines.append(f"({n+x+m},tap,1)")
                    elif trace == 2:
                        lines.append(f"({n+x+m},hold_mid,1)")
                        if tail == 1:
                            lines.append(f"({n+x+m},tap,0)")
                    elif trace == 3:
                        lines.append(f"({n+x+m},hold_mid,0)")
                        lines.append(f"({n+x+m},hold_mid,1)")
                else:
                    if trace == 1:
                        lines.append(f"({n+x+m},hold_mid,0)")
                    elif trace == 2:
                        lines.append(f"({n+x+m},hold_mid,1)")
                    elif trace == 3:
                        lines.append(f"({n+x+m},hold_mid,0)")
                        lines.append(f"({n+x+m},hold_mid,1)")
            n = n + x + (length - 1)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except Exception as exc:
        print(f"错误：无法写入文件 {output_path}: {exc}")
        return None

    return output_path


# ==== process_chart (adapted from chart_engine/rom_gen.py) ====
def process_chart(chart_name: str, output_filename: str = "ROM.v") -> bool:
    base_dir = Path(__file__).resolve().parent.parent
    chart_path = base_dir / "charts" / "Random" / f"{chart_name}.txt"
    if not chart_path.exists():
        print(f"[process_chart] 文件不存在: {chart_path}")
        return False

    if not chart_check(chart_name, chart_path):
        return False

    pattern = re.compile(r"^\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)$")
    try:
        lines = chart_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        print(f"[process_chart] 读取文件失败: {exc}")
        return False

    # 3.1. 解析第一行获取 BPM 并更新 MuseDash.v 的 div_cnt
    if not lines:
        print(f"[process_chart] 文件为空")
        return False
    bpm_line = lines[0].strip()
    if not bpm_line.startswith("bpm="):
        print(f"[process_chart] 第一行格式错误，应为 bpm=xxx: {bpm_line}")
        return False
    try:
        bpm = float(bpm_line[4:])  # 跳过 "bpm="
        if bpm <= 0:
            print(f"[process_chart] BPM 值无效: {bpm}")
            return False
        div_cnt = int(375000000 / bpm)
    except (ValueError, ZeroDivisionError) as exc:
        print(f"[process_chart] 解析 BPM 失败: {exc}")
        return False
    
    # 更新 MuseDash.v 的 div_cnt
    musedash_path = base_dir / "verilog" / "MuseDash.v"
    try:
        musedash_content = musedash_path.read_text(encoding="utf-8")
        # 使用正则表达式替换 div_cnt 的值
        div_cnt_pattern = r"(parameter\s+div_cnt\s*=\s*)\d+"
        replacement = f"\\g<1>{div_cnt}"
        musedash_content = re.sub(div_cnt_pattern, replacement, musedash_content)
        musedash_path.write_text(musedash_content, encoding="utf-8")
        print(f"[process_chart] 已更新 MuseDash.v 的 div_cnt = {div_cnt} (BPM = {bpm})")
    except Exception as exc:
        print(f"[process_chart] 更新 MuseDash.v 失败: {exc}")
        return False

    events = []
    max_time = 0
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            break
        match = pattern.match(line)
        if not match:
            print(f"[process_chart] 行格式错误: {line}")
            return False
        time_str, evt_type, trace_str = match.groups()
        time_val = int(time_str.strip())
        evt_type = evt_type.strip()
        trace_str = trace_str.strip()
        max_time = max(max_time, time_val)
        events.append((time_val, evt_type, trace_str))

    # 计算 ROM 长度：覆盖到 max_time，最小 1，最大 4096
    max_len = max(1 << max(max_time.bit_length(), 0), 1)
    if max_len > 4096:
        print(f"[process_chart] 谱面时间超过可支持范围: max_time={max_time}")
        return False
    rom_len = 4096

    rom = [0] * rom_len
    type_to_val = {"tap": 0b01, "hold_start": 0b10, "hold_mid": 0b11}
    for time_val, evt_type, trace_str in events:
        val = type_to_val[evt_type]
        if time_val >= rom_len:
            print(f"[process_chart] time 索引越界: time={time_val}, rom_len={rom_len}")
            return False
        if trace_str == "1":
            rom[time_val] = (rom[time_val] & 0b0011) | (val << 2)
        else:
            rom[time_val] = (rom[time_val] & 0b1100) | val

    verilog_path = base_dir / "verilog" / output_filename
    try:
        lines_out = [
            "module ROM (",
            "    input [11:0] addr,",
            "    output reg [1:0] noteup,",
            "    output reg [1:0] notedown",
            ");",
            "",
            f"reg [3:0] ROM [0:{rom_len - 1}];",
            "",
            "initial begin",
        ]
        for idx, val in enumerate(rom):
            lines_out.append(f"\tROM[{idx}] = 4'b{val:04b};")
        lines_out.extend(
            [
                "end",
                "",
                "always @(*) begin",
                "    {noteup, notedown} = ROM[addr];",
                "end",
                "",
                "endmodule",
                "",
            ]
        )
        verilog_path.write_text("\n".join(lines_out), encoding="utf-8")
    except Exception as exc:
        print(f"[process_chart] 写入 ROM 失败: {exc}")
        return False

    return True


def main():
    base_dir = Path(__file__).resolve().parent.parent
    charts_dir = base_dir / "charts" / "Random"

    print("[1/3] 生成随机谱面 Random_test.txt ...")
    chart_path = generate_random_chart(charts_dir, name="Random_test", bpm=120, length_seconds=200, seed=42)
    if chart_path is None:
        return
    print(f"    ✓ 已生成: {chart_path}")

    print("[2/3] 校验谱面 ...")
    if not chart_check("Random_test", chart_path):
        return
    print("    ✓ 校验通过")

    print("[3/3] 生成 Verilog ROM (test_rom.v) ...")
    if not process_chart("Random_test", output_filename="test_rom.v"):
        return
    print(f"    ✓ 已输出: {base_dir / 'verilog' / 'test_rom.v'}")


if __name__ == "__main__":
    main()

