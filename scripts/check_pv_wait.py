#!/usr/bin/env python3
"""Build-time guard for PA_PV_PARTIAL_WAIT.

The PV batch waits with a *partial* lgkmcnt: the j-th mfma only waits for
lgkmcnt(PV_BATCH-1-j), which is correct only while the LGKM queue holds
nothing but this batch's tr_loads.  LDS retires in order, but SMEM does not --
so a single s_load placed between two tr_loads makes the partial count return
early, the mfma reads stale V, and the damage lands on the NoPE columns only
(P and the RoPE PV stay bit-exact).  Three unrelated scheduling changes have
tripped this before; `_tr_load` is inline asm the compiler cannot see, so
nothing in the source prevents it.

This scans the built ISA and fails the build if the condition can occur.
Usage: check_pv_wait.py <disassembly.s>
"""
import re, sys

def main(path):
    ins = []
    for l in open(path):
        t = re.sub(r'//.*', '', l).strip()
        if t and not t.startswith(('0000', 'Disas')) and 'file format' not in t:
            ins.append(t)
    # Anchor on what the PV batch actually looks like, not on "a descending
    # lgkmcnt run" -- the compiler emits those for QK/RoPE too (32 of them with
    # the feature OFF), and a guard that cries wolf gets ignored.
    #
    # The PV partial wait is: s_waitcnt lgkmcnt(k>0) immediately feeding a
    # v_mfma_scale_..._f8f6f4, whose V operand came from ds_read_b64_tr_b8.
    # Scan back to the batch start (the previous full drain) and fail if any
    # SMEM sits in that window -- s_load retires out of order, so the partial
    # count would return before this o_tile's tr_loads have landed.
    W = re.compile(r's_waitcnt\s+lgkmcnt\((\d+)\)')
    seqs = bad = 0
    for i, t0 in enumerate(ins):
        m0 = W.match(t0)
        if not m0 or int(m0.group(1)) == 0:
            continue
        if not any(ins[k].startswith('v_mfma_scale')
                   for k in range(i + 1, min(i + 4, len(ins)))):
            continue                      # not a PV partial wait
        start, saw_tr = None, False
        for k in range(i - 1, max(-1, i - 200), -1):
            if ins[k].startswith('ds_read_b64_tr_b8'):
                saw_tr = True
            if W.match(ins[k]) and W.match(ins[k]).group(1) == '0':
                start = k
                break
        if not saw_tr:
            continue                      # this partial wait is not on tr_loads
        start = start if start is not None else max(0, i - 200)
        smem = [ins[k] for k in range(start, i)
                if ins[k].split()[0].startswith(('s_load', 's_buffer_load'))]
        seqs += 1
        if smem:
            bad += 1
            if bad <= 3:
                print(f"  !! PV partial-wait @ 指令 {i}: batch 窗口内混入 "
                      f"{smem[0][:58]}", file=sys.stderr)
    if seqs == 0:
        print(f"  [pv-wait guard] 未发现 partial-wait 序列(功能关闭),跳过")
        return 0
    if bad:
        print(f"  [pv-wait guard] 失败: {bad}/{seqs} 处序列中混入 SMEM "
              f"-> partial lgkmcnt 会提前返回,NoPE 列将读到脏 V", file=sys.stderr)
        return 1
    print(f"  [pv-wait guard] 通过: {seqs} 处 partial-wait 序列,LGKM 队列中无 SMEM")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
