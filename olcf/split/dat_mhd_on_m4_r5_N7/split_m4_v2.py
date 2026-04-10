import os
import sys
import re
import gc
import struct
import socket
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def exitti(ierr=0, s=""):
    ierr = comm.allreduce(ierr, op=MPI.MAX)
    if ierr > 0:
        if rank == 0:
            print(f"EXIT {ierr} {s}", flush=True)
        raise SystemExit(1)


def fmt_dt(dt):
    if dt < 1e-3:
        return f"{dt*1e6:.1f} us"
    elif dt < 1.0:
        return f"{dt*1e3:.3f} ms"
    else:
        return f"{dt:.3f} s"


def mpi_max(dt):
    return comm.allreduce(dt, op=MPI.MAX)


def ensure_dir(path):
    if rank == 0:
        os.makedirs(path, exist_ok=True)
    comm.Barrier()


def _read_proc_status_kb():
    """
    Return current RSS and peak RSS in kB from /proc/self/status.
    Linux only.
    """
    rss_kb = -1
    hwm_kb = -1

    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kb = int(line.split()[1])

    return rss_kb, hwm_kb


def fmt_mem_kb(kb):
    if kb < 0:
        return "n/a"
    gb = kb / 1024.0 / 1024.0
    if gb >= 1.0:
        return f"{gb:.3f} GB"
    mb = kb / 1024.0
    return f"{mb:.1f} MB"


def print_mem_usage(tag=""):
    """
    Print current and peak memory usage, max across ranks.
    """
    rss_kb, hwm_kb = _read_proc_status_kb()

    rss_max_kb = comm.allreduce(rss_kb, op=MPI.MAX)
    hwm_max_kb = comm.allreduce(hwm_kb, op=MPI.MAX)
    rss_sum_kb = comm.allreduce(rss_kb, op=MPI.SUM)

    if rank == 0:
        print(
            f"  memory      {tag}"
            f"  rss_max={fmt_mem_kb(rss_max_kb)}"
            f"  rss_sum={fmt_mem_kb(rss_sum_kb)}"
            f"  hwm_max={fmt_mem_kb(hwm_max_kb)}",
            flush=True,
        )

def print_run_info(fdr, case, out_fdr, select_list):
    host = socket.gethostname()

    nnode = comm.allreduce(1 if comm.rank == 0 else 0, op=MPI.SUM)
    # better node count from hostnames
    hosts = comm.gather(host, root=0)

    if rank == 0:
        uniq_hosts = sorted(set(hosts))
        nnode = len(uniq_hosts)

        print("", flush=True)
        print("=== Run info ===", flush=True)
        print(f"script      : {os.path.basename(sys.argv[0])}", flush=True)
        print(f"cwd         : {os.getcwd()}", flush=True)
        print(f"case dir    : {os.path.abspath(fdr)}", flush=True)
        print(f"case file   : {case}", flush=True)
        print(f"out dir     : {os.path.abspath(out_fdr)}", flush=True)
        print(f"mpi ranks   : {size}", flush=True)
        print(f"nodes       : {nnode}", flush=True)
        print(f"ranks/node  : {size // nnode if nnode > 0 else 'n/a'}", flush=True)
        print(f"host[0]     : {uniq_hosts[0] if uniq_hosts else 'n/a'}", flush=True)

        # slurm info if available
        print(f"job id      : {os.environ.get('SLURM_JOB_ID', 'n/a')}", flush=True)
        print(f"job name    : {os.environ.get('SLURM_JOB_NAME', 'n/a')}", flush=True)
        print(f"nodelist    : {os.environ.get('SLURM_JOB_NODELIST', 'n/a')}", flush=True)
        print(f"ntasks      : {os.environ.get('SLURM_NTASKS', 'n/a')}", flush=True)
        print(f"tasks/node  : {os.environ.get('SLURM_NTASKS_PER_NODE', 'n/a')}", flush=True)
        print("================", flush=True)
        print("", flush=True)



def fld_name_from_template(meta, ifile, file_id=0):
    tmpl = meta["file_template"]
    nfmt = tmpl.count("%")

    if nfmt == 1:
        return tmpl % ifile
    elif nfmt == 2:
        return tmpl % (file_id, ifile)
    else:
        raise ValueError(f"Unsupported template: {tmpl}")


def read_nek5000_meta(fnek5000):
    ierr = 0
    meta = None

    if rank == 0:
        try:
            with open(fnek5000, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"{fnek5000} file not found!", flush=True)
            ierr = 1
            lines = None

    exitti(ierr, f"cannot open {fnek5000}")

    if rank == 0:
        cname = lines[0].split()[1].split('%')[0]
        file_template = lines[0].split()[1].strip()

        fmts = re.findall(r"%(\d+)d", file_template)
        if len(fmts) == 0:
            raise ValueError(f"Cannot parse file template: {file_template}")

        num_zero = int(fmts[-1])

        meta = {
            "cname": cname,
            "case_name": os.path.basename(fnek5000),
            "file_template": file_template,
            "firsttimestep": int(lines[1].split()[1]),
            "numtimesteps": int(lines[2].split()[1]),
            "num_zero": num_zero,
        }

        print(meta, flush=True)

    return comm.bcast(meta, root=0)


def read_header_bytes(fh):
    header_bytes = bytearray(132)
    if rank == 0:
        fh.Read_at(0, header_bytes)
    comm.Bcast(header_bytes, root=0)

    header = bytes(header_bytes).split()
    if len(header) < 12:
        raise IOError("Header too short")
    return header_bytes, header


def parse_header_tokens(header):
    wdsz = int(header[1])
    orders = tuple(int(x) for x in header[2:5])
    nelgt = int(header[5])
    nelgv = int(header[6])
    time = float(header[7])
    istep = int(header[8])
    fid = int(header[9])
    nfileo = int(header[10])
    rdcode = header[11].decode() if isinstance(header[11], bytes) else header[11]

    ndim = 3 if orders[2] > 1 else 2
    npel = orders[0] * orders[1] * orders[2]

    ngeom = ndim if "X" in rdcode else 0
    nvel = ndim if "U" in rdcode else 0
    npres = 1 if "P" in rdcode else 0
    ntemp = 1 if "T" in rdcode else 0

    if "S" in rdcode:
        iscal = rdcode.index("S")
        nscal = int(rdcode[iscal + 1 :])
    else:
        nscal = 0

    return {
        "wdsz": wdsz,
        "orders": orders,
        "nelgt": nelgt,
        "nelgv": nelgv,
        "time": time,
        "istep": istep,
        "fid": fid,
        "nfileo": nfileo,
        "rdcode": rdcode,
        "ndim": ndim,
        "npel": npel,
        "ngeom": ngeom,
        "nvel": nvel,
        "npres": npres,
        "ntemp": ntemp,
        "nscal": nscal,
    }


def detect_endian_and_read_elmap(fh, nelgv):
    tagbuf = bytearray(4)
    if rank == 0:
        fh.Read_at(132, tagbuf)
    comm.Bcast(tagbuf, root=0)

    tag_l = struct.unpack("<f", tagbuf)[0]
    tag_b = struct.unpack(">f", tagbuf)[0]

    if abs(tag_l - 6.54321) < 1e-5:
        emode = "<"
    elif abs(tag_b - 6.54321) < 1e-5:
        emode = ">"
    else:
        raise ValueError("Could not determine endianness")

    raw = bytearray(4 * nelgv)
    if rank == 0:
        fh.Read_at(136, raw)
    comm.Bcast(raw, root=0)

    elmap = np.frombuffer(raw, dtype=np.dtype(emode + "i4")).copy()
    return emode, elmap


def make_header_bytes(h, nelgt_new, nelgv_new, fid=0, nfileo=1):
    header = (
        "#std %1i %2i %2i %2i %10i %10i %20.13E %9i %6i %6i %s"
        % (
            h["wdsz"],
            h["orders"][0],
            h["orders"][1],
            h["orders"][2],
            nelgt_new,
            nelgv_new,
            h["time"],
            h["istep"],
            fid,
            nfileo,
            h["rdcode"],
        )
    )
    return header.ljust(132).encode("utf-8")


def dtype_from_header(h, emode):
    if h["wdsz"] == 4:
        return np.dtype(emode + "f4")
    elif h["wdsz"] == 8:
        return np.dtype(emode + "f8")
    raise ValueError(f"Unsupported wdsz={h['wdsz']}")


def field_groups(h):
    groups = []

    if h["ngeom"] > 0:
        groups.append(("geom", h["ngeom"]))
    if h["nvel"] > 0:
        groups.append(("vel", h["nvel"]))
    if h["npres"] > 0:
        groups.append(("pres", h["npres"]))
    if h["ntemp"] > 0:
        groups.append(("temp", h["ntemp"]))
    for iscal in range(h["nscal"]):
        groups.append((("scal", iscal), 1))

    return groups


def group_offsets(h):
    base = 132 + 4 + 4 * h["nelgv"]
    bytes_elem = h["npel"] * h["wdsz"]

    offsets = {}
    cursor = base

    for key, ncomp in field_groups(h):
        offsets[key] = cursor
        cursor += h["nelgv"] * ncomp * bytes_elem

    return offsets, cursor


def compute_counts_displs(nlocal):
    counts = comm.allgather(int(nlocal))
    displs = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        displs[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return np.asarray(counts, dtype=np.int64), displs


def masks_from_select_list(elmap, select_list):
    masks = {}
    assigned = np.zeros(elmap.shape, dtype=bool)

    for name, sel in select_list.items():
        if isinstance(sel, tuple) and len(sel) == 2:
            lo, hi = sel
            mask = (elmap >= lo) & (elmap <= hi)
        else:
            arr = np.asarray(sel, dtype=np.int64).ravel()
            if np.any(arr <= 0):
                raise ValueError(f"{name}: global element ids must be positive")
            mask = np.isin(elmap, arr)

        masks[name] = mask
        assigned |= mask

    return masks, assigned


def local_indices_from_mask(mask):
    idx = np.flatnonzero(mask)
    n = len(idx)

    i0 = (n * rank) // size
    i1 = (n * (rank + 1)) // size
    return idx[i0:i1]


def split_runs(idx):
    if len(idx) == 0:
        return []

    runs = []
    s = int(idx[0])
    p = int(idx[0])

    for x in idx[1:]:
        x = int(x)
        if x == p + 1:
            p = x
        else:
            runs.append((s, p + 1))
            s = p = x

    runs.append((s, p + 1))
    return runs


def read_selected_elements_mpiio(fh, h, emode, offsets, local_idx):
    real_dtype = dtype_from_header(h, emode)
    npel = h["npel"]
    bytes_elem = npel * h["wdsz"]
    runs = split_runs(local_idx)

    out = {}

    for key, ncomp in field_groups(h):
        field0 = offsets[key]
        buf = np.empty((len(local_idx), ncomp, npel), dtype=real_dtype)

        pos = 0
        for i0, i1 in runs:
            nr = i1 - i0
            tmp = np.empty(nr * ncomp * npel, dtype=real_dtype)
            off = field0 + i0 * ncomp * bytes_elem
            fh.Read_at(off, tmp)
            buf[pos:pos + nr, :, :] = tmp.reshape(nr, ncomp, npel)
            pos += nr

        out[key] = buf

    return out


def read_many_selected_groups(fh, h, emode, offsets, masks):
    group_out = {}

    for name, mask in masks.items():
        local_idx = local_indices_from_mask(mask)
        data = read_selected_elements_mpiio(fh, h, emode, offsets, local_idx)
        group_out[name] = {
            "local_idx": local_idx,
            "data": data,
        }

    return group_out


def process_one_file_with_select_list(fname_in, select_list):
    t0 = MPI.Wtime()

    fh = MPI.File.Open(comm, fname_in, MPI.MODE_RDONLY)

    t1 = MPI.Wtime()
    _header_bytes, header = read_header_bytes(fh)
    h = parse_header_tokens(header)
    emode, elmap = detect_endian_and_read_elmap(fh, h["nelgv"])
    offsets, payload_end = group_offsets(h)
    tread_meta = MPI.Wtime() - t1

    t1 = MPI.Wtime()
    masks, assigned = masks_from_select_list(elmap, select_list)
    group_out = read_many_selected_groups(fh, h, emode, offsets, masks)
    tread_payload = MPI.Wtime() - t1
    print_mem_usage(tag="after_read")

    fh.Close()

    t1 = MPI.Wtime()
    for name in group_out:
        local_idx = group_out[name]["local_idx"]
        group_out[name]["elmap_local"] = elmap[local_idx].copy()
    tprepare = MPI.Wtime() - t1
    print_mem_usage(tag="after_prepare")

    ttotal = MPI.Wtime() - t0

    timers = {
        "read_meta": mpi_max(tread_meta),
        "read_payload": mpi_max(tread_payload),
        "prepare": mpi_max(tprepare),
        "total_process": mpi_max(ttotal),
    }

    info = {
        "nelgv": h["nelgv"],
        "rdcode": h["rdcode"],
        "payload_end": payload_end,
        "counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "nunassigned": int((~assigned).sum()),
    }

    return h, emode, group_out, timers, info


def write_split_file_mpiio(fname_out, h_in, emode, out_elmap_local, out_data):
    nlocal = len(out_elmap_local)
    counts, displs = compute_counts_displs(nlocal)
    nelgv_new = int(np.sum(counts))
    nelgt_new = nelgv_new

    if rank == 0:
        header_bytes = make_header_bytes(h_in, nelgt_new, nelgv_new, fid=0, nfileo=1)
        tag = np.array([6.54321], dtype=np.dtype(emode + "f4"))
    else:
        header_bytes = None
        tag = None

    fh = MPI.File.Open(comm, fname_out, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    fh.Set_size(0)

    if rank == 0:
        fh.Write_at(0, np.frombuffer(header_bytes, dtype=np.uint8))
        fh.Write_at(132, tag)

    comm.Barrier()

    base_elmap = 132 + 4
    my_elmap_off = base_elmap + int(displs[rank]) * 4
    arr_i4 = np.dtype(emode + "i4")
    if nlocal > 0:
        fh.Write_at_all(my_elmap_off, out_elmap_local.astype(arr_i4, copy=False))
    else:
        fh.Write_at_all(my_elmap_off, np.empty(0, dtype=arr_i4))

    comm.Barrier()

    npel = h_in["npel"]
    bytes_elem = npel * h_in["wdsz"]
    payload0 = 132 + 4 + 4 * nelgv_new
    cursor = payload0
    real_dtype = dtype_from_header(h_in, emode)

    for key, ncomp in field_groups(h_in):
        my_off = cursor + int(displs[rank]) * ncomp * bytes_elem
        buf = out_data[key]

        if nlocal > 0:
            fh.Write_at_all(my_off, buf.reshape(-1))
        else:
            fh.Write_at_all(my_off, np.empty(0, dtype=real_dtype))

        cursor += nelgv_new * ncomp * bytes_elem

    fh.Close()


def write_groups_for_one_input(fname_in, out_dir, h, emode, group_out, timers=None):
    ensure_dir(out_dir)

    base = os.path.basename(fname_in)
    write_times = {}

    for name, grp in group_out.items():
        out_elmap_local = grp["elmap_local"]
        out_data = grp["data"]

        fname_out = os.path.join(out_dir, f"{name}_{base}")
        nout = comm.allreduce(len(out_elmap_local), op=MPI.SUM)

        if rank == 0:
            print(f"  write start  {fname_out}   nelem={nout}", flush=True)

        tw0 = MPI.Wtime()
        write_split_file_mpiio(fname_out, h, emode, out_elmap_local, out_data)
        tw = mpi_max(MPI.Wtime() - tw0)
        write_times[name] = tw

        if rank == 0:
            print(f"  write done   {fname_out}   dt={fmt_dt(tw)}", flush=True)

    if timers is not None:
        timers["write"] = write_times

    return write_times


def write_nek5000_meta_file(meta, out_dir, prefix, numtimesteps=None, firsttimestep=None):
    ensure_dir(out_dir)

    if numtimesteps is None:
        numtimesteps = meta["numtimesteps"]
    if firsttimestep is None:
        firsttimestep = meta["firsttimestep"]

    in_template = meta["file_template"]
    out_template = f"{prefix}_{in_template}"

    case_base = os.path.basename(meta.get("case_name", "channel.nek5000"))
    if case_base.endswith(".nek5000"):
        case_base = case_base[:-8]

    meta_name = f"{prefix}_{case_base}.nek5000"
    meta_path = os.path.join(out_dir, meta_name)

    if rank == 0:
        print(f"meta start  : {meta_path}", flush=True)
        with open(meta_path, "w") as f:
            f.write(f"filetemplate: {out_template}\n")
            f.write(f"firsttimestep: {firsttimestep}\n")
            f.write(f"numtimesteps: {numtimesteps}\n")
        print(f"meta done   : {meta_path}", flush=True)

    comm.Barrier()
    return meta_path


def write_all_split_nek5000_meta(meta, out_dir, select_list):
    out = {}
    for prefix in select_list.keys():
        out[prefix] = write_nek5000_meta_file(meta, out_dir, prefix)
    return out


def main(fdr, case, out_fdr, select_list):
    case_path = os.path.abspath(os.path.join(fdr, case))
    case_dir = os.path.dirname(case_path)
    out_dir = os.path.abspath(out_fdr)

    print_run_info(fdr, case, out_fdr, select_list)

    meta = read_nek5000_meta(case_path)

    ensure_dir(out_dir)

    if rank == 0:
        print("", flush=True)
        print("=== Split fld setup ===", flush=True)
        print(f"case_path : {case_path}", flush=True)
        print(f"case_dir  : {case_dir}", flush=True)
        print(f"out_dir   : {out_dir}", flush=True)
        print(f"template  : {meta['file_template']}", flush=True)
        print(f"nsteps    : {meta['numtimesteps']}", flush=True)
        print("groups    :", flush=True)
        for name, sel in select_list.items():
            if isinstance(sel, tuple) and len(sel) == 2:
                lo, hi = sel
                print(f"  {name:>12s} : [{lo}, {hi}]  n={hi - lo + 1}", flush=True)
            else:
                arr = np.asarray(sel, dtype=np.int64).ravel()
                print(f"  {name:>12s} : n={arr.size}  [{arr.min()}, {arr.max()}]", flush=True)
        print("", flush=True)

    write_all_split_nek5000_meta(meta, out_dir, select_list)

    for i in range(meta["numtimesteps"]):
        ifile = meta["firsttimestep"] + i
        fname_rel = fld_name_from_template(meta, ifile, file_id=0)
        fname_in = os.path.join(case_dir, fname_rel)

        if rank == 0:
            print("--------------------------------------------------", flush=True)
            print(f"input start : {fname_in}", flush=True)

        t0 = MPI.Wtime()

        h, emode, group_out, timers, info = process_one_file_with_select_list(
            fname_in, select_list
        )

        if rank == 0:
            print(
                f"  input info  nelgv={info['nelgv']}  rdcode={info['rdcode']}  payload_end={info['payload_end']}",
                flush=True,
            )
            for name, cnt in info["counts"].items():
                print(f"  selected    {name:>12s} : {cnt}", flush=True)
            if info["nunassigned"] > 0:
                print(f"  unassigned  {'':>12s} : {info['nunassigned']}", flush=True)

            print(
                f"  timer       read_meta={fmt_dt(timers['read_meta'])}  "
                f"read_payload={fmt_dt(timers['read_payload'])}  "
                f"prepare={fmt_dt(timers['prepare'])}  "
                f"process={fmt_dt(timers['total_process'])}",
                flush=True,
            )
            print("input done", flush=True)

        write_times = write_groups_for_one_input(
            fname_in=fname_in,
            out_dir=out_dir,
            h=h,
            emode=emode,
            group_out=group_out,
            timers=timers,
        )

        ttot = mpi_max(MPI.Wtime() - t0)

        if rank == 0:
            if len(write_times) > 0:
                wmsg = "  ".join([f"{k}={fmt_dt(v)}" for k, v in write_times.items()])
                print(f"  timer       write: {wmsg}", flush=True)
            print(f"  total      {fmt_dt(ttot)}", flush=True)
            print("", flush=True)

        # free large per-file objects before next file
        del group_out
        del timers
        del info
        del h
        del emode
        del write_times
        gc.collect()

        print_mem_usage(tag="after_free")
        if rank == 0:
            print("", flush=True)

if __name__ == "__main__":
    if rank == 0:
        print(sys.argv[0], flush=True)

    fdr = "./"
    out_fdr = "./dat_split"

#    fdr = "./"
#    select_list = {
#        "file1": (1, 512),
#        "file2": (513, 1024),
#    }
#    case = "channel.nek5000"
#    main(fdr, case, out_fdr, select_list)

    # range-based selectors: much smaller memory footprint
    select_list = {
        "water_tbm":  (1,         38733372),
        "pbli":       (38733373,  140869280),
        "water_sh":   (140869281, 151935592),
        "steel_tbm":  (151935593, 245637464),
        "steel_sh":   (245637465, 257676220),
    }

    case = "pink.nek5000"
    main(fdr, case, out_fdr, select_list)

    case = "mhdpink.nek5000"
    main(fdr, case, out_fdr, select_list)

    # --- ending summary ---
    if rank == 0:
        print("==================================================", flush=True)
        print("=== Split complete ===", flush=True)
        print(f"out_dir     : {out_dir}", flush=True)
        print("status      : OK", flush=True)
        print("==================================================", flush=True)
        print("", flush=True)
