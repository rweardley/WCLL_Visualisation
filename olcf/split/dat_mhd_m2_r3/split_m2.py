import os
import re
import struct
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
            "case_name": os.path.basename(fnek5000),   # NEW
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
        fh.Read_at_all(0, header_bytes)
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
        fh.Read_at_all(132, tagbuf)
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
        fh.Read_at_all(136, raw)
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


def field_blocks(h):
    out = []
    for idim in range(h["ngeom"]):
        out.append(("geom", idim))
    for idim in range(h["nvel"]):
        out.append(("vel", idim))
    for ivar in range(h["npres"]):
        out.append(("pres", ivar))
    for ivar in range(h["ntemp"]):
        out.append(("temp", ivar))
    for iscal in range(h["nscal"]):
        out.append(("scal", iscal))
    return out

def field_groups(h):
    """
    Layout groups in file order.
    For X/U/P/T, data are stored element-major within each group.
    For S, each scalar is its own group, also element-major.
    """
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
    """
    Return base offset for each field group.

    For a group with ncomp components, the storage is:
        elem0 comp0
        elem0 comp1
        ...
        elem0 comp(ncomp-1)
        elem1 comp0
        ...
    """
    base = 132 + 4 + 4 * h["nelgv"]
    bytes_elem = h["npel"] * h["wdsz"]

    offsets = {}
    cursor = base

    for key, ncomp in field_groups(h):
        offsets[key] = cursor
        cursor += h["nelgv"] * ncomp * bytes_elem

    return offsets, cursor


def byte_offsets(h):
    base = 132 + 4 + 4 * h["nelgv"]
    bytes_elem = h["npel"] * h["wdsz"]

    offsets = {}
    cursor = base

    for idim in range(h["ngeom"]):
        offsets[("geom", idim)] = cursor
        cursor += h["nelgv"] * bytes_elem

    for idim in range(h["nvel"]):
        offsets[("vel", idim)] = cursor
        cursor += h["nelgv"] * bytes_elem

    for ivar in range(h["npres"]):
        offsets[("pres", ivar)] = cursor
        cursor += h["nelgv"] * bytes_elem

    for ivar in range(h["ntemp"]):
        offsets[("temp", ivar)] = cursor
        cursor += h["nelgv"] * bytes_elem

    for iscal in range(h["nscal"]):
        offsets[("scal", iscal)] = cursor
        cursor += h["nelgv"] * bytes_elem

    return offsets, cursor


def read_selected_elements_mpiio(fh, h, emode, offsets, local_idx):
    real_dtype = dtype_from_header(h, emode)
    npel = h["npel"]
    bytes_elem = npel * h["wdsz"]

    out = {}

    for key, ncomp in field_groups(h):
        # shape: [nlocal, ncomp, npel]
        buf = np.empty((len(local_idx), ncomp, npel), dtype=real_dtype)
        field0 = offsets[key]

        for iloc, e in enumerate(local_idx):
            off = field0 + int(e) * ncomp * bytes_elem
            tmp = np.empty(ncomp * npel, dtype=real_dtype)
            fh.Read_at(off, tmp)
            buf[iloc, :, :] = tmp.reshape(ncomp, npel)

        out[key] = buf

    return out


def compute_counts_displs(nlocal):
    counts = comm.allgather(int(nlocal))
    displs = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        displs[1:] = np.cumsum(counts[:-1], dtype=np.int64)
    return np.asarray(counts, dtype=np.int64), displs


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

    # elmap
    base_elmap = 132 + 4
    my_elmap_off = base_elmap + int(displs[rank]) * 4
    if nlocal > 0:
        fh.Write_at_all(
            my_elmap_off,
            out_elmap_local.astype(np.dtype(emode + "i4"), copy=False),
        )
    else:
        fh.Write_at_all(my_elmap_off, np.empty(0, dtype=np.dtype(emode + "i4")))

    comm.Barrier()

    # payload
    npel = h_in["npel"]
    bytes_elem = npel * h_in["wdsz"]
    payload0 = 132 + 4 + 4 * nelgv_new
    cursor = payload0
    real_dtype = dtype_from_header(h_in, emode)

    for key, ncomp in field_groups(h_in):
        my_off = cursor + int(displs[rank]) * ncomp * bytes_elem
        buf = out_data[key]   # shape [nlocal, ncomp, npel]

        if nlocal > 0:
            fh.Write_at_all(my_off, buf.reshape(-1))
        else:
            fh.Write_at_all(my_off, np.empty(0, dtype=real_dtype))

        cursor += nelgv_new * ncomp * bytes_elem


def build_selector(select_list):
    """
    select_list example:
        {
            "file1": np.arange(1, 512),
            "file2": np.arange(512, 1024),
        }

    Returns
    -------
    selector : dict
        selector[name] = sorted unique global element ids (1-based)
    owner : dict
        owner[global_eid] = name

    Checks:
      - no duplicated ownership
      - element ids must be positive
    """
    selector = {}
    owner = {}

    for name, elems in select_list.items():
        arr = np.asarray(elems, dtype=np.int64).ravel()
        arr = np.unique(arr)

        if np.any(arr <= 0):
            raise ValueError(f"{name}: global element ids must be 1-based positive integers")

        selector[name] = arr

        for e in arr:
            e = int(e)
            if e in owner:
                raise ValueError(
                    f"Global element {e} appears in both '{owner[e]}' and '{name}'"
                )
            owner[e] = name

    return selector, owner


def masks_from_select_list(elmap, select_list):
    """
    For current file-local elmap, build one boolean mask per output file.
    elmap is 1-based global element ids stored in this input fld.
    """
    selector, owner = build_selector(select_list)

    masks = {}
    assigned = np.zeros(elmap.shape, dtype=bool)

    for name, elems in selector.items():
        mask = np.isin(elmap, elems)
        masks[name] = mask
        assigned |= mask

    return masks, assigned, selector, owner


def local_indices_from_mask(mask):
    idx = np.flatnonzero(mask)
    return idx[rank::size]


def read_many_selected_groups(fh, h, emode, offsets, masks):
    """
    Read all requested groups from current input file.

    Returns
    -------
    group_out : dict
        group_out[name] = {
            "local_idx": ...,
            "elmap_local": ...,
            "data": {field_key: array[nlocal, npel]}
        }
    """
    group_out = {}

    for name, mask in masks.items():
        local_idx = local_indices_from_mask(mask)
        data = read_selected_elements_mpiio(fh, h, emode, offsets, local_idx)
        elmap_local = None
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
    masks, assigned, selector, owner = masks_from_select_list(elmap, select_list)
    group_out = read_many_selected_groups(fh, h, emode, offsets, masks)
    tread_payload = MPI.Wtime() - t1

    fh.Close()

    t1 = MPI.Wtime()
    for name in group_out:
        local_idx = group_out[name]["local_idx"]
        group_out[name]["elmap_local"] = elmap[local_idx].copy()
    tprepare = MPI.Wtime() - t1

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


def write_groups_for_one_input(fname_in, out_dir, h, emode, group_out, timers=None):
    """
    Write one output file per group for this input fld.

    output name:
      <out_dir>/<group>_<basename(fname_in)>

    Example:
      input : /path/channel0.f00001
      group : file1
      output: /out/path/file1_channel0.f00001
    """
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


def debug_print_selection(elmap, select_list, max_print=8):
    if rank == 0:
        print("selection check:", flush=True)
        for name, elems in select_list.items():
            mask = np.isin(elmap, elems)
            picked = elmap[mask]
            head = picked[:max_print]
            tail = picked[-max_print:] if picked.size > max_print else np.array([], dtype=picked.dtype)

            print(f"  {name}: n={picked.size}", flush=True)
            if picked.size > 0:
                print(f"    first: {head}", flush=True)
                if tail.size > 0:
                    print(f"    last : {tail}", flush=True)

def write_nek5000_meta_file(meta, out_dir, prefix, numtimesteps=None, firsttimestep=None):
    """
    Write one .nek5000 metadata file for one split output family.

    Example output:
        out_dir/file1_channel.nek5000

    with template:
        file1_channel%01d.f%05d
    """
    ensure_dir(out_dir)

    if numtimesteps is None:
        numtimesteps = meta["numtimesteps"]
    if firsttimestep is None:
        firsttimestep = meta["firsttimestep"]

    in_template = meta["file_template"]
    out_template = f"{prefix}_{in_template}"

    # use the original case basename, but add prefix
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
    """
    Write one .nek5000 file per output family.
    """
    out = {}
    for prefix in select_list.keys():
        out[prefix] = write_nek5000_meta_file(meta, out_dir, prefix)
    return out

# ----------------------------------------------------------------------
def main(fdr, case, out_fdr, select_list):

    case_path = os.path.abspath(os.path.join(fdr, case))
    case_dir = os.path.dirname(case_path)
    out_dir = os.path.abspath(out_fdr)

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
        for name, elems in select_list.items():
            arr = np.asarray(elems, dtype=np.int64).ravel()
            print(f"  {name:>12s} : n={arr.size}  [{arr.min()}, {arr.max()}]", flush=True)
        print("", flush=True)

    # NEW: write one .nek5000 metadata file for each output family
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
            print(f"input done", flush=True)

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

if __name__ == "__main__":
# 1) helium/water TBM:         eg <= 20,337,300
# 2) Pb:        20,337,300 < eg <= 79,602,344
# 3) water shield: 79,602,344 < eg <= 85,626,260
# 4) steel TBM:   85,626,260 < eg <= 139,809,796
# 5) steel shield:139,809,796 < eg <= 146,347,326

    fdr = "./"
    out_fdr = "./dat_split"

    select_list = {
        "water_tbm": np.arange(1, 20337300 + 1, dtype=np.int64),
        "pbli":      np.arange(20337300 + 1, 79602344 + 1, dtype=np.int64),
        "water_sh":  np.arange(79602344 + 1, 85626260 + 1, dtype=np.int64),
        "steel_tbm": np.arange(85626260 + 1, 139809796 + 1, dtype=np.int64),
        "steel_sh":  np.arange(139809796 + 1, 146347326 + 1, dtype=np.int64),
    }

    case = "pink.nek5000"
    main(fdr, case, out_fdr, select_list)

    case = "mhdpink.nek5000"
    main(fdr, case, out_fdr, select_list)

