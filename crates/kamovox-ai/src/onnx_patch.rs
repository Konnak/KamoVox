//! Minimal protobuf surgery on ONNX files: exposes an internal tensor as an extra
//! float graph output, so one embedder file serves both RVC v1 (layer 9) and v2.

const MODEL_GRAPH: u64 = 7;
const GRAPH_OUTPUT: u64 = 12;

fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *buf.get(*pos)?;
        *pos += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn write_bytes_field(out: &mut Vec<u8>, field: u64, data: &[u8]) {
    write_varint(out, field << 3 | 2);
    write_varint(out, data.len() as u64);
    out.extend_from_slice(data);
}

fn float_value_info(name: &str) -> Vec<u8> {
    // TypeProto { tensor_type: { elem_type: FLOAT } }
    let mut tensor = Vec::new();
    write_varint(&mut tensor, 1 << 3);
    write_varint(&mut tensor, 1);
    let mut ty = Vec::new();
    write_bytes_field(&mut ty, 1, &tensor);
    let mut info = Vec::new();
    write_bytes_field(&mut info, 1, name.as_bytes());
    write_bytes_field(&mut info, 2, &ty);
    info
}

/// Copies every top-level field of `msg`, letting `edit` replace length-delimited ones.
fn rewrite(msg: &[u8], out: &mut Vec<u8>, mut edit: impl FnMut(u64, &[u8], &mut Vec<u8>) -> bool) -> Option<()> {
    let mut pos = 0;
    while pos < msg.len() {
        let start = pos;
        let key = read_varint(msg, &mut pos)?;
        let (field, wire) = (key >> 3, key & 7);
        match wire {
            0 => {
                read_varint(msg, &mut pos)?;
            }
            1 => pos += 8,
            5 => pos += 4,
            2 => {
                let len = read_varint(msg, &mut pos)? as usize;
                let body = msg.get(pos..pos.checked_add(len)?)?;
                pos += len;
                if edit(field, body, out) {
                    continue;
                }
            }
            _ => return None,
        }
        out.extend_from_slice(msg.get(start..pos)?);
    }
    Some(())
}

/// Returns a copy of `model` whose only graph output is the float tensor `tensor_name`,
/// so ONNX Runtime skips every node after it.
pub fn replace_outputs(model: &[u8], tensor_name: &str) -> Result<Vec<u8>, String> {
    let bad = || "arquivo ONNX inválido".to_string();
    let mut out = Vec::with_capacity(model.len() + 256);
    let mut patched = false;
    rewrite(model, &mut out, |field, body, out| {
        if field != MODEL_GRAPH || patched {
            return false;
        }
        let mut graph = Vec::with_capacity(body.len() + 128);
        let ok = rewrite(body, &mut graph, |f, _, _| f == GRAPH_OUTPUT);
        if ok.is_none() {
            return false;
        }
        write_bytes_field(&mut graph, GRAPH_OUTPUT, &float_value_info(tensor_name));
        write_bytes_field(out, MODEL_GRAPH, &graph);
        patched = true;
        true
    })
    .ok_or_else(bad)?;
    if patched {
        Ok(out)
    } else {
        Err(bad())
    }
}
