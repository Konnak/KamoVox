//! Reader for PyTorch zip checkpoints (`torch.save` since 1.6), limited to what
//! RVC model files contain: dicts, lists, numbers, strings and tensors.
//! Nothing is executed; unknown globals become inert values.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};
use std::rc::Rc;

use zip::ZipArchive;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
    I64,
}

impl DType {
    fn size(self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 | DType::I64 => 8,
        }
    }

    fn from_storage(name: &str) -> Option<Self> {
        Some(match name {
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::BF16,
            "FloatStorage" => DType::F32,
            "DoubleStorage" => DType::F64,
            "LongStorage" => DType::I64,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct TensorRef {
    pub key: String,
    pub dtype: DType,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Rc<RefCell<Vec<Value>>>),
    Tuple(Vec<Value>),
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    Global(String, String),
    Storage(String, DType),
    Tensor(TensorRef),
    Object(Box<Value>, Vec<Value>),
    Mark,
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(*b as i64),
            Value::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        match self {
            Value::Dict(d) => d.borrow().iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v.clone()),
            _ => None,
        }
    }

    pub fn items(&self) -> Vec<(Value, Value)> {
        match self {
            Value::Dict(d) => d.borrow().clone(),
            _ => Vec::new(),
        }
    }

    pub fn elements(&self) -> Vec<Value> {
        match self {
            Value::List(l) => l.borrow().clone(),
            Value::Tuple(t) => t.clone(),
            _ => Vec::new(),
        }
    }
}

fn err<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(msg.into())
}

struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
}

impl<'a> Unpickler<'a> {
    fn byte(&mut self) -> Result<u8, String> {
        let b = *self.data.get(self.pos).ok_or("pickle truncado")?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.data.len()).ok_or("pickle truncado")?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn line(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self.byte()? != b'\n' {}
        Ok(String::from_utf8_lossy(&self.data[start..self.pos - 1]).into_owned())
    }

    fn pop(&mut self) -> Result<Value, String> {
        self.stack.pop().ok_or_else(|| "pilha do pickle vazia".to_string())
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>, String> {
        let at = self.stack.iter().rposition(|v| matches!(v, Value::Mark)).ok_or("MARK ausente")?;
        let items = self.stack.split_off(at + 1);
        self.stack.pop();
        Ok(items)
    }

    fn top(&mut self) -> Result<&mut Value, String> {
        self.stack.last_mut().ok_or_else(|| "pilha do pickle vazia".to_string())
    }

    fn run(mut self) -> Result<Value, String> {
        loop {
            let op = self.byte()?;
            match op {
                0x80 => {
                    self.byte()?;
                }
                0x95 => {
                    self.take(8)?;
                }
                b'.' => return self.pop(),
                b'(' => self.stack.push(Value::Mark),
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'J' => {
                    let v = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
                    self.stack.push(Value::Int(v as i64));
                }
                b'K' => {
                    let v = self.byte()?;
                    self.stack.push(Value::Int(v as i64));
                }
                b'M' => {
                    let v = u16::from_le_bytes(self.take(2)?.try_into().unwrap());
                    self.stack.push(Value::Int(v as i64));
                }
                0x8a => {
                    let n = self.byte()? as usize;
                    let bytes = self.take(n)?;
                    let mut v: i64 = 0;
                    for (i, &b) in bytes.iter().enumerate().take(8) {
                        v |= (b as i64) << (8 * i);
                    }
                    if n > 0 && n < 8 && bytes[n - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * n);
                    }
                    self.stack.push(Value::Int(v));
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(Value::Float(v));
                }
                b'X' => {
                    let n = self.u32()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                0x8c => {
                    let n = self.byte()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                b'T' | b'B' => {
                    let n = self.u32()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(if op == b'T' { Value::Str(String::from_utf8_lossy(&b).into_owned()) } else { Value::Bytes(b) });
                }
                b'U' | b'C' => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(if op == b'U' { Value::Str(String::from_utf8_lossy(&b).into_owned()) } else { Value::Bytes(b) });
                }
                b'}' => self.stack.push(Value::Dict(Rc::default())),
                b']' => self.stack.push(Value::List(Rc::default())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'd' => {
                    let items = self.pop_mark()?;
                    let pairs = items.chunks(2).filter(|c| c.len() == 2).map(|c| (c[0].clone(), c[1].clone())).collect();
                    self.stack.push(Value::Dict(Rc::new(RefCell::new(pairs))));
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(Rc::new(RefCell::new(items))));
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85..=0x87 => {
                    let n = (op - 0x84) as usize;
                    if self.stack.len() < n {
                        return err("pilha do pickle vazia");
                    }
                    let items = self.stack.split_off(self.stack.len() - n);
                    self.stack.push(Value::Tuple(items));
                }
                b'q' => {
                    let k = self.byte()? as u32;
                    let v = self.top()?.clone();
                    self.memo.insert(k, v);
                }
                b'r' => {
                    let k = self.u32()?;
                    let v = self.top()?.clone();
                    self.memo.insert(k, v);
                }
                0x94 => {
                    let k = self.memo.len() as u32;
                    let v = self.top()?.clone();
                    self.memo.insert(k, v);
                }
                b'h' => {
                    let k = self.byte()? as u32;
                    self.stack.push(self.memo.get(&k).cloned().ok_or("memo ausente")?);
                }
                b'j' => {
                    let k = self.u32()?;
                    self.stack.push(self.memo.get(&k).cloned().ok_or("memo ausente")?);
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global(module, name));
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    self.stack.push(Value::Global(
                        module.as_str().unwrap_or_default().to_string(),
                        name.as_str().unwrap_or_default().to_string(),
                    ));
                }
                b'Q' => {
                    let pid = self.pop()?.elements();
                    // ('storage', <dtype class>, key, location, numel)
                    let dtype = match pid.get(1) {
                        Some(Value::Global(_, n)) => DType::from_storage(n),
                        _ => None,
                    };
                    let key = pid.get(2).and_then(|k| k.as_str()).map(str::to_string);
                    match (dtype, key) {
                        (Some(d), Some(k)) => self.stack.push(Value::Storage(k, d)),
                        _ => return err("tipo de tensor não suportado no modelo"),
                    }
                }
                b'R' | 0x81 => {
                    let args = self.pop()?.elements();
                    let callable = self.pop()?;
                    self.stack.push(call(callable, args)?);
                }
                b'b' => {
                    let state = self.pop()?;
                    // OrderedDict / Parameter state is irrelevant for weights.
                    let _ = state;
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    if let Value::Dict(d) = self.top()? {
                        d.borrow_mut().push((k, v));
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if let Value::Dict(d) = self.top()? {
                        let mut d = d.borrow_mut();
                        for c in items.chunks(2).filter(|c| c.len() == 2) {
                            d.push((c[0].clone(), c[1].clone()));
                        }
                    }
                }
                b'a' => {
                    let v = self.pop()?;
                    if let Value::List(l) = self.top()? {
                        l.borrow_mut().push(v);
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    if let Value::List(l) = self.top()? {
                        l.borrow_mut().extend(items);
                    }
                }
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    self.pop_mark()?;
                }
                b'2' => {
                    let v = self.top()?.clone();
                    self.stack.push(v);
                }
                0x8f => self.stack.push(Value::List(Rc::default())),
                0x90 => {
                    let items = self.pop_mark()?;
                    if let Value::List(l) = self.top()? {
                        l.borrow_mut().extend(items);
                    }
                }
                _ => return err(format!("opcode de pickle não suportado: 0x{op:02x}")),
            }
        }
    }
}

fn call(callable: Value, args: Vec<Value>) -> Result<Value, String> {
    let Value::Global(module, name) = &callable else {
        return Ok(Value::Object(Box::new(callable), args));
    };
    match (module.as_str(), name.as_str()) {
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
            let (key, dtype) = match args.first() {
                Some(Value::Storage(k, d)) => (k.clone(), *d),
                _ => return err("tensor sem armazenamento"),
            };
            let offset = args.get(1).and_then(Value::as_int).unwrap_or(0) as usize;
            let dims = |v: Option<&Value>| -> Vec<usize> {
                v.map(|v| v.elements().iter().filter_map(Value::as_int).map(|i| i as usize).collect()).unwrap_or_default()
            };
            let shape = dims(args.get(2));
            let stride = dims(args.get(3));
            Ok(Value::Tensor(TensorRef { key, dtype, offset, shape, stride }))
        }
        ("torch._utils", "_rebuild_parameter") | ("torch._utils", "_rebuild_parameter_with_state") => {
            Ok(args.into_iter().next().unwrap_or(Value::None))
        }
        ("collections", "OrderedDict") | ("builtins", "dict") => Ok(Value::Dict(Rc::default())),
        _ => Ok(Value::Object(Box::new(callable), args)),
    }
}

/// A loaded checkpoint: the pickled object tree plus access to tensor data.
pub struct Checkpoint {
    pub root: Value,
    archive: ZipArchive<Cursor<Vec<u8>>>,
    prefix: String,
}

impl Checkpoint {
    pub fn open(bytes: Vec<u8>) -> Result<Self, String> {
        let mut archive = ZipArchive::new(Cursor::new(bytes)).map_err(|_| "o arquivo não é um modelo PyTorch (.pth) válido".to_string())?;
        let pkl = (0..archive.len())
            .filter_map(|i| archive.name_for_index(i).map(str::to_string))
            .find(|n| n.ends_with("data.pkl"))
            .ok_or("data.pkl não encontrado no modelo")?;
        let prefix = pkl.trim_end_matches("data.pkl").to_string();
        let mut data = Vec::new();
        archive.by_name(&pkl).map_err(|e| e.to_string())?.read_to_end(&mut data).map_err(|e| e.to_string())?;
        let root = Unpickler { data: &data, pos: 0, stack: Vec::new(), memo: HashMap::new() }.run()?;
        Ok(Self { root, archive, prefix })
    }

    /// Reads a tensor as `f32`, honouring offset and strides.
    pub fn read_f32(&mut self, t: &TensorRef) -> Result<Vec<f32>, String> {
        let name = format!("{}data/{}", self.prefix, t.key);
        let mut raw = Vec::new();
        read_entry(&mut self.archive, &name, &mut raw)?;
        let size = t.dtype.size();
        let numel: usize = t.shape.iter().product();
        let convert = |b: &[u8]| match t.dtype {
            DType::F16 => half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
            DType::BF16 => half::bf16::from_le_bytes([b[0], b[1]]).to_f32(),
            DType::F32 => f32::from_le_bytes(b.try_into().unwrap()),
            DType::F64 => f64::from_le_bytes(b.try_into().unwrap()) as f32,
            DType::I64 => i64::from_le_bytes(b.try_into().unwrap()) as f32,
        };

        let mut contiguous = true;
        let mut expect = 1;
        for (dim, stride) in t.shape.iter().zip(&t.stride).rev() {
            if *dim > 1 && *stride != expect {
                contiguous = false;
            }
            expect *= dim;
        }
        if contiguous {
            let bytes = raw.get(t.offset * size..(t.offset + numel) * size).ok_or("tensor fora dos limites")?;
            return Ok(bytes.chunks_exact(size).map(convert).collect());
        }

        let mut out = Vec::with_capacity(numel);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..numel {
            let pos = t.offset + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>();
            let b = raw.get(pos * size..pos * size + size).ok_or("tensor fora dos limites")?;
            out.push(convert(b));
            for d in (0..idx.len()).rev() {
                idx[d] += 1;
                if idx[d] < t.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(out)
    }
}

fn read_entry<R: Read + Seek>(archive: &mut ZipArchive<R>, name: &str, out: &mut Vec<u8>) -> Result<(), String> {
    archive.by_name(name).map_err(|_| format!("dados do tensor ausentes: {name}"))?.read_to_end(out).map_err(|e| e.to_string())?;
    Ok(())
}
