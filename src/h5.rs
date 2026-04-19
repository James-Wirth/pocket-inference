//! Minimal HDF5 subset parser for Keras 3 `.keras` weight files.
//!
//! Supports only the narrow format h5py produces by default for Keras:
//! superblock v0, object header v1, symbol-table groups (B-tree v1 + local
//! heap), dataspace v1, datatype v1 (IEEE_F32LE), data layout v3 contiguous.
//! No attributes, compression, chunking, or non-f32 dtypes.

use thiserror::Error;

const SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1a, b'\n'];

#[derive(Error, Debug)]
pub enum H5Error {
    #[error("not an HDF5 file (signature missing)")]
    InvalidSignature,
    #[error("unsupported superblock version {0} (expected 0)")]
    UnsupportedSuperblock(u8),
    #[error("unsupported size of offsets {0} (expected 8)")]
    UnsupportedOffsetSize(u8),
    #[error("unsupported size of lengths {0} (expected 8)")]
    UnsupportedLengthSize(u8),
    #[error("unsupported object header version {0} (expected 1)")]
    UnsupportedObjectHeader(u8),
    #[error("unsupported dataspace version {0} (expected 1)")]
    UnsupportedDataspace(u8),
    #[error("unsupported datatype (only IEEE_F32LE supported)")]
    UnsupportedDatatype,
    #[error("unsupported data layout class {0} (only contiguous supported)")]
    UnsupportedLayoutClass(u8),
    #[error("unsupported data layout version {0} (expected 3)")]
    UnsupportedLayoutVersion(u8),
    #[error("path not found: {0}")]
    NotFound(String),
    #[error("object header missing required message: {0}")]
    MissingMessage(&'static str),
    #[error("truncated HDF5 data")]
    Truncated,
    #[error("malformed HDF5: {0}")]
    Malformed(String),
}

pub type Result<T> = std::result::Result<T, H5Error>;

pub struct H5Reader<'a> {
    bytes: &'a [u8],
    root_oh: u64,
}

impl<'a> H5Reader<'a> {
    pub fn open(bytes: &'a [u8]) -> Result<Self> {
        let mut base = 0usize;
        loop {
            if base + SIGNATURE.len() > bytes.len() {
                return Err(H5Error::InvalidSignature);
            }
            if bytes[base..base + SIGNATURE.len()] == SIGNATURE {
                break;
            }
            base = if base == 0 { 512 } else { base * 2 };
        }

        let mut c = Cursor::new(bytes, base + SIGNATURE.len());
        let sb_ver = c.u8()?;
        if sb_ver != 0 {
            return Err(H5Error::UnsupportedSuperblock(sb_ver));
        }
        c.skip(3)?;
        let _shmsg_ver = c.u8()?;
        let off_size = c.u8()?;
        let len_size = c.u8()?;
        if off_size != 8 {
            return Err(H5Error::UnsupportedOffsetSize(off_size));
        }
        if len_size != 8 {
            return Err(H5Error::UnsupportedLengthSize(len_size));
        }
        c.skip(1)?;
        c.skip(2)?;
        c.skip(2)?;
        c.skip(4)?;
        c.skip(8)?;
        c.skip(8)?;
        c.skip(8)?;
        c.skip(8)?;

        c.skip(8)?;
        let root_oh = c.u64()?;

        Ok(Self { bytes, root_oh })
    }

    pub fn group(&self, path: &str) -> Result<Group<'a>> {
        let mut cur = Group {
            bytes: self.bytes,
            oh_addr: self.root_oh,
        };
        for part in path.split('/').filter(|s| !s.is_empty()) {
            cur = cur.group(part)?;
        }
        Ok(cur)
    }
}

pub struct Group<'a> {
    bytes: &'a [u8],
    oh_addr: u64,
}

impl<'a> Group<'a> {
    pub fn group(&self, name: &str) -> Result<Group<'a>> {
        let child = self.find_child(name)?;
        Ok(Group {
            bytes: self.bytes,
            oh_addr: child,
        })
    }

    pub fn dataset(&self, name: &str) -> Result<Dataset<'a>> {
        let child = self.find_child(name)?;
        let oh = parse_object_header(self.bytes, child)?;
        let ds = oh.dataspace.ok_or(H5Error::MissingMessage("dataspace"))?;
        let dt = oh.datatype.ok_or(H5Error::MissingMessage("datatype"))?;
        if !matches!(dt, Datatype::F32Le) {
            return Err(H5Error::UnsupportedDatatype);
        }
        let Layout::Contiguous {
            addr: data_addr,
            size: data_size,
        } = oh.layout.ok_or(H5Error::MissingMessage("data layout"))?;
        let num_elements: u64 = ds.dims.iter().product();
        Ok(Dataset {
            bytes: self.bytes,
            shape: ds.dims.iter().map(|&d| d as usize).collect(),
            data_addr,
            data_size,
            num_elements: num_elements as usize,
        })
    }

    fn find_child(&self, name: &str) -> Result<u64> {
        let oh = parse_object_header(self.bytes, self.oh_addr)?;
        let btree = oh.btree_addr.ok_or(H5Error::NotFound(name.into()))?;
        let heap = oh.heap_addr.ok_or(H5Error::NotFound(name.into()))?;
        let heap_data_addr = parse_local_heap(self.bytes, heap)?;
        walk_btree(self.bytes, btree, heap_data_addr, name)
    }
}

pub struct Dataset<'a> {
    bytes: &'a [u8],
    shape: Vec<usize>,
    data_addr: u64,
    data_size: u64,
    num_elements: usize,
}

impl Dataset<'_> {
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn read_raw(&self) -> Result<Vec<f32>> {
        let byte_len = self.num_elements.saturating_mul(4);
        if (byte_len as u64) > self.data_size {
            return Err(H5Error::Malformed(format!(
                "dataset size {} < expected {} for {} f32 elements",
                self.data_size, byte_len, self.num_elements
            )));
        }
        let start = self.data_addr as usize;
        let end = start.checked_add(byte_len).ok_or(H5Error::Truncated)?;
        if end > self.bytes.len() {
            return Err(H5Error::Truncated);
        }
        let mut out = Vec::with_capacity(self.num_elements);
        for chunk in self.bytes[start..end].chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }
}

struct ObjectHeader {
    btree_addr: Option<u64>,
    heap_addr: Option<u64>,
    dataspace: Option<Dataspace>,
    datatype: Option<Datatype>,
    layout: Option<Layout>,
}

struct Dataspace {
    dims: Vec<u64>,
}

#[derive(Debug, PartialEq)]
enum Datatype {
    F32Le,
    Other,
}

enum Layout {
    Contiguous { addr: u64, size: u64 },
}

fn parse_object_header(bytes: &[u8], addr: u64) -> Result<ObjectHeader> {
    let mut c = Cursor::new(bytes, addr as usize);
    let version = c.u8()?;
    if version != 1 {
        return Err(H5Error::UnsupportedObjectHeader(version));
    }
    c.skip(1)?;
    c.skip(2)?;
    c.skip(4)?;
    let header_size = c.u32()? as u64;
    c.skip(4)?;

    let start = c.tell();
    let end = start + header_size;

    let mut oh = ObjectHeader {
        btree_addr: None,
        heap_addr: None,
        dataspace: None,
        datatype: None,
        layout: None,
    };
    parse_messages(bytes, start, end, &mut oh)?;
    Ok(oh)
}

fn parse_messages(
    bytes: &[u8],
    block_start: u64,
    block_end: u64,
    oh: &mut ObjectHeader,
) -> Result<()> {
    let mut c = Cursor::new(bytes, block_start as usize);
    while c.tell() + 8 <= block_end {
        let msg_type = c.u16()?;
        let msg_size = c.u16()? as u64;
        c.skip(4)?;
        let data_start = c.tell();
        let data_end = data_start.checked_add(msg_size).ok_or(H5Error::Truncated)?;
        if data_end > block_end {
            return Err(H5Error::Malformed(
                "object header message overflows block".into(),
            ));
        }

        match msg_type {
            0x0000 => {} // NIL
            0x0001 => oh.dataspace = Some(parse_dataspace(bytes, data_start)?),
            0x0003 => oh.datatype = Some(parse_datatype(bytes, data_start)?),
            0x0008 => oh.layout = Some(parse_layout(bytes, data_start)?),
            0x0010 => {
                let mut cc = Cursor::new(bytes, data_start as usize);
                let cont_addr = cc.u64()?;
                let cont_len = cc.u64()?;
                parse_messages(bytes, cont_addr, cont_addr + cont_len, oh)?;
            }
            0x0011 => {
                let mut cc = Cursor::new(bytes, data_start as usize);
                oh.btree_addr = Some(cc.u64()?);
                oh.heap_addr = Some(cc.u64()?);
            }
            _ => {}
        }

        c.seek(data_end)?;
    }
    Ok(())
}

fn parse_dataspace(bytes: &[u8], addr: u64) -> Result<Dataspace> {
    let mut c = Cursor::new(bytes, addr as usize);
    let version = c.u8()?;
    if version != 1 {
        return Err(H5Error::UnsupportedDataspace(version));
    }
    let rank = c.u8()? as usize;
    c.skip(1)?;
    c.skip(5)?;
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(c.u64()?);
    }
    Ok(Dataspace { dims })
}

fn parse_datatype(bytes: &[u8], addr: u64) -> Result<Datatype> {
    let mut c = Cursor::new(bytes, addr as usize);
    let class_and_ver = c.u8()?;
    let class = class_and_ver & 0x0F;
    let bf0 = c.u8()?;
    c.skip(2)?;
    let size = c.u32()?;

    if class == 1 && size == 4 && (bf0 & 0x01) == 0 {
        Ok(Datatype::F32Le)
    } else {
        Ok(Datatype::Other)
    }
}

fn parse_layout(bytes: &[u8], addr: u64) -> Result<Layout> {
    let mut c = Cursor::new(bytes, addr as usize);
    let version = c.u8()?;
    if version != 3 {
        return Err(H5Error::UnsupportedLayoutVersion(version));
    }
    let class = c.u8()?;
    if class != 1 {
        return Err(H5Error::UnsupportedLayoutClass(class));
    }
    let addr = c.u64()?;
    let size = c.u64()?;
    Ok(Layout::Contiguous { addr, size })
}

fn walk_btree(bytes: &[u8], node_addr: u64, heap_data_addr: u64, target: &str) -> Result<u64> {
    let mut c = Cursor::new(bytes, node_addr as usize);
    let sig = c.take(4)?;
    if sig != b"TREE" {
        return Err(H5Error::Malformed("expected TREE signature".into()));
    }
    let node_type = c.u8()?;
    let node_level = c.u8()?;
    let entries_used = c.u16()? as usize;
    c.skip(8)?;
    c.skip(8)?;
    if node_type != 0 {
        return Err(H5Error::Malformed(format!(
            "expected B-tree group-node type 0, got {}",
            node_type
        )));
    }

    let mut children = Vec::with_capacity(entries_used);
    for _ in 0..entries_used {
        c.skip(8)?;
        children.push(c.u64()?);
    }

    if node_level == 0 {
        for &snod_addr in &children {
            if let Some(oh) = scan_snod(bytes, snod_addr, heap_data_addr, target)? {
                return Ok(oh);
            }
        }
    } else {
        for &child in &children {
            match walk_btree(bytes, child, heap_data_addr, target) {
                Ok(oh) => return Ok(oh),
                Err(H5Error::NotFound(_)) => continue,
                Err(e) => return Err(e),
            }
        }
    }
    Err(H5Error::NotFound(target.into()))
}

fn scan_snod(bytes: &[u8], addr: u64, heap_data_addr: u64, target: &str) -> Result<Option<u64>> {
    let mut c = Cursor::new(bytes, addr as usize);
    let sig = c.take(4)?;
    if sig != b"SNOD" {
        return Err(H5Error::Malformed("expected SNOD signature".into()));
    }
    c.skip(1)?;
    c.skip(1)?;
    let num_symbols = c.u16()? as usize;
    for _ in 0..num_symbols {
        let link_name_off = c.u64()?;
        let oh_addr = c.u64()?;
        c.skip(4)?;
        c.skip(4)?;
        c.skip(16)?;
        let name = read_heap_string(bytes, heap_data_addr + link_name_off)?;
        if name == target {
            return Ok(Some(oh_addr));
        }
    }
    Ok(None)
}

fn parse_local_heap(bytes: &[u8], addr: u64) -> Result<u64> {
    let mut c = Cursor::new(bytes, addr as usize);
    let sig = c.take(4)?;
    if sig != b"HEAP" {
        return Err(H5Error::Malformed("expected HEAP signature".into()));
    }
    c.skip(1)?;
    c.skip(3)?;
    c.skip(8)?;
    c.skip(8)?;
    c.u64()
}

fn read_heap_string(bytes: &[u8], addr: u64) -> Result<String> {
    let start = addr as usize;
    if start > bytes.len() {
        return Err(H5Error::Truncated);
    }
    let end = bytes[start..]
        .iter()
        .position(|&b| b == 0)
        .ok_or(H5Error::Truncated)?;
    std::str::from_utf8(&bytes[start..start + end])
        .map(String::from)
        .map_err(|_| H5Error::Malformed("non-utf8 link name".into()))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }
    fn tell(&self) -> u64 {
        self.pos as u64
    }
    fn seek(&mut self, pos: u64) -> Result<()> {
        let p = pos as usize;
        if p > self.bytes.len() {
            return Err(H5Error::Truncated);
        }
        self.pos = p;
        Ok(())
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(H5Error::Truncated)?;
        if end > self.bytes.len() {
            return Err(H5Error::Truncated);
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn load_weights_h5(keras_path: &str) -> Vec<u8> {
        let file = std::fs::File::open(keras_path).unwrap();
        let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file)).unwrap();
        let mut entry = archive.by_name("model.weights.h5").unwrap();
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).unwrap();
        buf
    }

    #[test]
    #[ignore = "requires local Keras fixtures under examples/data/"]
    fn reads_simple_model_weights() {
        let bytes = load_weights_h5("examples/data/simple_model.keras");
        let r = H5Reader::open(&bytes).unwrap();
        let ds = r.group("layers/dense/vars").unwrap().dataset("0").unwrap();
        assert_eq!(ds.shape(), vec![4, 8]);
        let data = ds.read_raw().unwrap();
        assert_eq!(data.len(), 32);
        let bias = r.group("layers/dense/vars").unwrap().dataset("1").unwrap();
        assert_eq!(bias.shape(), vec![8]);
        assert_eq!(bias.read_raw().unwrap().len(), 8);
    }

    #[test]
    #[ignore = "requires local Keras fixtures under examples/data/"]
    fn reads_conv_model_weights() {
        let bytes = load_weights_h5("examples/data/conv_model.keras");
        let r = H5Reader::open(&bytes).unwrap();
        let kernel = r.group("layers/conv2d/vars").unwrap().dataset("0").unwrap();
        assert_eq!(kernel.shape(), vec![3, 3, 1, 32]);
        assert_eq!(kernel.read_raw().unwrap().len(), 3 * 3 * 1 * 32);
    }

    #[test]
    #[ignore = "requires local Keras fixtures under examples/data/"]
    fn reads_batch_norm_model_weights() {
        let bytes = load_weights_h5("examples/data/batch_norm_model.keras");
        let r = H5Reader::open(&bytes).unwrap();
        let gamma = r
            .group("layers/batch_normalization/vars")
            .unwrap()
            .dataset("0")
            .unwrap();
        assert_eq!(gamma.shape(), vec![8]);
        assert_eq!(gamma.read_raw().unwrap().len(), 8);
    }

    #[test]
    #[ignore = "requires local Keras fixtures under examples/data/"]
    fn missing_path_errors_cleanly() {
        let bytes = load_weights_h5("examples/data/simple_model.keras");
        let r = H5Reader::open(&bytes).unwrap();
        match r.group("layers/not_a_layer") {
            Err(H5Error::NotFound(_)) => {}
            other => panic!("expected NotFound, got {:?}", other.err()),
        }
    }

    #[test]
    fn rejects_non_hdf5_input() {
        match H5Reader::open(b"not an hdf5 file") {
            Err(H5Error::InvalidSignature) => {}
            other => panic!("expected InvalidSignature, got {:?}", other.err()),
        }
    }
}
