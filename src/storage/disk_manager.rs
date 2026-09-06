//! Direct I/O disk manager with async support
//!
//! Handles low-level file operations with optional io_uring backend.

use crate::error::{DatabaseError, Result};
use crate::storage::page::{PageId, PAGE_SIZE};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info};

/// Disk manager for page-based storage
pub struct DiskManager {
    /// Data file path
    data_dir: PathBuf,
    /// Open file handles (one per table/file)
    files: Arc<RwLock<std::collections::HashMap<u32, File>>>,
}

impl DiskManager {
    /// Create a new disk manager
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;
        info!("DiskManager initialized at {:?}", data_dir);

        Ok(DiskManager {
            data_dir,
            files: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Get or open a file for a given table
    fn get_or_open_file(&self, file_id: u32) -> Result<File> {
        let mut files = self.files.write();

        if let Some(file) = files.get(&file_id) {
            return Ok(file.try_clone()?);
        }

        let path = self.data_dir.join(format!("table_{}.db", file_id));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;

        files.insert(file_id, file.try_clone()?);
        Ok(file)
    }

    /// Read a page from disk
    ///
    /// # Arguments
    /// * `file_id` - Table/file identifier
    /// * `page_id` - Page identifier to read
    /// * `buffer` - Buffer to read into (must be PAGE_SIZE bytes)
    pub fn read_page(&self, file_id: u32, page_id: PageId, buffer: &mut [u8]) -> Result<()> {
        if buffer.len() < PAGE_SIZE {
            return Err(DatabaseError::Unknown(
                "Buffer too small for page read".to_string(),
            ));
        }

        let mut file = self.get_or_open_file(file_id)?;
        let offset = (page_id.as_u64() as u64) * (PAGE_SIZE as u64);

        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut buffer[..PAGE_SIZE])?;

        debug!("Read page {} from file {}", page_id.as_u64(), file_id);
        Ok(())
    }

    /// Write a page to disk
    ///
    /// # Arguments
    /// * `file_id` - Table/file identifier
    /// * `page_id` - Page identifier to write
    /// * `data` - Page data to write (must be PAGE_SIZE bytes)
    pub fn write_page(&self, file_id: u32, page_id: PageId, data: &[u8]) -> Result<()> {
        if data.len() < PAGE_SIZE {
            return Err(DatabaseError::Unknown(
                "Data too small for page write".to_string(),
            ));
        }

        let mut file = self.get_or_open_file(file_id)?;
        let offset = (page_id.as_u64() as u64) * (PAGE_SIZE as u64);

        file.seek(SeekFrom::Start(offset))?;
        file.write_all(&data[..PAGE_SIZE])?;
        file.sync_all()?;

        debug!("Wrote page {} to file {}", page_id.as_u64(), file_id);
        Ok(())
    }

    /// Flush all writes to disk
    pub fn flush(&self) -> Result<()> {
        let files = self.files.read();
        for (_, file) in files.iter() {
            file.sync_all()?;
        }
        Ok(())
    }

    /// Get file size in pages
    pub fn file_size_pages(&self, file_id: u32) -> Result<u64> {
        let file = self.get_or_open_file(file_id)?;
        let metadata = file.metadata()?;
        Ok(metadata.len() / (PAGE_SIZE as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_disk_manager_read_write() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let dm = DiskManager::new(temp_dir.path().to_path_buf())?;

        let mut write_buffer = vec![0u8; PAGE_SIZE];
        write_buffer[0..5].copy_from_slice(b"hello");

        dm.write_page(1, PageId::new(0), &write_buffer)?;

        let mut read_buffer = vec![0u8; PAGE_SIZE];
        dm.read_page(1, PageId::new(0), &mut read_buffer)?;

        assert_eq!(&read_buffer[0..5], b"hello");
        Ok(())
    }
}
