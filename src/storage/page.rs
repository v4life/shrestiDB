//! Slotted page memory layout and management

use serde::{Deserialize, Serialize};
use std::mem;

/// Unique identifier for a page
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageId(pub u64);

impl PageId {
    pub fn new(id: u64) -> Self {
        PageId(id)
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Standard page size: 4KB
pub const PAGE_SIZE: usize = 4096;

/// Header size for page metadata
pub const PAGE_HEADER_SIZE: usize = 128;

/// Maximum data capacity per page
pub const PAGE_DATA_SIZE: usize = PAGE_SIZE - PAGE_HEADER_SIZE;

/// Page header metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHeader {
    /// Page identifier
    pub page_id: PageId,

    /// Number of slots used
    pub slot_count: u16,

    /// Free space offset
    pub free_offset: u16,

    /// LSN (Log Sequence Number) for recovery
    pub lsn: u64,

    /// Page version for MVCC
    pub version: u32,

    /// Dirty flag
    pub is_dirty: bool,
}

impl PageHeader {
    pub fn new(page_id: PageId) -> Self {
        PageHeader {
            page_id,
            slot_count: 0,
            free_offset: 0,
            lsn: 0,
            version: 0,
            is_dirty: false,
        }
    }

    pub fn available_space(&self) -> usize {
        (PAGE_DATA_SIZE as u16).saturating_sub(self.free_offset) as usize
    }
}

/// Slot directory entry (16 bytes)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SlotEntry {
    /// Offset of data within page
    pub offset: u16,

    /// Length of data
    pub length: u16,

    /// Flags (valid, deleted, etc.)
    pub flags: u8,
}

impl SlotEntry {
    pub fn new(offset: u16, length: u16) -> Self {
        SlotEntry {
            offset,
            length,
            flags: 0x01, // valid flag
        }
    }

    pub fn is_valid(&self) -> bool {
        (self.flags & 0x01) != 0
    }

    pub fn mark_deleted(&mut self) {
        self.flags &= !0x01;
    }
}

/// Slotted page with variable-length records
#[derive(Debug)]
pub struct SlottedPage {
    /// Page header
    header: PageHeader,

    /// Raw page data
    data: [u8; PAGE_DATA_SIZE],

    /// Slot directory (grows from the end backwards)
    slots: Vec<SlotEntry>,
}

impl SlottedPage {
    /// Create a new empty slotted page
    pub fn new(page_id: PageId) -> Self {
        SlottedPage {
            header: PageHeader::new(page_id),
            data: [0u8; PAGE_DATA_SIZE],
            slots: Vec::new(),
        }
    }

    /// Get the page ID
    pub fn page_id(&self) -> PageId {
        self.header.page_id
    }

    /// Get the header
    pub fn header(&self) -> &PageHeader {
        &self.header
    }

    /// Get mutable header reference
    pub fn header_mut(&mut self) -> &mut PageHeader {
        &mut self.header
    }

    /// Insert a variable-length record into the page
    ///
    /// Returns the slot index if successful, None if page is full
    pub fn insert_record(&mut self, record: &[u8]) -> Option<usize> {
        let record_len = record.len();

        // Check if there's enough space
        let required_space = record_len + mem::size_of::<SlotEntry>();
        if self.header.available_space() < required_space {
            return None;
        }

        // Place data at current free offset
        let slot_offset = self.header.free_offset;
        let slot_index = self.slots.len();

        self.data[slot_offset as usize..(slot_offset as usize + record_len)]
            .copy_from_slice(record);

        // Add slot entry
        self.slots.push(SlotEntry::new(slot_offset, record_len as u16));

        // Update header
        self.header.free_offset += record_len as u16;
        self.header.slot_count += 1;
        self.header.is_dirty = true;

        Some(slot_index)
    }

    /// Retrieve a record by slot index
    pub fn get_record(&self, slot_index: usize) -> Option<&[u8]> {
        if slot_index >= self.slots.len() {
            return None;
        }

        let slot = &self.slots[slot_index];
        if !slot.is_valid() {
            return None;
        }

        let offset = slot.offset as usize;
        let length = slot.length as usize;

        Some(&self.data[offset..offset + length])
    }

    /// Delete a record (mark as invalid)
    pub fn delete_record(&mut self, slot_index: usize) -> bool {
        if slot_index >= self.slots.len() {
            return false;
        }

        self.slots[slot_index].mark_deleted();
        self.header.is_dirty = true;
        true
    }

    /// Get number of records in page
    pub fn record_count(&self) -> usize {
        self.slots.len()
    }

    /// Check if page is full
    pub fn is_full(&self) -> bool {
        self.header.available_space() < PAGE_HEADER_SIZE
    }

    /// Serialize page to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut buffer = Vec::with_capacity(PAGE_SIZE);

        // Serialize header
        let header_bytes = bincode::serialize(&self.header).unwrap();
        buffer.extend_from_slice(&header_bytes);

        // Pad to PAGE_HEADER_SIZE
        while buffer.len() < PAGE_HEADER_SIZE {
            buffer.push(0);
        }

        // Serialize slot directory (reverse order for safety)
        for slot in self.slots.iter().rev() {
            let slot_bytes = bincode::serialize(slot).unwrap();
            buffer.extend_from_slice(&slot_bytes);
        }

        // Serialize data
        buffer.extend_from_slice(&self.data);

        buffer.truncate(PAGE_SIZE);
        buffer
    }
}

/// Trait for page-like objects
pub trait Page {
    fn page_id(&self) -> PageId;
    fn is_dirty(&self) -> bool;
    fn mark_clean(&mut self);
    fn mark_dirty(&mut self);
}

impl Page for SlottedPage {
    fn page_id(&self) -> PageId {
        self.header.page_id
    }

    fn is_dirty(&self) -> bool {
        self.header.is_dirty
    }

    fn mark_clean(&mut self) {
        self.header.is_dirty = false;
    }

    fn mark_dirty(&mut self) {
        self.header.is_dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slotted_page_creation() {
        let page = SlottedPage::new(PageId::new(1));
        assert_eq!(page.page_id(), PageId::new(1));
        assert_eq!(page.record_count(), 0);
        assert!(!page.is_full());
    }

    #[test]
    fn test_insert_and_retrieve() {
        let mut page = SlottedPage::new(PageId::new(1));
        let data = b"hello world";

        let slot_idx = page.insert_record(data).expect("Insert failed");
        assert_eq!(slot_idx, 0);

        let retrieved = page.get_record(0).expect("Retrieve failed");
        assert_eq!(retrieved, data);
    }

    #[test]
    fn test_multiple_records() {
        let mut page = SlottedPage::new(PageId::new(1));

        let records = vec![b"record1", b"record2", b"record3"];
        for record in &records {
            page.insert_record(*record).expect("Insert failed");
        }

        assert_eq!(page.record_count(), 3);

        for (i, expected) in records.iter().enumerate() {
            let retrieved = page.get_record(i).expect("Retrieve failed");
            assert_eq!(retrieved, *expected);
        }
    }

    #[test]
    fn test_delete_record() {
        let mut page = SlottedPage::new(PageId::new(1));
        page.insert_record(b"data").expect("Insert failed");

        page.delete_record(0);
        assert!(page.get_record(0).is_none());
    }
}
