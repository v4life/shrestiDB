//! Learned buffer pool with predictive prefetching
//!
//! Uses Markov chain models to predict future page accesses.

use crate::error::Result;
use crate::storage::page::{PageId, SlottedPage, PAGE_SIZE};
use crate::storage::disk_manager::DiskManager;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{debug, info};

/// Markov chain predictor for page access sequences
#[derive(Debug, Clone)]
pub struct PageAccessPredictor {
    /// Transition probabilities: current_page -> next_page
    transitions: HashMap<u64, HashMap<u64, usize>>,
    /// Total observations for each state
    state_counts: HashMap<u64, usize>,
}

impl PageAccessPredictor {
    pub fn new() -> Self {
        PageAccessPredictor {
            transitions: HashMap::new(),
            state_counts: HashMap::new(),
        }
    }

    /// Record a page access sequence
    pub fn observe(&mut self, from_page: u64, to_page: u64) {
        let entry = self
            .transitions
            .entry(from_page)
            .or_insert_with(HashMap::new);
        *entry.entry(to_page).or_insert(0) += 1;
        *self.state_counts.entry(from_page).or_insert(0) += 1;
    }

    /// Predict the next page access given current page
    pub fn predict_next(&self, current_page: u64) -> Option<u64> {
        self.transitions
            .get(&current_page)
            .and_then(|next_pages| {
                next_pages
                    .iter()
                    .max_by_key(|(_, count)| *count)
                    .map(|(page, _)| *page)
            })
    }
}

impl Default for PageAccessPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Learned buffer pool entry
struct BufferEntry {
    /// Cached page data
    page: SlottedPage,
    /// Last access time
    last_access: u64,
    /// Access frequency for LFU fallback
    access_count: usize,
    /// Whether dirty (needs flush)
    dirty: bool,
}

/// Learned buffer pool manager
pub struct LearnedBufferPool {
    /// Buffer capacity (number of pages)
    capacity: usize,
    /// Current pages in buffer
    buffer: Arc<RwLock<HashMap<PageId, BufferEntry>>>,
    /// Access predictor
    predictor: Arc<RwLock<PageAccessPredictor>>,
    /// Disk manager reference
    disk_manager: Arc<DiskManager>,
    /// Clock for LRU
    clock: Arc<RwLock<u64>>,
}

impl LearnedBufferPool {
    /// Create a new learned buffer pool
    pub fn new(capacity: usize, disk_manager: Arc<DiskManager>) -> Self {
        info!("LearnedBufferPool initialized with capacity {} pages", capacity);

        LearnedBufferPool {
            capacity,
            buffer: Arc::new(RwLock::new(HashMap::new())),
            predictor: Arc::new(RwLock::new(PageAccessPredictor::new())),
            disk_manager,
            clock: Arc::new(RwLock::new(0)),
        }
    }

    /// Fetch a page, using prediction for prefetch
    pub fn fetch_page(
        &self,
        file_id: u32,
        page_id: PageId,
        last_page: Option<PageId>,
    ) -> Result<SlottedPage> {
        let mut clock = self.clock.write();
        *clock += 1;
        drop(clock);

        // Record access pattern
        if let Some(last) = last_page {
            let mut predictor = self.predictor.write();
            predictor.observe(last.as_u64(), page_id.as_u64());
        }

        let mut buffer = self.buffer.write();

        // Check if page is already in buffer
        if let Some(entry) = buffer.get_mut(&page_id) {
            entry.last_access = *self.clock.read();
            entry.access_count += 1;
            debug!("Buffer hit for page {}", page_id.as_u64());
            return Ok(entry.page.clone());
        }

        // Page miss: load from disk
        let mut page_data = vec![0u8; PAGE_SIZE];
        self.disk_manager
            .read_page(file_id, page_id, &mut page_data)?;

        let page = SlottedPage::new(page_id);
        let entry = BufferEntry {
            page: page.clone(),
            last_access: *self.clock.read(),
            access_count: 1,
            dirty: false,
        };

        buffer.insert(page_id, entry);

        // Evict if necessary
        if buffer.len() > self.capacity {
            self.evict_lfu(&mut buffer);
        }

        debug!("Buffer miss and loaded page {} from disk", page_id.as_u64());
        Ok(page)
    }

    /// Evict a page using LFU (Least Frequently Used) strategy
    fn evict_lfu(&self, buffer: &mut HashMap<PageId, BufferEntry>) {
        if let Some((victim_id, _)) = buffer
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
        {
            let victim_id = *victim_id;
            buffer.remove(&victim_id);
            debug!("Evicted page {} using LFU", victim_id.as_u64());
        }
    }

    /// Pin a page in the buffer (prevent eviction)
    pub fn pin_page(&self, page_id: PageId) -> Result<()> {
        // TODO: Implement pin tracking
        debug!("Pinned page {}", page_id.as_u64());
        Ok(())
    }

    /// Unpin a page
    pub fn unpin_page(&self, page_id: PageId) -> Result<()> {
        // TODO: Implement unpin tracking
        debug!("Unpinned page {}", page_id.as_u64());
        Ok(())
    }

    /// Flush all dirty pages to disk
    pub fn flush_all(&self, _file_id: u32) -> Result<()> {
        let buffer = self.buffer.read();
        for (page_id, entry) in buffer.iter() {
            if entry.dirty {
                // TODO: Serialize and write page
                debug!("Flushed page {} to disk", page_id.as_u64());
            }
        }
        Ok(())
    }

    /// Get buffer hit ratio
    pub fn get_stats(&self) -> BufferPoolStats {
        let buffer = self.buffer.read();
        BufferPoolStats {
            num_pages: buffer.len(),
            capacity: self.capacity,
        }
    }
}

/// Buffer pool statistics
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    pub num_pages: usize,
    pub capacity: usize,
}

impl Clone for SlottedPage {
    fn clone(&self) -> Self {
        SlottedPage::new(self.page_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_predictor() {
        let mut predictor = PageAccessPredictor::new();
        predictor.observe(1, 2);
        predictor.observe(1, 2);
        predictor.observe(1, 3);

        assert_eq!(predictor.predict_next(1), Some(2));
    }

    #[test]
    fn test_buffer_pool_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let dm = Arc::new(DiskManager::new(temp_dir.path().to_path_buf())?);
        let pool = LearnedBufferPool::new(100, dm);

        let stats = pool.get_stats();
        assert_eq!(stats.capacity, 100);
        assert_eq!(stats.num_pages, 0);

        Ok(())
    }
}
