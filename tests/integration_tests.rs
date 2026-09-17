//! Integration tests for the learned database kernel

#[cfg(test)]
mod tests {
    use shrestidb::index::pgm::PGMIndex;
    use shrestidb::index::btree::BTree;
    use shrestidb::index::rmi::{RMIIndex, RMIStage};
    use shrestidb::index::models::LinearModel;
    use shrestidb::optimizer::cardinality::ColumnDistribution;
    use shrestidb::optimizer::cost_model::CostModel;
    use shrestidb::execution::catalog::Catalog;
    use shrestidb::execution::operators::Value;
    use shrestidb::execution::transaction::TransactionManager;

    #[test]
    fn test_pgm_index_correctness() {
        let keys = vec![1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
        let pgm = PGMIndex::build(keys, 1);

        // Test searching for existing keys
        assert!(pgm.search(10.0).is_some());
        assert!(pgm.search(25.0).is_some());
        
        // Non-existent key should return None
        assert!(pgm.search(7.0).is_none());
    }

    #[test]
    fn test_btree_basic_operations() {
        let mut btree = BTree::new(4);
        btree.insert(10.0, 1).unwrap();
        btree.insert(20.0, 2).unwrap();
        btree.insert(15.0, 3).unwrap();

        assert_eq!(btree.search(10.0), Some(1));
        assert_eq!(btree.search(20.0), Some(2));
        assert_eq!(btree.search(15.0), Some(3));
        assert_eq!(btree.search(25.0), None);
    }

    #[test]
    fn test_rmi_index_creation() {
        let keys = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let positions = vec![0, 1, 2, 3, 4];
        let model = LinearModel::new(1.0, 0.0);
        let stage = RMIStage::new(vec![model]);
        let rmi = RMIIndex::new(vec![stage], keys, positions);

        let pred = rmi.search(3.0);
        assert!(pred < 5);
    }

    #[test]
    fn test_cardinality_estimator_accuracy() {
        // A real accuracy check against a known distribution -- the old
        // version of this test only checked a trivial identity (empty
        // predicates return the full count) and would have passed
        // against the previous, entirely fake estimator too. This one
        // actually exercises the real PGM-backed CDF: 9,000 rows at or
        // below 900 and 1,000 rows above it, so ">= 900" should estimate
        // close to the true 1,000, not a generic guess.
        let mut values: Vec<f64> = (0..9000).map(|_| 500.0).collect();
        values.extend((0..1000).map(|i| 901.0 + i as f64));
        let dist = ColumnDistribution::build_numeric(values, 32);

        let count = dist.estimate_row_count(">=", &Value::Float(900.0), 10000).unwrap();
        assert!((count as i64 - 1000).abs() < 100, "estimated {count} rows, expected close to 1000");
    }

    #[test]
    fn test_cost_model_comparison() {
        use shrestidb::optimizer::cost_model::{OperatorCost, OperatorType};
        
        let cost_model = CostModel::new();
        
        let index_scan = OperatorCost::new(OperatorType::IndexScan, 10000, 100, 0.01);
        let table_scan = OperatorCost::new(OperatorType::TableScan, 10000, 100, 0.01);
        
        let index_cost = cost_model.estimate_cost(&index_scan);
        let table_cost = cost_model.estimate_cost(&table_scan);
        
        // Index scan should be cheaper
        assert!(index_cost < table_cost);
    }

    #[test]
    fn test_catalog_schema_management() {
        use shrestidb::execution::catalog::{TableSchema, Column, DataType};
        
        let mut catalog = Catalog::new();
        let mut schema = TableSchema::new(1, "test_table".to_string());
        
        schema.add_column(Column {
            id: 1,
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: true,
        });

        catalog.register_table(schema);
        
        let retrieved = catalog.get_table("test_table");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().columns.len(), 1);
    }

    #[test]
    fn test_transaction_lifecycle() {
        let tm = TransactionManager::new();
        let tx = tm.begin();
        tm.commit(tx);
        
        let txs = tm.transactions.read().unwrap();
        assert!(txs.contains_key(&tx.0));
    }

    #[test]
    fn test_multiple_transactions() {
        let tm = TransactionManager::new();
        let tx1 = tm.begin();
        let tx2 = tm.begin();
        let tx3 = tm.begin();
        
        tm.commit(tx1);
        tm.abort(tx2);
        tm.commit(tx3);
        
        let txs = tm.transactions.read().unwrap();
        assert_eq!(txs.len(), 3);
    }

    #[test]
    fn test_index_build_performance_comparison() {
        use std::time::Instant;
        
        let keys: Vec<f64> = (0..10000).map(|i| i as f64).collect();
        
        let start = Instant::now();
        let _pgm = PGMIndex::build(keys.clone(), 32);
        let pgm_time = start.elapsed();
        
        let start = Instant::now();
        let mut _btree = BTree::new(4);
        for (i, key) in keys.iter().enumerate() {
            let _ = _btree.insert(*key, i);
        }
        let btree_time = start.elapsed();
        
        println!("PGM build time: {:?}", pgm_time);
        println!("B-Tree build time: {:?}", btree_time);
    }

    #[test]
    fn test_learned_vs_traditional_lookup() {
        use std::time::Instant;
        
        let keys: Vec<f64> = (0..10000).map(|i| (i as f64 * 1.5) % 100000.0).collect();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let pgm = PGMIndex::build(sorted_keys.clone(), 32);
        let mut btree = BTree::new(4);
        for (i, key) in sorted_keys.iter().enumerate() {
            let _ = btree.insert(*key, i);
        }
        
        let search_keys: Vec<f64> = sorted_keys.iter().step_by(100).copied().collect();
        
        let start = Instant::now();
        for key in &search_keys {
            let _ = pgm.search(*key);
        }
        let pgm_time = start.elapsed();
        
        let start = Instant::now();
        for key in &search_keys {
            let _ = btree.search(*key);
        }
        let btree_time = start.elapsed();
        
        println!("PGM lookup time: {:?}", pgm_time);
        println!("B-Tree lookup time: {:?}", btree_time);
        println!("Speedup: {:.2}x", btree_time.as_secs_f64() / pgm_time.as_secs_f64());
    }
}
