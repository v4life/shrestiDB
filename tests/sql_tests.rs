//! SQL parsing and query handling tests

#[cfg(test)]
mod sql_tests {
    use shrestidb::execution::catalog::{Catalog, TableSchema, Column, DataType};
    use shrestidb::optimizer::planner::QueryPlanner;

    #[test]
    fn test_simple_select_parsing() {
        let planner = QueryPlanner::new();
        let query = "SELECT * FROM users WHERE age > 18";
        let plan = planner.plan(query).unwrap();
        
        assert!(!plan.nodes.is_empty());
    }

    #[test]
    fn test_schema_with_multiple_columns() {
        let mut schema = TableSchema::new(1, "employees".to_string());
        
        schema.add_column(Column {
            id: 1,
            name: "id".to_string(),
            data_type: DataType::Integer,
            nullable: false,
            primary_key: true,
        });
        
        schema.add_column(Column {
            id: 2,
            name: "name".to_string(),
            data_type: DataType::String,
            nullable: false,
            primary_key: false,
        });
        
        schema.add_column(Column {
            id: 3,
            name: "salary".to_string(),
            data_type: DataType::Float,
            nullable: true,
            primary_key: false,
        });
        
        assert_eq!(schema.columns.len(), 3);
        
        let name_col = schema.get_column("name");
        assert!(name_col.is_some());
        assert_eq!(name_col.unwrap().data_type, DataType::String);
    }

    #[test]
    fn test_catalog_with_multiple_tables() {
        let mut catalog = Catalog::new();
        
        for table_id in 1..=5 {
            let schema = TableSchema::new(table_id, format!("table_{}", table_id));
            catalog.register_table(schema);
        }
        
        assert_eq!(catalog.tables.len(), 5);
        assert!(catalog.get_table("table_1").is_some());
        assert!(catalog.get_table("table_5").is_some());
        assert!(catalog.get_table("table_999").is_none());
    }

    #[test]
    fn test_query_plan_generation() {
        let planner = QueryPlanner::new();
        
        let plan1 = planner.plan("SELECT * FROM users").unwrap();
        let plan2 = planner.plan("SELECT id, name FROM users WHERE age > 18").unwrap();
        let plan3 = planner.plan("SELECT * FROM users JOIN orders ON users.id = orders.user_id").unwrap();
        
        assert!(!plan1.nodes.is_empty());
        assert!(!plan2.nodes.is_empty());
        assert!(!plan3.nodes.is_empty());
        
        println!("Plan 1 estimated cost: {}", plan1.estimated_cost);
        println!("Plan 2 estimated cost: {}", plan2.estimated_cost);
        println!("Plan 3 estimated cost: {}", plan3.estimated_cost);
    }

    #[test]
    fn test_vector_ddl_insert_and_similarity_search() {
        use shrestidb::execution::QueryExecutor;

        let executor = QueryExecutor::new(Catalog::new());

        // 1. CREATE TABLE with VECTOR(3)
        executor.execute_sql("CREATE TABLE items (id INT PRIMARY KEY, name TEXT, embedding VECTOR(3))").unwrap();

        // 2. INSERT items with 3D vector embeddings
        executor.execute_sql("INSERT INTO items VALUES (1, 'item_x', '[1.0, 0.0, 0.0]')").unwrap();
        executor.execute_sql("INSERT INTO items VALUES (2, 'item_y', '[0.0, 1.0, 0.0]')").unwrap();
        executor.execute_sql("INSERT INTO items VALUES (3, 'item_xy', '[0.707, 0.707, 0.0]')").unwrap();

        // 3. Simple SELECT to verify storage & formatting
        let rows = executor.execute_sql("SELECT id, name, embedding FROM items").unwrap();
        assert_eq!(rows.len(), 3);

        // 4. pgvector-style Top-K nearest neighbor search by Cosine Distance (<=>)
        // Query near X-axis: [0.9, 0.1, 0.0] -> item_x (1) must be first, then item_xy (3), then item_y (2)
        let knn_cosine = executor.execute_sql("SELECT id, name FROM items ORDER BY embedding <=> '[0.9, 0.1, 0.0]' LIMIT 2").unwrap();
        assert_eq!(knn_cosine.len(), 2);
        assert_eq!(knn_cosine[0][0], "1"); // item_x closest
        assert_eq!(knn_cosine[1][0], "3"); // item_xy second closest

        // 5. Nearest neighbor search by L2 Distance (<->)
        // Query near origin: [0.05, 0.05, 0.0] -> item_xy (norm 1.0) is further than nothing, but let's test specific origin point
        let knn_l2 = executor.execute_sql("SELECT id, name FROM items ORDER BY embedding <-> '[0.0, 0.9, 0.0]' LIMIT 1").unwrap();
        assert_eq!(knn_l2.len(), 1);
        assert_eq!(knn_l2[0][0], "2"); // item_y is at [0.0, 1.0, 0.0], distance is 0.1

        // 6. Nearest neighbor search by Inner Product (<#>)
        let knn_ip = executor.execute_sql("SELECT id, name FROM items ORDER BY embedding <#> '[1.0, 0.0, 0.0]' LIMIT 1").unwrap();
        assert_eq!(knn_ip.len(), 1);
        assert_eq!(knn_ip[0][0], "1"); // item_x has highest dot product (1.0), lowest negated IP (-1.0)
    }
}
