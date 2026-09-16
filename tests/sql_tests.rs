//! SQL parsing and query handling tests

#[cfg(test)]
mod sql_tests {
    use shrestidb::execution::catalog::{Catalog, TableSchema, Column, DataType};
    use shrestidb::optimizer::planner::QueryPlanner;

    #[test]
    fn test_simple_select_parsing() {
        let planner = QueryPlanner::new();
        let query = "SELECT * FROM users WHERE age > 18";
        let plan = planner.plan(query);
        
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
        
        let plan1 = planner.plan("SELECT * FROM users");
        let plan2 = planner.plan("SELECT id, name FROM users WHERE age > 18");
        let plan3 = planner.plan("SELECT * FROM users JOIN orders ON users.id = orders.user_id");
        
        assert!(!plan1.nodes.is_empty());
        assert!(!plan2.nodes.is_empty());
        assert!(!plan3.nodes.is_empty());
        
        println!("Plan 1 estimated cost: {}", plan1.estimated_cost);
        println!("Plan 2 estimated cost: {}", plan2.estimated_cost);
        println!("Plan 3 estimated cost: {}", plan3.estimated_cost);
    }
}
