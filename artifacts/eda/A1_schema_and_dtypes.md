# A1: Dataset Schema and Data Types

| column | dtype |
| --- | --- |
| id | Int64 |
| title | String |
| body | String |
| created_at | String |
| rate | Float64 |
| recommendation_status | String |
| is_buyer | Boolean |
| product_id | Int64 |
| advantages | String |
| disadvantages | String |
| likes | Int64 |
| dislikes | Int64 |
| seller_title | String |
| seller_code | Int64 |
| true_to_size_rate | Float64 |

## Metadata
- **total_records**: `6156289`
- **unique_products**: `331599`
- **unique_seller_codes**: `0`
- **unique_seller_titles**: `29228`
- **num_columns**: `15`
- **csv_file_size_mb**: `1219.3`
- **parquet_file_size_mb**: `234.79`
- **product_review_distribution**: `{'min_reviews_per_product': 1, 'p25_reviews': 1.0, 'median_reviews': 4.0, 'mean_reviews': 18.565463104532885, 'std_reviews': 40.9340966816239, 'p75_reviews': 13.0, 'p90_reviews': 46.0, 'p99_reviews': 200.0, 'max_reviews_per_product': 400, 'products_with_only_1_review': 91944, 'products_with_le_5_reviews': 198458, 'products_with_100plus_reviews': 18047, 'products_with_1000plus_reviews': 0}`
