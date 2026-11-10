# A1: Dataset Size, Schema and Concentration (raw vs clean)

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
| seller_code | String |
| true_to_size_rate | String |

- CSV size: 1219.3 MB; raw Parquet: 260.01 MB (78.7% smaller, 4.69x)
- Extra pipeline columns in Parquet: ['row_idx', 'rate_out_of_range']

## RAW
- **total_records**: `6156289`
- **unique_products**: `331599`
- **unique_seller_codes_excl_nan**: `29213`
- **unique_seller_titles_excl_nan**: `29228`
- **rows_without_seller**: `302181`
- **unique_ids**: `6153060`
- **product_review_distribution**: `{'min': 1, 'p25': 1.0, 'median': 4.0, 'mean': 18.57, 'std': 40.93, 'p75': 13.0, 'p90': 46.0, 'p99': 200.0, 'max': 400, 'products_with_1_review': 91944, 'products_with_le_5_reviews': 198458, 'products_with_100plus_reviews': 18047, 'products_with_200plus_reviews': 9734, 'products_with_1000plus_reviews': 0}`
- **parquet_file_size_mb**: `260.01`

## CLEAN
- **total_records**: `6152858`
- **unique_products**: `331598`
- **unique_seller_codes_excl_nan**: `29213`
- **unique_seller_titles_excl_nan**: `29228`
- **rows_without_seller**: `301968`
- **unique_ids**: `6152858`
- **product_review_distribution**: `{'min': 1, 'p25': 1.0, 'median': 4.0, 'mean': 18.56, 'std': 40.9, 'p75': 13.0, 'p90': 46.0, 'p99': 200.0, 'max': 233, 'products_with_1_review': 91980, 'products_with_le_5_reviews': 198490, 'products_with_100plus_reviews': 18040, 'products_with_200plus_reviews': 9698, 'products_with_1000plus_reviews': 0}`
- **parquet_file_size_mb**: `258.34`
