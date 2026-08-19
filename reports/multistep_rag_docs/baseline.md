# Target GUI multi-step baseline

- created_at: `2026-08-19T10:46:16.669243+00:00`
- model: `Qwen/Qwen2.5-VL-7B-Instruct`
- prior_knowledge: **rag (docs)**
- dataset: `C:/Proga/GUI2Model/data/target_app/eval/tasks.json`
- base_url: `http://localhost:7770`
- tasks: 10
- rag_config: `{"type": "rag", "mode": "docs", "query_mode": "hybrid", "max_chars": 6000, "stores_root": "C:/Proga/GUI2Model/rag/stores"}`

## Summary

| metric | value |
|---|---:|
| success_rate | 40.0% |
| n_success | 4 / 10 |
| avg_steps | 2.4 |
| avg_expected_steps | 2.3 |
| avg_steps_delta | +0.1 |
| avg_steps (success only) | 1.2 |
| total_tokens | 195161 |
| avg_tokens | 19516.1 |

## Per task

| id | success | steps | expected | delta | tokens |
|---|---|---:|---:|---:|---:|
| eval_search_shirt_results | no | 3 | 3 | +0 | 26059 |
| eval_open_create_account | yes | 1 | 1 | +0 | 8579 |
| eval_forgot_password_from_login | yes | 1 | 1 | +0 | 6730 |
| eval_browse_video_games | yes | 1 | 1 | +0 | 8581 |
| eval_product_add_wishlist | no | 4 | 2 | +2 | 28819 |
| eval_empty_cart_from_home | no | 3 | 2 | +1 | 26084 |
| eval_advanced_search_fill_name | no | 3 | 5 | -2 | 26074 |
| eval_login_then_address_book | no | 3 | 4 | -1 | 20608 |
| eval_category_to_headphones | yes | 2 | 1 | +1 | 17374 |
| eval_minicart_to_full_cart | no | 3 | 3 | +0 | 26253 |
